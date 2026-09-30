//! The streaming, cancellable decode loop.
//!
//! This is the internal streaming API the engine is built around (story 7156). The backend-neutral
//! `core-llm` contract (story 7154) is **extracted from this working loop** — built concrete first,
//! not designed in a vacuum. The loop is model-agnostic: it drives anything implementing [`Decode`]
//! (the Llama decoder today, Qwen3 / BYO architectures later), emitting a [`StreamEvent`] per token
//! through a callback.
//!
//! Cancellation follows the established contract: a request that is *already cancelled* before any
//! work returns the typed [`Error::Canceled`]; a cancel that trips
//! *mid-stream* stops promptly and returns the partial output marked
//! [`FinishReason::Cancelled`].

use std::time::Instant;

use core_llm::GenerationTimings;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

use super::BufferRelease;
use crate::error::{Error, Result};
use crate::primitives::input_ids;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::kv_cache::{ContiguousKvCache, KV_BLOCK_TOKENS};
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};

/// A decoder the streaming loop can drive: it makes its own cache and produces last-position logits.
pub trait Decode {
    /// A fresh KV cache sized for this decoder.
    fn make_cache(&self) -> Box<dyn KvCache>;

    /// One forward step over `input_ids` (`[batch, seq]`) returning last-position logits
    /// `[batch, vocab]`. `offset` is the RoPE offset (positions already cached).
    fn step(&self, input_ids: &Array, cache: &mut dyn KvCache, offset: i32) -> Result<Array>;
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop / EOS token was sampled.
    StopToken,
    /// The `max_new_tokens` budget was reached.
    MaxTokens,
    /// Cancellation tripped mid-stream.
    Cancelled,
    /// A host stop condition halted generation after emitting a token — e.g. a request `stop`
    /// string was detected in the decoded text (see the `should_stop` predicate on
    /// [`generate_with`]). Distinct from [`StopToken`](FinishReason::StopToken) (an EOS *id*) but
    /// maps to the same contract `Stop` finish reason.
    Stopped,
}

/// An event emitted as decoding proceeds.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamEvent {
    /// A newly generated token. `step` is 0-based over generated tokens.
    Token {
        /// The sampled token id.
        id: i32,
        /// Index of this token among the generated tokens.
        step: usize,
    },
    /// Terminal event: generation finished.
    Done {
        /// Why it stopped.
        reason: FinishReason,
        /// How many tokens were generated.
        generated: usize,
    },
}

/// Generation parameters.
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    /// Maximum new tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling knobs.
    pub sampling: SamplingParams,
    /// RNG seed; `None` ⇒ a fresh per-call seed (non-reproducible).
    pub seed: Option<u64>,
    /// Token ids that stop generation when sampled (EOS / EOT / …). The stop token is excluded
    /// from the output.
    pub stop_tokens: Vec<i32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            sampling: SamplingParams::default(),
            seed: None,
            stop_tokens: Vec::new(),
        }
    }
}

/// The result of a generation run.
#[derive(Clone, Debug)]
pub struct GenerationOutput {
    /// Generated token ids (excludes the prompt and any stop token).
    pub tokens: Vec<i32>,
    /// Why generation stopped.
    pub finish_reason: FinishReason,
}

/// A generation result whose synchronized phase timer remains live until provider-side stream
/// processing has completed. The provider finishes the timer after detokenization, stop handling,
/// and the terminal callback so `decode` includes the complete stream-dispatch path.
pub(crate) struct TimedGenerationOutput {
    pub(crate) output: GenerationOutput,
    pub(crate) timer: GenerationTimer,
}

/// Two-phase timer with an explicit accelerator synchronization boundary between prefill and
/// decode. Keeping this stateful prevents a caller from accidentally measuring lazy MLX graph
/// submission as completed prefill work.
pub(crate) struct GenerationTimer {
    prefill_started: Instant,
    prefill: Option<std::time::Duration>,
    decode_started: Option<Instant>,
}

impl GenerationTimer {
    pub(crate) fn start() -> Self {
        Self::start_at(Instant::now())
    }

    pub(crate) fn start_at(prefill_started: Instant) -> Self {
        Self {
            prefill_started,
            prefill: None,
            decode_started: None,
        }
    }

    /// Evaluate every prefill result/cache sentinel before closing the prefill phase.
    pub(crate) fn finish_prefill<'a>(
        &mut self,
        arrays: impl IntoIterator<Item = &'a Array>,
    ) -> Result<()> {
        self.finish_prefill_after(|| Ok(eval(arrays)?))
    }

    fn finish_prefill_after(&mut self, synchronize: impl FnOnce() -> Result<()>) -> Result<()> {
        synchronize()?;
        self.prefill = Some(self.prefill_started.elapsed());
        self.decode_started = Some(Instant::now());
        Ok(())
    }

    /// Finish after the provider has dispatched its terminal stream event.
    pub(crate) fn finish(self) -> GenerationTimings {
        GenerationTimings {
            prefill: self
                .prefill
                .expect("prefill must be synchronized before decode starts"),
            decode: self
                .decode_started
                .expect("prefill must be synchronized before decode starts")
                .elapsed(),
        }
    }
}

/// A per-step logit constraint (e.g. JSON grammar). Before each token the loop asks for the
/// [`ConstraintMask::allowed`] mask (passed to the sampler so disallowed ids are forced to `-inf`),
/// and after a token is chosen it calls [`ConstraintMask::accept`]. The engine owns no grammar
/// policy — `core_llm::JsonConstraint` is one implementation behind this seam.
pub trait ConstraintMask {
    /// The per-vocab allow mask for the current step.
    fn allowed(&mut self) -> &[bool];
    /// Advance the constraint after `token` is chosen.
    fn accept(&mut self, token: i32);
}

/// Stream tokens from `decoder`, starting from `prompt_ids`, emitting a [`StreamEvent`] per token
/// through `on_event`. Unconstrained convenience wrapper over [`generate_with`].
pub fn generate(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<GenerationOutput> {
    generate_with(decoder, prompt_ids, config, cancel, on_event, None, None)
}

/// Like [`generate`], with an optional per-step [`ConstraintMask`] (structured-output decoding) and
/// an optional `should_stop` predicate.
///
/// `should_stop` is checked after each token is emitted and counted; returning `true` halts
/// generation with [`FinishReason::Stopped`]. It is the seam the provider uses to honor request
/// `stop` strings — the predicate inspects state the provider's detokenizing `on_event` maintains
/// (a `core_llm::StopMatcher`), which is why stop-string matching lives in the text/detok layer and
/// not in this token-id loop.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference runs; otherwise runs
/// to a stop token, the token budget, a mid-stream cancel, or a tripped `should_stop`, returning the
/// generated tokens.
pub fn generate_with(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    generate_with_observer(
        decoder,
        prompt_ids,
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
        None,
    )
}

/// Internal campaign-only variant.  `None` preserves the ordinary zero-overhead path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_observer(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    mut observer: Option<&mut dyn crate::campaign::Observer>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        if let Some(observer) = observer.as_deref_mut() {
            observer.phase("cancellation-cleanup");
        }
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate: empty prompt".into()));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let output = {
        let mut cache = decoder.make_cache();
        let mut observed_cache = ObservedCache::default();
        // Prefill the whole prompt at offset 0; logits are for the last prompt position.  The
        // observation is deliberately after dispatch so a sampler sees the actual prefill peak.
        let prompt = input_ids(prompt_ids);
        let logits = decoder.step(&prompt, cache.as_mut(), 0)?;
        if let Some(observer) = observer.as_deref_mut() {
            // This host read is campaign-only.  Normal generation does not materialize logits or
            // pay this synchronization cost; the receipt producer needs actual product logits for
            // its independent fp32/reference-quality calculation.
            let values = logits.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
            observer.logits("prefill", &values);
        }
        if let Some(observer) = observer.as_deref_mut() {
            observer.phase("prefill-peak");
        }
        observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;
        let output = decode_loop(
            decoder,
            cache.as_mut(),
            logits,
            rng,
            prompt_ids.to_vec(),
            config,
            cancel,
            on_event,
            constraint,
            should_stop,
            &mut observer,
        )?;
        if let Some(observer) = observer.as_deref_mut() {
            observer.phase("decode-steady");
        }
        observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;
        observe_packed_evidence(cache.as_ref(), &mut observer);
        if matches!(output.finish_reason, FinishReason::Cancelled) {
            if let Some(observer) = observer.as_deref_mut() {
                observer.phase("cancellation-cleanup");
            }
        }
        cache.reset()?;
        observe_cache_events(cache.as_mut(), &mut observed_cache, &mut observer)?;
        output
    };
    Ok(output)
}

/// Campaign observation state for one decoder cache.
#[derive(Default)]
pub(super) struct ObservedCache {
    /// Dense cache events already forwarded.
    dense_events: usize,
    /// Physical packed bytes of the last forwarded packed snapshot, still owned by the cache.
    packed_live_bytes: Option<u64>,
    /// Full-cache dense reconstructions already forwarded as explicit materializations.
    dequantizations: usize,
}

/// Export only cache-owned byte observations.  Unsupported cache implementations produce no
/// synthetic events; the receipt producer must then record an explicit fallback rather than
/// inventing an allocation total.
pub(super) fn observe_cache_events(
    cache: &mut dyn KvCache,
    state: &mut ObservedCache,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) -> Result<()> {
    let Some(observer) = observer.as_deref_mut() else {
        return Ok(());
    };
    if let Some(evidence) = cache.packed_evidence() {
        // Compressed-arm cache (SC-20676): the live compressed representation is the persistent
        // KV. A transition that rebuilt the whole history as dense K/V is an explicit full-cache
        // materialization, never a quiet change of the persistent total.
        let storage = cache.compressed_storage()?;
        if let Some(bytes) = state.packed_live_bytes.take() {
            if storage.is_none() {
                observer.release_event("cache_release", "cache", bytes);
            } else {
                state.packed_live_bytes = Some(bytes);
            }
        }
        if let Some(dense) = cache.compressed_dense_fallback() {
            observe_contiguous(dense, &mut state.dense_events, observer)?;
        }
        if evidence.full_cache_dequantizations > state.dequantizations {
            let bytes = cache
                .compressed_dense_fallback()
                .map(ContiguousKvCache::retained_snapshot)
                .transpose()?
                .flatten()
                .map(|(bytes, _, _, _)| bytes)
                .ok_or_else(|| {
                    Error::Msg(
                        "compressed cache recorded a dense reconstruction without dense data"
                            .into(),
                    )
                })?;
            observer.dense_reconstruction(bytes);
            state.dequantizations = evidence.full_cache_dequantizations;
        }
        if let Some(storage) = storage {
            // Capacity is the dense-equivalent allocation for the live tokens, so the receipt's
            // theoretical dense KV denominator describes the same geometry as a dense row.
            let capacity = storage
                .tokens
                .div_ceil(KV_BLOCK_TOKENS as u64)
                .saturating_mul(KV_BLOCK_TOKENS as u64);
            observer.cache_snapshot(
                storage.device_bytes(),
                storage.tokens,
                capacity,
                storage.element_bytes,
            );
            observer.compressed_storage(&storage);
            state.packed_live_bytes = Some(storage.device_bytes());
        }
        return Ok(());
    }
    let Some(cache) = cache.as_any_mut().downcast_ref::<ContiguousKvCache>() else {
        return Ok(());
    };
    observe_contiguous(cache, &mut state.dense_events, observer)
}

fn observe_contiguous(
    cache: &ContiguousKvCache,
    seen: &mut usize,
    observer: &mut dyn crate::campaign::Observer,
) -> Result<()> {
    if *seen > cache.events().len() {
        // A packed cache replaced its dense fallback instance; its events start afresh.
        *seen = 0;
    }
    for event in cache.events().iter().skip(*seen) {
        if event.role == "cache" && event.lifetime == "persistent" {
            continue;
        } else if event.lifetime == "released" {
            observer.release_event(event.operation, event.role, event.bytes);
        } else {
            observer.allocation_event(event.operation, event.role, event.lifetime, event.bytes);
        }
    }
    *seen = cache.events().len();
    if let Some((bytes, tokens, capacity, element_bytes)) = cache.retained_snapshot()? {
        observer.cache_snapshot(bytes, tokens, capacity, element_bytes);
    }
    Ok(())
}

/// Forward a compressed cache's immutable model-boundary evidence exactly once per decoder cache,
/// after decode and before reset, so fused/fallback counts are never double counted.
pub(super) fn observe_packed_evidence(
    cache: &dyn KvCache,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) {
    if let (Some(observer), Some(evidence)) = (observer.as_deref_mut(), cache.packed_evidence()) {
        observer.packed_cache_evidence(&evidence);
    }
}

/// Synchronized two-phase variant of [`generate_with`]. Tokenization and template rendering happen
/// before this function; the returned timer deliberately remains live for provider-side stream
/// processing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_with_timings(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<TimedGenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_with_timings: empty prompt".into()));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let mut cache = decoder.make_cache();
    let prompt = input_ids(prompt_ids);
    let mut timer = GenerationTimer::start();
    let logits = decoder.step(&prompt, cache.as_mut(), 0)?;
    timer.finish_prefill([&logits])?;
    let output = decode_loop(
        decoder,
        cache.as_mut(),
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
        &mut None,
    )?;
    Ok(TimedGenerationOutput { output, timer })
}

/// Like [`generate`], but driving a **caller-provided** KV cache that may already hold a prefix
/// (e.g. a [`PagedKvCache`](crate::primitives::PagedKvCache) seeded with shared blocks). Prefills
/// only `prompt_ids[cache.offset()..]` at that offset, then decodes. The cache is borrowed (not
/// consumed) so the caller can inspect or seed siblings from it afterward.
///
/// `cache.offset()` must be `< prompt_ids.len()` (there must be at least one token to prefill).
/// Returns [`Error::Canceled`] on an already-set cancel.
pub fn generate_with_cache(
    decoder: &dyn Decode,
    prompt_ids: &[i32],
    cache: &mut dyn KvCache,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_with_cache: empty prompt".into()));
    }
    let offset = cache.offset() as usize;
    if offset >= prompt_ids.len() {
        return Err(Error::Msg(format!(
            "generate_with_cache: cache offset {offset} leaves no prompt suffix to prefill \
             (prompt len {})",
            prompt_ids.len()
        )));
    }

    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let suffix = input_ids(&prompt_ids[offset..]);
    let logits = decoder.step(&suffix, cache, offset as i32)?;

    decode_loop(
        decoder,
        cache,
        logits,
        rng,
        prompt_ids.to_vec(),
        config,
        cancel,
        on_event,
        None,
        None,
        &mut None,
    )
}

/// Drive generation from a **caller-supplied prefill**: a `cache` already advanced over the prompt
/// and its last-position `first_logits`. The multimodal path prefills outside this loop — it embeds
/// the prompt, splices the encoder's image features at the image-token rows, and runs the decoder
/// with interleaved M-RoPE — then hands the result here to sample + stream the continuation. `history`
/// seeds the repetition-penalty window (the full, image-token-expanded prompt ids).
///
/// The `decoder` drives the **decode steps** only (the prompt is already cached); for the Qwen3.6
/// multimodal path that decoder shifts the RoPE offset by `mrope_delta` so post-image text positions
/// continue correctly. Returns [`Error::Canceled`] on an already-set cancel.
#[allow(clippy::too_many_arguments)]
pub fn generate_from_prefill(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    first_logits: Array,
    history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    decode_loop(
        decoder,
        cache,
        first_logits,
        rng,
        history,
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
        &mut None,
    )
}

/// [`generate_from_prefill`] with a campaign observer attached to the decode loop (used for
/// teacher-forced greedy agreement; product generation never attaches one here).
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_from_prefill_observed(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    first_logits: Array,
    history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    observer: &mut dyn crate::campaign::Observer,
) -> Result<GenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    decode_loop(
        decoder,
        cache,
        first_logits,
        rng,
        history,
        config,
        cancel,
        on_event,
        None,
        None,
        &mut Some(observer),
    )
}

/// Synchronized variant of [`generate_from_prefill`] for a prefill whose conditioning began at
/// `prefill_started`. Qwen-VL starts this clock before image/video encoding and fusion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_from_prefill_with_timings(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    first_logits: Array,
    history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    prefill_started: Instant,
) -> Result<TimedGenerationOutput> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    let mut timer = GenerationTimer::start_at(prefill_started);
    timer.finish_prefill([&first_logits])?;
    let rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let output = decode_loop(
        decoder,
        cache,
        first_logits,
        rng,
        history,
        config,
        cancel,
        on_event,
        constraint,
        should_stop,
        &mut None,
    )?;
    Ok(TimedGenerationOutput { output, timer })
}

/// The token-by-token decode loop shared by [`generate_with`] and the prefix-cached path
/// ([`crate::decode::generate_cached`]): given the prefill `logits`, an RNG, the seeded `history`
/// (the prompt — the repetition-penalty window), and a `cache` already positioned past the prompt,
/// sample, emit, and step until a stop token, the budget, or a mid-stream cancel.
///
/// The two entry points differ only in how the cache + first `logits` are produced (cold prefill vs.
/// shared-prefix reuse); the loop is identical, so a cached run is token-for-token the same as a cold
/// one for the same prompt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_loop(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    mut logits: Array,
    mut rng: SplitMix64,
    mut history: Vec<i32>,
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn ConstraintMask>,
    should_stop: Option<&dyn Fn() -> bool>,
    observer: &mut Option<&mut dyn crate::campaign::Observer>,
) -> Result<GenerationOutput> {
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;
    // Every caller hands us `logits` straight from its prefill — still lazy. The first `sample`
    // below evaluates that graph, but the prefill `logits` array itself stays live until step 0
    // reassigns the binding at the bottom of this loop; the post-prefill release therefore rides
    // the first `release.advance`, which sits just after that reassignment.
    let mut release = BufferRelease::new();

    for step in 0..config.max_new_tokens {
        // Pulling logits to host for sampling forces a graph eval each step, so this check is
        // genuinely effective despite MLX's lazy evaluation.
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        // Campaign teacher forcing (never set in product use): the forced stream decides what is
        // fed back, and its end stops generation before another position is sampled.
        let forced = match observer
            .as_deref_mut()
            .map(|o| o.teacher_forced_token(step))
        {
            Some(crate::campaign::TeacherForcing::Token(token)) => Some(token),
            Some(crate::campaign::TeacherForcing::Exhausted) => {
                finish = FinishReason::Stopped;
                break;
            }
            Some(crate::campaign::TeacherForcing::Off) | None => None,
        };

        // Apply the constraint mask (if any) for this step, then sample. The mask borrow is scoped
        // so the constraint is free to be advanced again below.
        let next = {
            let mask = constraint.as_mut().map(|c| c.allowed());
            sample(&logits, &history, &config.sampling, &mut rng, mask)?
        };

        // The model's own choice is what the observer records, forced or not.
        if let Some(observer) = observer.as_deref_mut() {
            observer.token_probability("decode", next, selected_token_probability(&logits, next)?);
        }
        let next = forced.unwrap_or(next);

        if config.stop_tokens.contains(&next) {
            finish = FinishReason::StopToken;
            break;
        }

        if let Some(c) = &mut constraint {
            c.accept(next);
        }

        on_event(StreamEvent::Token { id: next, step });
        if step == 0 {
            if let Some(observer) = observer.as_deref_mut() {
                observer.phase("first-token");
            }
        }
        generated.push(next);
        history.push(next);

        // A host stop condition (e.g. a request `stop` string detected by `on_event`'s detokenizer)
        // halts here, after the triggering token is counted. The next decode step is skipped.
        if should_stop.is_some_and(|f| f()) {
            finish = FinishReason::Stopped;
            break;
        }

        if step + 1 == config.max_new_tokens {
            break; // budget reached; finish stays MaxTokens
        }

        // Feed the new token back; its absolute position is the current cache length.
        let offset = cache.offset();
        let tok = input_ids(&[next]);
        // This reassignment drops the previous `logits`; on step 0 that is the prefill's
        // prompt-length logits, so the release below is the first moment they are freeable.
        logits = decoder.step(&tok, cache, offset)?;
        release.advance(1);
    }

    on_event(StreamEvent::Done {
        reason: finish,
        generated: generated.len(),
    });
    Ok(GenerationOutput {
        tokens: generated,
        finish_reason: finish,
    })
}

/// A fixed-length steady-decode measurement (SC-20671): `tokens` greedy tokens after a prefill,
/// generated with every stop token ignored, and the synchronized duration of all but the first.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ForcedDecode {
    /// Every generated id, including the first (which closes the prefill and is not timed).
    pub(crate) tokens: Vec<i32>,
    /// Tokens inside the timed window: every generated token after the first.
    pub(crate) timed_tokens: u64,
    /// Milliseconds from the first token's GPU completion to the last token's GPU completion.
    pub(crate) decode_ms: f64,
    /// Generated ids that are stop tokens and were decoded through instead of ending generation.
    pub(crate) forced_stop_tokens: u64,
}

/// Prefill `prompt_ids` into `cache`, then greedily decode exactly `tokens` ids, feeding each back
/// even when it is a stop token (`stop_tokens` are only counted, never obeyed), so every
/// measurement covers the same number of decode steps whatever the model would emit.
///
/// Timing boundary: each token is stamped immediately after the product sampler returned its id,
/// and that host readback cannot complete before the GPU has finished the step's logits — the same
/// synchronized boundary for every cache representation. `boundary` is called with the sampled
/// logits right before each stamp (a test seam that proves the array is materialized there). The
/// timed window opens at the first token's stamp, so prefill and time-to-first-token are excluded.
/// The loop body mirrors [`decode_loop`] (product sampler, KV feed, buffer-release cadence), minus
/// stop handling, constraints, and the stream callback.
pub(crate) fn forced_greedy_decode(
    decoder: &dyn Decode,
    cache: &mut dyn KvCache,
    prompt_ids: &[i32],
    tokens: usize,
    stop_tokens: &[i32],
    boundary: &mut dyn FnMut(&Array),
) -> Result<ForcedDecode> {
    if prompt_ids.is_empty() || tokens < 2 {
        return Err(Error::Msg(
            "forced steady decode needs a prompt and at least two tokens".into(),
        ));
    }
    let greedy = SamplingParams::default();
    let mut rng = SplitMix64::new(0);
    let mut history = prompt_ids.to_vec();
    let mut release = BufferRelease::new();
    let mut generated: Vec<i32> = Vec::with_capacity(tokens);
    let mut logits = decoder.step(&input_ids(prompt_ids), cache, 0)?;
    let mut opened: Option<Instant> = None;
    let mut closed: Option<Instant> = None;
    for step in 0..tokens {
        if step > 0 {
            let previous = generated[step - 1];
            let offset = cache.offset();
            logits = decoder.step(&input_ids(&[previous]), cache, offset)?;
            release.advance(1);
        }
        let next = sample(&logits, &history, &greedy, &mut rng, None)?;
        boundary(&logits);
        let stamped = Instant::now();
        if opened.is_none() {
            opened = Some(stamped);
        } else {
            closed = Some(stamped);
        }
        generated.push(next);
        history.push(next);
    }
    let (Some(opened), Some(closed)) = (opened, closed) else {
        return Err(Error::Msg(
            "forced steady decode closed no timed window".into(),
        ));
    };
    let decode_ms = closed.duration_since(opened).as_secs_f64() * 1_000.0;
    if generated.len() != tokens || !decode_ms.is_finite() || decode_ms <= 0.0 {
        return Err(Error::Msg(
            "forced steady decode did not produce its fixed length in positive time".into(),
        ));
    }
    let forced_stop_tokens = generated
        .iter()
        .filter(|token| stop_tokens.contains(token))
        .count() as u64;
    Ok(ForcedDecode {
        timed_tokens: (tokens - 1) as u64,
        tokens: generated,
        decode_ms,
        forced_stop_tokens,
    })
}

/// Product-owned selected-token probability.  This runs only while an evidence observer is
/// attached, after the exact production logits have been produced and before sampling mutates the
/// decode state.  The max-shifted reduction is deliberately finite/checked so malformed logits
/// cannot be turned into a plausible campaign quality value.
fn selected_token_probability(logits: &Array, token: i32) -> Result<f64> {
    if token < 0 {
        return Err(Error::Msg("negative sampled token id".into()));
    }
    let logits_f32 = logits.as_dtype(Dtype::Float32)?;
    let values = logits_f32.as_slice::<f32>();
    let token = token as usize;
    if token >= values.len() || values.iter().any(|value| !value.is_finite()) {
        return Err(Error::Msg(
            "invalid product logits for campaign observation".into(),
        ));
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator = values
        .iter()
        .map(|value| f64::from((*value - maximum).exp()))
        .sum::<f64>();
    let numerator = f64::from((values[token] - maximum).exp());
    let probability = numerator / denominator;
    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(Error::Msg("invalid selected-token probability".into()));
    }
    Ok(probability)
}

/// A non-reproducible seed for `GenerationConfig::seed == None`.
pub(crate) fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

pub use super::cancel::CancelFlag;

/// Test-only MLX evaluation probe: whether `array` is materialized (mlx-c's internal
/// `_mlx_array_is_available`, linked through mlx-rs). A timer whose region ends while this is still
/// false measured lazy graph construction, not GPU completion.
#[cfg(test)]
pub(crate) fn mlx_array_is_available(array: &Array) -> bool {
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RawMlxArray {
        ctx: *mut std::ffi::c_void,
    }
    extern "C" {
        fn _mlx_array_is_available(res: *mut bool, arr: RawMlxArray) -> std::ffi::c_int;
    }
    let handle = array.as_ptr();
    assert_eq!(
        std::mem::size_of_val(&handle),
        std::mem::size_of::<RawMlxArray>()
    );
    // SAFETY: mlx-c's `mlx_array` is `struct { void* ctx; }`, mirrored by `RawMlxArray` (sizes
    // asserted above); the call only reads the handle, which `array` keeps alive.
    let raw: RawMlxArray = unsafe { std::mem::transmute_copy(&handle) };
    let mut available = false;
    // SAFETY: `available` is a valid out-pointer and `raw` a live array handle.
    let status = unsafe { _mlx_array_is_available(&mut available, raw) };
    assert_eq!(status, 0, "mlx-c availability probe failed");
    available
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedDecoder;

    impl Decode for FixedDecoder {
        fn make_cache(&self) -> Box<dyn KvCache> {
            Box::new(ContiguousKvCache::new(0))
        }

        fn step(
            &self,
            _input_ids: &Array,
            _cache: &mut dyn KvCache,
            _offset: i32,
        ) -> Result<Array> {
            Ok(Array::from_slice(&[0.0_f32, 1.0, -1.0], &[1, 3]))
        }
    }

    #[derive(Default)]
    struct DecodeObserver {
        phases: Vec<&'static str>,
        logits: usize,
        probabilities: Vec<(i32, f64)>,
        cache_snapshots: Vec<(u64, u64, u64, u64)>,
    }

    impl crate::campaign::Observer for DecodeObserver {
        fn phase(&mut self, name: &'static str) {
            self.phases.push(name);
        }

        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}

        fn logits(&mut self, stage: &'static str, values: &[f32]) {
            assert_eq!(stage, "prefill");
            assert_eq!(values.len(), 3);
            self.logits += 1;
        }

        fn token_probability(&mut self, stage: &'static str, token: i32, probability: f64) {
            assert_eq!(stage, "decode");
            self.probabilities.push((token, probability));
        }

        fn cache_snapshot(&mut self, bytes: u64, tokens: u64, capacity: u64, element_bytes: u64) {
            self.cache_snapshots
                .push((bytes, tokens, capacity, element_bytes));
        }
    }

    #[test]
    fn observed_generation_forwards_decode_probability_and_first_token_phase() {
        let mut observer = DecodeObserver::default();
        let output = generate_with_observer(
            &FixedDecoder,
            &[7],
            &GenerationConfig {
                max_new_tokens: 1,
                seed: Some(0),
                ..Default::default()
            },
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            Some(&mut observer),
        )
        .unwrap();
        assert_eq!(output.tokens.len(), 1);
        assert_eq!(observer.logits, 1);
        assert_eq!(observer.probabilities.len(), 1);
        assert!(observer.probabilities[0].1.is_finite());
        assert_eq!(
            observer.phases,
            vec!["prefill-peak", "first-token", "decode-steady"]
        );
    }

    /// Teacher forcing feeds the forced stream, records the model's own greedy choice at every
    /// position, and stops when the forced stream ends (before sampling another position).
    #[test]
    fn teacher_forced_generation_feeds_the_stream_and_records_own_choices() {
        struct Forcing {
            forced: Vec<i32>,
            choices: Vec<i32>,
        }
        impl crate::campaign::Observer for Forcing {
            fn phase(&mut self, _name: &'static str) {}
            fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
            fn teacher_forced_token(&mut self, step: usize) -> crate::campaign::TeacherForcing {
                self.forced
                    .get(step)
                    .map_or(crate::campaign::TeacherForcing::Exhausted, |token| {
                        crate::campaign::TeacherForcing::Token(*token)
                    })
            }
            fn token_probability(&mut self, _stage: &'static str, token: i32, _probability: f64) {
                self.choices.push(token);
            }
        }
        let run = |forced: Vec<i32>, stop_tokens: Vec<i32>| {
            let mut observer = Forcing {
                forced,
                choices: Vec::new(),
            };
            let output = generate_with_observer(
                &FixedDecoder,
                &[7],
                &GenerationConfig {
                    max_new_tokens: 10,
                    seed: Some(0),
                    stop_tokens,
                    ..Default::default()
                },
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                Some(&mut observer),
            )
            .unwrap();
            (output, observer.choices)
        };
        // FixedDecoder's greedy choice is always token 1.
        let (output, choices) = run(vec![2, 0, 2], Vec::new());
        assert_eq!(output.tokens, vec![2, 0, 2]);
        assert_eq!(choices, vec![1, 1, 1]);
        assert_eq!(output.finish_reason, FinishReason::Stopped);
        // A forced stop token ends generation exactly as the reference stream did.
        let (output, choices) = run(vec![2, 0, 2], vec![0]);
        assert_eq!(output.tokens, vec![2]);
        assert_eq!(choices, vec![1, 1]);
        assert_eq!(output.finish_reason, FinishReason::StopToken);
    }

    /// Returns lazy logits (an unevaluated multiply) and keeps a handle to every array it returned,
    /// so an observer can probe whether each had reached GPU completion at a phase timestamp.
    struct LazyDecoder {
        produced: std::rc::Rc<std::cell::RefCell<Vec<Array>>>,
    }

    impl Decode for LazyDecoder {
        fn make_cache(&self) -> Box<dyn KvCache> {
            Box::new(ContiguousKvCache::new(0))
        }

        fn step(
            &self,
            _input_ids: &Array,
            _cache: &mut dyn KvCache,
            _offset: i32,
        ) -> Result<Array> {
            let base = Array::from_slice(&[0.0_f32, 1.0, -1.0], &[1, 3]);
            let logits = base.multiply(Array::from_f32(2.0))?;
            assert!(
                !mlx_array_is_available(&logits),
                "the probe must see lazy logits"
            );
            self.produced.borrow_mut().push(logits.clone());
            Ok(logits)
        }
    }

    /// At each timing phase, whether every logits array produced so far was materialized.
    struct BoundaryProbe {
        produced: std::rc::Rc<std::cell::RefCell<Vec<Array>>>,
        phases: Vec<(&'static str, usize, bool)>,
    }

    impl crate::campaign::Observer for BoundaryProbe {
        fn phase(&mut self, name: &'static str) {
            let produced = self.produced.borrow();
            let complete = produced.iter().all(mlx_array_is_available);
            self.phases.push((name, produced.len(), complete));
        }

        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
    }

    /// SC-20671 timings are phase deltas (`prefill-peak`, `first-token`, `decode-steady`). Every
    /// phase must be stamped only after the logits it closes over reached GPU completion — the
    /// same boundary for the dense and compressed arms, which share this decode loop — so a lazy
    /// cache or reader cannot turn a timing into host dispatch time.
    #[test]
    fn timing_phases_are_stamped_after_the_sampled_logits_complete() {
        let produced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let decoder = LazyDecoder {
            produced: produced.clone(),
        };
        let mut probe = BoundaryProbe {
            produced,
            phases: Vec::new(),
        };
        let output = generate_with_observer(
            &decoder,
            &[7],
            &GenerationConfig {
                max_new_tokens: 3,
                seed: Some(0),
                ..Default::default()
            },
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            Some(&mut probe),
        )
        .unwrap();
        assert_eq!(output.tokens.len(), 3);
        assert_eq!(
            probe.phases,
            vec![
                ("prefill-peak", 1, true),
                ("first-token", 1, true),
                ("decode-steady", 3, true),
            ]
        );
    }

    /// SC-20671 steady decode is a fixed-length measurement: a sampled stop token is decoded
    /// through (and counted), never a reason to end early and shrink the timed window.
    #[test]
    fn forced_steady_decode_ignores_stop_tokens_and_produces_exactly_n() {
        let mut cache = FixedDecoder.make_cache();
        // FixedDecoder's greedy token is 1; declare it the stop token.
        let measured =
            forced_greedy_decode(&FixedDecoder, cache.as_mut(), &[7, 8], 6, &[1], &mut |_| {})
                .unwrap();
        assert_eq!(measured.tokens, vec![1; 6]);
        assert_eq!(measured.timed_tokens, 5, "the first token is not timed");
        assert_eq!(measured.forced_stop_tokens, 6);
        assert!(
            forced_greedy_decode(&FixedDecoder, cache.as_mut(), &[7], 1, &[], &mut |_| {}).is_err(),
            "one token has no steady window"
        );
    }

    /// Every steady-decode stamp is taken after the logits it closes over reached GPU completion,
    /// so a lazy-only timer (stamping graph construction) cannot pass as decode throughput.
    #[test]
    fn forced_steady_decode_stamps_only_after_the_sampled_logits_complete() {
        let produced = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let decoder = LazyDecoder {
            produced: produced.clone(),
        };
        let mut cache = decoder.make_cache();
        let mut boundaries = Vec::new();
        let measured =
            forced_greedy_decode(&decoder, cache.as_mut(), &[7], 4, &[], &mut |logits| {
                let produced = produced.borrow();
                boundaries.push((
                    produced.len(),
                    mlx_array_is_available(logits) && produced.iter().all(mlx_array_is_available),
                ));
            })
            .unwrap();
        assert_eq!(measured.tokens.len(), 4);
        assert_eq!(boundaries, vec![(1, true), (2, true), (3, true), (4, true)]);
    }

    #[test]
    fn unchanged_live_cache_is_snapshotted_at_each_observed_phase() {
        let mut cache = ContiguousKvCache::new(1);
        let key = Array::from_slice(&[1.0_f32, 2.0], &[1, 1, 1, 2]);
        cache.update(0, &key, &key).unwrap();
        let mut seen = ObservedCache::default();
        let mut observer = DecodeObserver::default();
        let mut attached: Option<&mut dyn crate::campaign::Observer> = Some(&mut observer);
        observe_cache_events(&mut cache, &mut seen, &mut attached).unwrap();
        observe_cache_events(&mut cache, &mut seen, &mut attached).unwrap();
        assert_eq!(
            observer.cache_snapshots,
            vec![(4096, 1, 256, 4), (4096, 1, 256, 4)]
        );
    }

    #[derive(Default)]
    struct CompressedCapture {
        snapshots: Vec<(u64, u64, u64, u64)>,
        releases: Vec<u64>,
        reconstructions: Vec<u64>,
        storages: Vec<crate::primitives::CompressedCacheStorage>,
        evidence: Vec<crate::primitives::PackedCacheEvidence>,
    }

    impl crate::campaign::Observer for CompressedCapture {
        fn phase(&mut self, _name: &'static str) {}
        fn allocation(&mut self, _role: &'static str, _lifetime: &'static str, _bytes: u64) {}
        fn cache_snapshot(&mut self, bytes: u64, tokens: u64, capacity: u64, element_bytes: u64) {
            self.snapshots
                .push((bytes, tokens, capacity, element_bytes));
        }
        fn release_event(&mut self, _kind: &'static str, _role: &'static str, bytes: u64) {
            self.releases.push(bytes);
        }
        fn dense_reconstruction(&mut self, bytes: u64) {
            self.reconstructions.push(bytes);
        }
        fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
            self.storages.push(*storage);
        }
        fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
            self.evidence.push(evidence.clone());
        }
    }

    /// The compressed arm's persistent KV is the measured packed storage; an explicit dense
    /// transition releases it and is witnessed as a full-cache materialization; reset releases
    /// the dense fallback.
    #[cfg(target_os = "macos")]
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn compressed_cache_observation_reports_packed_storage_and_dense_reconstruction() {
        use crate::primitives::{
            select_decoder_cache_with_reader, CompiledKernelHandle, PackedAttentionMask,
            PackedCacheRequest, PackedMetalKernel, PACKED_METAL_QUANT_GROUP_SIZE,
        };
        let kernel = PackedMetalKernel::new().unwrap();
        let identity = crate::primitives::RetainedPackedKernel::cache_identity(&kernel).to_owned();
        let handle = CompiledKernelHandle::new(std::sync::Arc::new(kernel));
        let request = PackedCacheRequest {
            enabled: true,
            backend: "mlx-metal".into(),
            identity,
            layers: 1,
            batch: 1,
            kv_heads: 1,
            head_dimension: 64,
            group_size: PACKED_METAL_QUANT_GROUP_SIZE,
            query_length: 1,
            has_mask: false,
        };
        let mut cache = select_decoder_cache_with_reader(request, handle).into_cache();
        let values = (0..64).map(|i| (i % 7) as f32 * 0.01).collect::<Vec<_>>();
        let kv = Array::from_slice(&values, &[1, 1, 1, 64])
            .as_dtype(Dtype::Float16)
            .unwrap();
        let q = Array::from_slice(&values, &[1, 1, 1, 64])
            .as_dtype(Dtype::Float16)
            .unwrap();
        assert!(cache
            .try_packed_attention(0, &q, &kv, &kv, PackedAttentionMask::Causal, 0.125, false)
            .unwrap()
            .is_some());
        let mut state = ObservedCache::default();
        let mut capture = CompressedCapture::default();
        let mut attached: Option<&mut dyn crate::campaign::Observer> = Some(&mut capture);
        observe_cache_events(cache.as_mut(), &mut state, &mut attached).unwrap();
        observe_packed_evidence(cache.as_ref(), &mut attached);
        cache
            .prepare_dense_fallback("campaign-test", "forced dense transition")
            .unwrap();
        observe_cache_events(cache.as_mut(), &mut state, &mut attached).unwrap();
        cache.reset().unwrap();
        observe_cache_events(cache.as_mut(), &mut state, &mut attached).unwrap();

        let storage = capture.storages[0];
        assert_eq!(capture.storages.len(), 1);
        assert!(storage.device_code_bytes > 0 && storage.device_metadata_bytes > 0);
        assert_eq!(
            storage.host_payload_bytes, 0,
            "the decoder route's packed K/V is device-resident; no host copy exists"
        );
        assert_eq!(
            capture.snapshots[0],
            (storage.device_bytes(), 1, 256, 2),
            "packed persistent KV is the measured device storage at the dense-equivalent geometry"
        );
        assert!(
            storage.device_bytes() < 2 * 64 * 2 * 256,
            "the block-preallocated packed store stays below one dense f16 K/V block"
        );
        let dense_bytes = capture.snapshots[1].0;
        assert_eq!(capture.reconstructions, vec![dense_bytes]);
        assert_eq!(
            capture.releases,
            vec![storage.device_bytes(), dense_bytes],
            "packed ownership is released at the transition, dense ownership at reset"
        );
        assert_eq!(capture.evidence.len(), 1);
        assert_eq!(capture.evidence[0].accepted_direct_calls, 1);
        assert_eq!(capture.evidence[0].full_cache_dequantizations, 0);
    }
}

#[cfg(test)]
mod timing_tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn prefill_boundary_advances_only_after_synchronization_succeeds() {
        let mut failed = GenerationTimer::start();
        let error = failed
            .finish_prefill_after(|| Err(Error::Msg("sync failed".into())))
            .unwrap_err();
        assert_eq!(error.to_string(), "sync failed");
        assert!(failed.prefill.is_none());
        assert!(failed.decode_started.is_none());

        let synchronized = Cell::new(false);
        let mut completed = GenerationTimer::start();
        completed
            .finish_prefill_after(|| {
                synchronized.set(true);
                Ok(())
            })
            .unwrap();
        assert!(synchronized.get());
        assert!(completed.prefill.is_some());
        assert!(completed.decode_started.is_some());
        let _measured = completed.finish();
    }
}

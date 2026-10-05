//! The autoregressive text-generation runtime (sc-3187) — KV-cache incremental decode, token
//! sampling, and the `_generate_think` think/no-think rollout.
//!
//! SenseNova-U1 is a *generating* LLM, so beyond a forward pass it needs an AR runtime: a prefix is
//! prefilled into a [`KvCache`], then tokens are decoded one at a time —
//! each new token forwarded through the cached backbone at the next temporal position to produce
//! the logits for the token after it. This module ports the reference's three pieces
//! (`modeling_neo_chat.py`):
//!
//! * [`Qwen3Backbone::decode_logits`] — one cached single-token forward → next-token logits (the
//!   inner step of the reference's `self.language_model(input_ids=next_token.unsqueeze(0), …)`).
//! * [`Qwen3Backbone::append_tokens`] — splice a run of known tokens into the cache without
//!   sampling (the reference `_append_text_tokens_to_cache`; e.g. the `\n\n<img>` that follows a
//!   think block).
//! * [`Qwen3Backbone::generate`] — greedy/sampled rollout to an EOS or token budget (the runtime
//!   under `chat`/`answer_question`, sc-3191).
//! * [`Qwen3Backbone::generate_think`] — the `_generate_think` think rollout: greedy-decode a
//!   `<think>…</think>` block, then append `\n\n<img>` to the cache, leaving it primed for image
//!   generation (sc-3187's deliverable for T2I think-mode + interleave).
//!
//! Positions: text tokens advance the temporal axis by one per token (`h = w = 0`), matching the
//! reference, which sets `model.current_index = t_idx` before each step and lets the forward
//! increment it. The understanding path ([`Path::Und`]) drives text decode.
//!
//! Every text rollout here — and the interleave rollout's text segments (`t2i.rs`) — is the shared
//! MLX engine's token-at-a-time loop ([`generate_speculative`] with [`NoProposer`], epic sc-24432
//! E8) over the understanding path (`UndTarget`), drawing each token through `UndSampler` — the
//! rollouts' own draw (the host argmax of a prefix row, the on-device argmax of a decode step, the
//! shared stochastic sampler over the host row) from their own seeded stream — so every stream is
//! token-identical to the pre-engine loops. The pipeline's cancel is bridged onto the engine's
//! flag, so a cancel ends a rollout as the engine's typed `Cancelled` finish.

use std::cell::RefCell;

use mlx_rs::{Array, Dtype};

use mlx_gen::attention::AttentionPlan;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_llm::core_llm::{HostSampleReason, SamplerPath};
use mlx_llm::decode::{
    generate_speculative, EngineOptions, FinishReason, GenerationConfig, LogitsScope,
    NoDraftRollback, NoProposer, Pipelining, SampledToken, SpeculativePrompt, SpeculativeTarget,
    StreamEvent, TargetOutput, TokenSampler,
};
use mlx_llm::primitives::KvCache as _;

// Shared decode sampler (sc-7159): on-device greedy argmax + the unified temperature/top-k/top-p
// sampler + the deterministic SplitMix64. The bespoke think/no-think + dual-path rollout stays here.
use mlx_llm::primitives::sampler::{argmax_device, argmax_host, sample as mll_sample};
use mlx_llm::primitives::{SamplingParams, SplitMix64};

use crate::qwen3::{KvCache, Path, Qwen3Backbone};

/// Map an mlx-llm primitive error onto the gen-core contract error.
fn mll<E: std::fmt::Display>(e: E) -> Error {
    Error::Msg(e.to_string())
}

/// How the next token is chosen from a logits row.
#[derive(Clone, Copy, Debug)]
pub enum Sampler {
    /// Argmax — the reference `_generate_think` rollout and the deterministic chat path.
    Greedy,
    /// Temperature + nucleus (top-p) + top-k sampling. `top_p`/`top_k` of `1.0`/`0` disable that
    /// stage; `temperature` must be `> 0`.
    Sample {
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: u64,
    },
}

impl Sampler {
    /// Pick a token id from a `[vocab]` logits row, advancing `rng` for the stochastic variants.
    ///
    /// Delegates to the shared sampler (sc-7159): greedy is the shared lowest-index [`argmax_host`];
    /// the stochastic variant is the unified [`mll_sample`] (temperature + top-k + nucleus top-p, no
    /// repetition penalty) over the row lifted to an `Array`. The greedy path is bit-identical to the
    /// prior local argmax; the stochastic path is a valid resample from the same shaped distribution
    /// (the shared sampler draws over candidates in index/top-k order rather than the prior
    /// sorted-descending order — no golden pins a stochastic sequence, and every gated rollout, plus
    /// all image-token decoding, is greedy).
    fn pick(&self, logits: &[f32], rng: &mut SplitMix64) -> Result<i32> {
        match *self {
            Sampler::Greedy => Ok(argmax_host(logits)),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => {
                let row = Array::from_slice(logits, &[1, logits.len() as i32]);
                let params = SamplingParams {
                    temperature,
                    top_p,
                    top_k,
                    repetition_penalty: 1.0,
                    repetition_context: 0,
                    presence_penalty: 0.0,
                };
                mll_sample(&row, &[], &params, rng, None).map_err(mll)
            }
        }
    }

    fn seed(&self) -> u64 {
        match *self {
            Sampler::Greedy => 0,
            Sampler::Sample { seed, .. } => seed,
        }
    }
}

/// The result of a [`Qwen3Backbone::generate_think`] rollout.
pub struct ThinkRollout {
    /// The think-block token ids (everything the model emitted up to and including `</think>`, or
    /// up to EOS). Decode with the tokenizer for the human-readable reasoning text.
    pub think_token_ids: Vec<i32>,
    /// The temporal index after the rollout and the appended `\n\n<img>` — the `text_len` the first
    /// image block is placed after.
    pub t_idx: i32,
}

impl Qwen3Backbone {
    /// One cached single-token forward on the understanding path: embed `token`, run it at temporal
    /// position `pos_t` (`h = w = 0`), persist its K/V, and return the `[vocab]` next-token logits.
    pub fn decode_logits(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<Vec<f32>> {
        self.decode_logits_budgeted(token, pos_t, cache, AttentionPlan::UNBOUNDED)
    }

    pub fn decode_logits_budgeted(
        &self,
        token: i32,
        pos_t: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<Vec<f32>> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached_budgeted(
            &embeds,
            &[pos_t],
            &[0],
            &[0],
            Path::Und,
            cache,
            true,
            attention,
        )?;
        let logits = self.lm_head(&hidden)?; // [1, 1, vocab]
        let vocab = logits.shape()[2];
        // F-144: `as_slice::<f32>()` reinterprets the raw buffer — it is only correct if `logits` is
        // f32. Today that holds by accident (the RoPE path promotes to f32), but make it explicit so a
        // future bf16/f16 lm_head can't silently mis-read the bytes. A no-op when already f32.
        let logits = logits.reshape(&[vocab])?.as_dtype(Dtype::Float32)?;
        Ok(logits.as_slice::<f32>().to_vec())
    }

    /// Like [`decode_logits`](Self::decode_logits) but reduces to the greedy next token **on device**
    /// — only the single argmax index is copied to host, not the whole `[vocab]` f32 row (~600 KB).
    /// MLX `argmax` breaks ties to the lowest index, matching the host `argmax` (`torch.argmax`), so
    /// the greedy stream is bit-identical (F-140).
    pub fn decode_argmax(&self, token: i32, pos_t: i32, cache: &mut KvCache) -> Result<i32> {
        self.decode_argmax_budgeted(token, pos_t, cache, AttentionPlan::UNBOUNDED)
    }

    pub fn decode_argmax_budgeted(
        &self,
        token: i32,
        pos_t: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<i32> {
        let ids = Array::from_slice(&[token], &[1, 1]);
        let embeds = self.embed(&ids)?;
        let hidden = self.forward_cached_budgeted(
            &embeds,
            &[pos_t],
            &[0],
            &[0],
            Path::Und,
            cache,
            true,
            attention,
        )?;
        // The shared on-device argmax flattens the `[1, 1, vocab]` logits internally and breaks ties
        // to the lowest index — the same single-element host transfer + tie rule as the prior local
        // `argmax_device` (F-140).
        let logits = self.lm_head(&hidden)?;
        argmax_device(&logits).map_err(mll)
    }

    /// Splice a run of known tokens into the cache (no sampling), advancing the temporal axis from
    /// `t_idx`. Returns the new `t_idx`. Mirrors the reference `_append_text_tokens_to_cache`: the
    /// tokens take positions `t_idx+1 .. t_idx+len` (`h = w = 0`), so the within-run mask is causal
    /// and they attend to all cached context.
    pub fn append_tokens(&self, ids: &[i32], t_idx: i32, cache: &mut KvCache) -> Result<i32> {
        self.append_tokens_budgeted(ids, t_idx, cache, AttentionPlan::UNBOUNDED)
    }

    pub fn append_tokens_budgeted(
        &self,
        ids: &[i32],
        t_idx: i32,
        cache: &mut KvCache,
        attention: AttentionPlan<'_>,
    ) -> Result<i32> {
        if ids.is_empty() {
            return Ok(t_idx);
        }
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let embeds = self.embed(&ids_arr)?;
        let temporal: Vec<i32> = (t_idx + 1..=t_idx + n).collect();
        let zeros = vec![0i32; ids.len()];
        self.forward_cached_budgeted(
            &embeds,
            &temporal,
            &zeros,
            &zeros,
            Path::Und,
            cache,
            true,
            attention,
        )?;
        Ok(t_idx + n)
    }

    /// Greedy/sampled AR text rollout. `first_logits` are the prefix's last-position logits (the
    /// distribution over the first generated token); `t_idx` is the prefix's max temporal index.
    /// Decoding stops at any id in `eos` (not emitted) or after `max_new_tokens`. Returns the
    /// generated token ids. This is the runtime under `chat`/`answer_question` (sc-3191).
    ///
    /// `cancel` is the cooperative cancellation handle (F-037): checked before each decoded token so a
    /// worker-consumed VQA / Document Studio rollout is cancellable (each token forces a host sync, so
    /// the check observes the trip). Returns [`Error::Canceled`] on trip.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        self.generate_budgeted(
            first_logits,
            cache,
            t_idx,
            eos,
            max_new_tokens,
            sampler,
            cancel,
            AttentionPlan::UNBOUNDED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_budgeted(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<Vec<i32>> {
        // Greedy decodes argmax on device (single-index host transfer per token); sampling needs the
        // full logits row on host — `UndSampler` keeps that split (F-140).
        let rollout = self.rollout_budgeted(
            first_logits,
            cache,
            t_idx,
            eos,
            max_new_tokens,
            sampler,
            cancel,
            attention,
        )?;
        // The pre-engine loop forwarded every emitted token, the last one included, so a budget end
        // leaves the caller's cache holding the whole stream (the engine stops before that forward).
        if rollout.stop.is_none() {
            if let Some(&last) = rollout.tokens.last() {
                let n = rollout.tokens.len() as i32;
                self.append_tokens_budgeted(&[last], t_idx + n - 1, cache, attention)?;
            }
        }
        Ok(rollout.tokens)
    }

    /// The `_generate_think` think/no-think rollout. Greedily decodes a think block from
    /// `first_logits` (the prefix's last-position logits) until `</think>` (`think_end_id`) or any
    /// `eos`, forwarding each emitted token into `cache`; on `</think>` it forwards that token too
    /// (keeping the cache aligned). It then appends `append_ids` (the tokenizer's `\n\n<img>`,
    /// `add_special_tokens=False`) so the cache is primed at the image boundary. Returns the think
    /// token ids and the post-append temporal index. Greedy-only, matching the reference.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_think(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        think_end_id: i32,
        eos: i32,
        append_ids: &[i32],
        max_think_tokens: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<ThinkRollout> {
        self.generate_think_budgeted(
            first_logits,
            cache,
            t_idx,
            think_end_id,
            eos,
            append_ids,
            max_think_tokens,
            cancel,
            AttentionPlan::UNBOUNDED,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_think_budgeted(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        think_end_id: i32,
        eos: i32,
        append_ids: &[i32],
        max_think_tokens: usize,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<ThinkRollout> {
        let mut t = t_idx;
        let mut next = argmax(first_logits);
        let mut think_token_ids = Vec::new();
        let mut closed = false;
        if max_think_tokens > 0 {
            let rollout = self.rollout_budgeted(
                first_logits,
                cache,
                t_idx,
                &[eos, think_end_id],
                max_think_tokens,
                Sampler::Greedy,
                cancel,
                attention,
            )?;
            think_token_ids = rollout.tokens;
            let n = think_token_ids.len() as i32;
            t = t_idx + n;
            match rollout.stop {
                // `eos` is checked first, as the pre-engine loop did.
                Some(stop) if stop == eos => next = eos,
                Some(_) => {
                    // Forward `</think>` so the cache includes it, then stop. No logits needed
                    // here, so splice it in without an lm_head projection (F-140).
                    t = self.append_tokens_budgeted(&[think_end_id], t, cache, attention)?;
                    think_token_ids.push(think_end_id);
                    next = think_end_id;
                    closed = true;
                }
                None => {
                    // The budget ran out: the pre-engine loop had forwarded the last token too, and
                    // the argmax it drew decides the close below.
                    let last = *think_token_ids
                        .last()
                        .expect("a budget end emitted a token");
                    next = self.decode_argmax_budgeted(last, t, cache, attention)?;
                }
            }
        }
        // Budget exhausted before `</think>` (and the model didn't emit `eos`): synthesize the close
        // so the cache is not primed on an unclosed `<think>` token sequence the model was never
        // trained on, which would degrade the subsequent image generation (F-013).
        if !closed && next != eos {
            t = self.append_tokens_budgeted(&[think_end_id], t, cache, attention)?;
            think_token_ids.push(think_end_id);
        }
        t = self.append_tokens_budgeted(append_ids, t, cache, attention)?;
        Ok(ThinkRollout {
            think_token_ids,
            t_idx: t,
        })
    }
}

/// A text rollout on the shared engine: the tokens emitted and, on a stop-token end, the stop drawn
/// (never emitted, never fed). Every emitted token but the last was fed through the cache; on a
/// stop-token end the last was too.
pub(crate) struct UndRollout {
    pub(crate) tokens: Vec<i32>,
    pub(crate) stop: Option<i32>,
}

impl Qwen3Backbone {
    /// Roll out up to `max_new_tokens` understanding-path tokens on the shared engine from
    /// `first_logits` (the prefix's last-position row), the first fed at temporal `t_idx + 1`, until
    /// any of `stops` is drawn. `cancel` is bridged onto the engine's flag — set before the run, the
    /// engine refuses it before any draw; set mid-run, it is observed after the token being emitted
    /// and the engine finishes `Cancelled` — and either way returns [`Error::Canceled`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollout_budgeted(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        stops: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
    ) -> Result<UndRollout> {
        self.rollout_observed(
            first_logits,
            cache,
            t_idx,
            stops,
            max_new_tokens,
            sampler,
            cancel,
            attention,
            &mut |_| {},
        )
    }

    /// [`rollout_budgeted`](Self::rollout_budgeted), handing every engine event to `on_event`
    /// before the cancel bridge reads the pipeline's flag.
    #[allow(clippy::too_many_arguments)]
    fn rollout_observed(
        &self,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        stops: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
        attention: AttentionPlan<'_>,
        on_event: &mut dyn FnMut(&StreamEvent),
    ) -> Result<UndRollout> {
        let target = UndTarget {
            backbone: self,
            attention,
            failure: RefCell::new(None),
        };
        let mut draw = UndSampler::new(sampler);
        let generation = GenerationConfig {
            max_new_tokens,
            sampling: draw.params,
            seed: Some(sampler.seed()),
            stop_tokens: stops.to_vec(),
        };
        let start = cache.offset();
        // The engine's history is the penalty window and the stochastic draw reads none; the prefix
        // (spliced image rows included) is its length in placeholder ids.
        let history = vec![0; start.max(1) as usize];
        let logits = Array::from_slice(first_logits, &[1, first_logits.len() as i32]);
        // The pipeline's cancel, bridged onto the engine's flag: before a run that would draw (the
        // engine's typed pre-inference refusal; a zero budget draws nothing and ends `Ok`, as the
        // pre-engine loops did) and after every emitted token (its `Cancelled` finish).
        let engine_cancel = mlx_llm::CancelFlag::new();
        if max_new_tokens > 0 && cancel.is_some_and(CancelFlag::is_cancelled) {
            engine_cancel.cancel();
        }
        let bridged = engine_cancel.clone();
        let mut bridge = |event: StreamEvent| {
            on_event(&event);
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                bridged.cancel();
            }
        };
        let run = generate_speculative(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Prefilled {
                cache,
                logits,
                hidden: None,
                history: &history,
                // The first decode step runs at temporal `t_idx + 1`, wherever the cache ends.
                position_delta: t_idx + 1 - start,
            },
            &generation,
            0,
            &engine_cancel,
            &mut bridge,
            EngineOptions {
                sampler: Some(&mut draw),
                // Every draw is read back before the next step; nothing pipelines.
                pipelining: Pipelining::Off,
                ..EngineOptions::default()
            },
        );
        // A model or draw failure keeps its own typed error.
        if let Some(error) = target.failure.take().or_else(|| draw.failure.take()) {
            return Err(error);
        }
        let run = run.map_err(|e| match e {
            mlx_llm::Error::Canceled => Error::Canceled,
            e => Error::Msg(format!("sensenova text decode: {e}")),
        })?;
        if run.output.finish_reason == FinishReason::Cancelled {
            return Err(Error::Canceled);
        }
        let stop = (run.output.finish_reason == FinishReason::StopToken)
            .then_some(draw.last)
            .flatten();
        Ok(UndRollout {
            tokens: run.output.tokens,
            stop,
        })
    }
}

/// The understanding path as a target of the shared MLX engine: one cached single-token forward at
/// the temporal position the engine hands it (`h = w = 0`), projected to the `[1, 1, vocab]`
/// logits. Token-at-a-time only — a verify forward over drafts is refused.
struct UndTarget<'a> {
    backbone: &'a Qwen3Backbone,
    attention: AttentionPlan<'a>,
    /// The backbone's own error from a failed forward, returned as-is by the rollout.
    failure: RefCell<Option<Error>>,
}

impl UndTarget<'_> {
    fn step(&self, cache: &mut KvCache, ids: &Array, pos_t: i32) -> Result<Array> {
        let embeds = self.backbone.embed(ids)?;
        let hidden = self.backbone.forward_cached_budgeted(
            &embeds,
            &[pos_t],
            &[0],
            &[0],
            Path::Und,
            cache,
            true,
            self.attention,
        )?;
        self.backbone.lm_head(&hidden) // [1, 1, vocab]
    }
}

impl SpeculativeTarget for UndTarget<'_> {
    type Cache = KvCache;
    type Rollback = NoDraftRollback;

    fn new_cache(&self) -> KvCache {
        self.backbone.new_cache()
    }

    fn cache_len(&self, cache: &KvCache) -> i32 {
        cache.offset()
    }

    fn rollback(&self, _: usize) -> NoDraftRollback {
        NoDraftRollback
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        ids: &Array,
        rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> mlx_llm::Result<TargetOutput> {
        if want_hidden || scope == LogitsScope::All || ids.shape() != [1, 1] {
            return Err(mlx_llm::Error::Unsupported(
                "the SenseNova understanding target steps one token, last-position logits only"
                    .into(),
            ));
        }
        match self.step(cache, ids, rope_offset) {
            Ok(logits) => Ok(TargetOutput {
                logits,
                hidden: None,
            }),
            Err(error) => {
                let message = error.to_string();
                *self.failure.borrow_mut() = Some(error);
                Err(mlx_llm::Error::Msg(message))
            }
        }
    }

    fn attention_label(&self) -> &'static str {
        // K/V heads repeated to the query heads before the (budgeted) SDPA (`qwen3.rs`).
        "expanded"
    }
}

/// A rollout's draw on the engine's sampler seam — exactly the pre-engine loops' draws. The first
/// draw reads the prefix row it was handed on the host ([`Sampler::pick`]); a greedy decode step
/// reduces to the argmax on device (one index read back, F-140); a stochastic decode step pulls the
/// f32 row to the host and draws [`Sampler::pick`] from the rollout's seeded stream.
struct UndSampler {
    sampler: Sampler,
    rng: SplitMix64,
    /// The same knobs in the engine's vocabulary (reported, never drawn from).
    params: SamplingParams,
    device_draws: u64,
    host_draws: u64,
    /// The last token drawn — on a stop-token end, the stop the engine does not emit.
    last: Option<i32>,
    /// A draw's own error, returned as-is by the rollout.
    failure: Option<Error>,
}

impl UndSampler {
    fn new(sampler: Sampler) -> Self {
        let params = match sampler {
            Sampler::Greedy => SamplingParams::default(),
            Sampler::Sample {
                temperature,
                top_p,
                top_k,
                ..
            } => SamplingParams {
                temperature,
                top_p,
                top_k,
                repetition_penalty: 1.0,
                repetition_context: 0,
                presence_penalty: 0.0,
            },
        };
        Self {
            sampler,
            rng: SplitMix64::new(sampler.seed()),
            params,
            device_draws: 0,
            host_draws: 0,
            last: None,
            failure: None,
        }
    }

    fn draw(&mut self, logits: &Array) -> Result<i32> {
        let first = self.host_draws + self.device_draws == 0;
        let token = if matches!(self.sampler, Sampler::Greedy) && !first {
            self.device_draws += 1;
            argmax_device(logits).map_err(mll)?
        } else {
            self.host_draws += 1;
            let vocab = *logits.shape().last().expect("logits rank >= 1");
            let row = logits.reshape(&[vocab])?.as_dtype(Dtype::Float32)?;
            self.sampler.pick(row.as_slice::<f32>(), &mut self.rng)?
        };
        self.last = Some(token);
        Ok(token)
    }
}

impl TokenSampler for UndSampler {
    fn params(&self) -> &SamplingParams {
        &self.params
    }

    fn sample(
        &mut self,
        logits: &Array,
        _: &[i32],
        allowed: Option<&[bool]>,
    ) -> mlx_llm::Result<SampledToken> {
        if allowed.is_some() {
            return Err(mlx_llm::Error::Unsupported(
                "the SenseNova rollout draw takes no constraint mask".into(),
            ));
        }
        match self.draw(logits) {
            Ok(token) => Ok(SampledToken::Host(token)),
            Err(error) => {
                let message = error.to_string();
                self.failure = Some(error);
                Err(mlx_llm::Error::Msg(message))
            }
        }
    }

    fn argmax_rows(&mut self, _: &Array) -> mlx_llm::Result<Vec<i32>> {
        Err(mlx_llm::Error::Unsupported(
            "the SenseNova rollouts decode token-at-a-time; they never verify drafts".into(),
        ))
    }

    fn distribution(
        &mut self,
        _: &Array,
        _: &[i32],
        _: Option<&[bool]>,
    ) -> mlx_llm::Result<Vec<(i32, f32)>> {
        Err(mlx_llm::Error::Unsupported(
            "the SenseNova rollouts decode token-at-a-time; they never verify drafts".into(),
        ))
    }

    fn uniform(&mut self) -> f32 {
        use mlx_llm::primitives::TokenRng as _;
        self.rng.next_f32()
    }

    fn path(&self) -> Option<SamplerPath> {
        match (self.host_draws, self.device_draws) {
            (0, 0) => None,
            (0, _) => Some(SamplerPath::Device),
            _ => Some(SamplerPath::Host(HostSampleReason::Reference)),
        }
    }

    fn reads_history(&self) -> bool {
        false
    }
}

/// Index of the maximum logit (ties → lowest index, matching `torch.argmax`). Delegates to the
/// shared host argmax (sc-7159); kept as a crate-internal name for the greedy draws over `[vocab]`
/// rows already on the host (a rollout's first draw, a zero-budget think rollout's close decision).
pub(crate) fn argmax(logits: &[f32]) -> i32 {
    argmax_host(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_breaks_ties_to_lowest_index() {
        assert_eq!(argmax(&[0.1, 0.5, 0.5, 0.2]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn top_k_one_is_argmax() {
        // top_k = 1 collapses the shaped distribution to the single max → deterministic argmax,
        // whatever the seed. Exercises the shared sampler through `Sampler::pick`.
        let logits = [0.1, 2.0, 0.5, 1.0];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 1,
            seed: 123,
        };
        let mut rng = SplitMix64::new(s.seed());
        for _ in 0..16 {
            assert_eq!(s.pick(&logits, &mut rng).unwrap(), 1);
        }
    }

    #[test]
    fn sampling_is_seed_deterministic() {
        let logits = [0.2, 1.5, 0.3, 0.9, 0.1];
        let s = Sampler::Sample {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            seed: 42,
        };
        let run = || {
            let mut rng = SplitMix64::new(s.seed());
            (0..8)
                .map(|_| s.pick(&logits, &mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run(), "same seed → identical token sequence");
    }

    // ---- E8 (sc-24446): the rollouts on the shared engine are the pre-engine loops ----

    use crate::config::NeoChatConfig;
    use mlx_gen::weights::Weights;

    /// The committed synthetic runtime fixture's backbone, its prefix (ids + tri-axis indexes).
    fn fixture() -> (Qwen3Backbone, Array, [Vec<i32>; 3]) {
        let w = Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/runtime_golden.safetensors"
        ))
        .unwrap();
        let m = |k: &str| w.metadata(k).unwrap().to_string();
        let llm = serde_json::json!({
            "model_type": "qwen3",
            "hidden_size": m("hidden_size").parse::<u64>().unwrap(),
            "intermediate_size": m("intermediate_size").parse::<u64>().unwrap(),
            "num_hidden_layers": m("num_hidden_layers").parse::<u64>().unwrap(),
            "num_attention_heads": m("num_attention_heads").parse::<u64>().unwrap(),
            "num_key_value_heads": m("num_key_value_heads").parse::<u64>().unwrap(),
            "head_dim": m("head_dim").parse::<u64>().unwrap(),
            "rms_norm_eps": m("rms_norm_eps").parse::<f64>().unwrap(),
            "rope_theta": m("rope_theta").parse::<f64>().unwrap(),
            "rope_theta_hw": m("rope_theta_hw").parse::<f64>().unwrap(),
            "vocab_size": m("vocab_size").parse::<u64>().unwrap(),
            "attention_bias": false,
        });
        let v = serde_json::json!({ "model_type": "neo_chat", "tie_word_embeddings": false, "llm_config": llm, "vision_config": {} });
        let cfg = NeoChatConfig::from_config_json(&v).unwrap();
        let model = Qwen3Backbone::from_weights(&w, &cfg, "language_model").unwrap();
        let ids = w.require("prefix.input_ids").unwrap().clone();
        let idx = w.require("prefix.indexes").unwrap();
        let s = idx.shape()[1] as usize;
        let flat = idx.as_slice::<i32>();
        let row = |r: usize| flat[r * s..(r + 1) * s].to_vec();
        (model, ids, [row(0), row(1), row(2)])
    }

    /// Prefill the prefix: the cache, the last-position host row and the prefix's max temporal index.
    fn prefill(
        model: &Qwen3Backbone,
        ids: &Array,
        idx: &[Vec<i32>; 3],
    ) -> (KvCache, Vec<f32>, i32) {
        let embeds = model.embed(ids).unwrap();
        let mut cache = model.new_cache();
        let hidden = model
            .forward_cached(
                &embeds,
                &idx[0],
                &idx[1],
                &idx[2],
                Path::Und,
                &mut cache,
                true,
            )
            .unwrap();
        let logits = model.lm_head(&hidden).unwrap();
        let (n, vocab) = (logits.shape()[1], logits.shape()[2]);
        let last = logits
            .take_axis(Array::from_slice(&[n - 1], &[1]), 1)
            .unwrap()
            .reshape(&[vocab])
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let t_idx = *idx[0].iter().max().unwrap();
        (cache, last.as_slice::<f32>().to_vec(), t_idx)
    }

    /// The next decode row after a rollout — what the cache it left holds, read through one probe.
    fn probe(model: &Qwen3Backbone, cache: &mut KvCache, t: i32) -> Vec<f32> {
        model.decode_logits(7, t + 1, cache).unwrap()
    }

    /// The pre-engine `generate_budgeted` loops, verbatim.
    #[allow(clippy::too_many_arguments)]
    fn reference_generate(
        model: &Qwen3Backbone,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        eos: &[i32],
        max_new_tokens: usize,
        sampler: Sampler,
        cancel: Option<&CancelFlag>,
    ) -> Result<Vec<i32>> {
        let attention = AttentionPlan::UNBOUNDED;
        if let Sampler::Greedy = sampler {
            let mut next = argmax(first_logits);
            let mut out = Vec::new();
            let mut t = t_idx;
            for _ in 0..max_new_tokens {
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    return Err(Error::Canceled);
                }
                if eos.contains(&next) {
                    break;
                }
                out.push(next);
                t += 1;
                next = model.decode_argmax_budgeted(next, t, cache, attention)?;
            }
            return Ok(out);
        }

        let mut rng = SplitMix64::new(sampler.seed());
        let mut logits = first_logits.to_vec();
        let mut out = Vec::new();
        let mut t = t_idx;
        for _ in 0..max_new_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let next = sampler.pick(&logits, &mut rng)?;
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            t += 1;
            logits = model.decode_logits_budgeted(next, t, cache, attention)?;
        }
        Ok(out)
    }

    /// The pre-engine `generate_think_budgeted` loop, verbatim.
    #[allow(clippy::too_many_arguments)]
    fn reference_think(
        model: &Qwen3Backbone,
        first_logits: &[f32],
        cache: &mut KvCache,
        t_idx: i32,
        think_end_id: i32,
        eos: i32,
        append_ids: &[i32],
        max_think_tokens: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<ThinkRollout> {
        let attention = AttentionPlan::UNBOUNDED;
        let mut t = t_idx;
        let mut next = argmax(first_logits);
        let mut think_token_ids = Vec::new();
        let mut closed = false;
        for _ in 0..max_think_tokens {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            if next == eos {
                break;
            }
            if next == think_end_id {
                t = model.append_tokens_budgeted(&[next], t, cache, attention)?;
                think_token_ids.push(next);
                closed = true;
                break;
            }
            think_token_ids.push(next);
            next = model.decode_argmax_budgeted(next, t + 1, cache, attention)?;
            t += 1;
        }
        if !closed && next != eos {
            t = model.append_tokens_budgeted(&[think_end_id], t, cache, attention)?;
            think_token_ids.push(think_end_id);
        }
        t = model.append_tokens_budgeted(append_ids, t, cache, attention)?;
        Ok(ThinkRollout {
            think_token_ids,
            t_idx: t,
        })
    }

    /// The pre-engine interleave text segment (`t2i.rs`), verbatim: `(tokens, next, hit_max)`.
    fn reference_segment(
        model: &Qwen3Backbone,
        mut next: i32,
        cache: &mut KvCache,
        t_cond: &mut usize,
        stops: [i32; 2],
        total_tokens: &mut usize,
        max_new_tokens: usize,
    ) -> (Vec<i32>, i32, bool) {
        let mut gen_tokens = Vec::new();
        let mut hit_max = false;
        loop {
            if next == stops[0] || next == stops[1] {
                break;
            }
            gen_tokens.push(next);
            *total_tokens += 1;
            next = model
                .decode_argmax_budgeted(next, (*t_cond + 1) as i32, cache, AttentionPlan::UNBOUNDED)
                .unwrap();
            *t_cond += 1;
            if *total_tokens >= max_new_tokens {
                hit_max = true;
                break;
            }
        }
        (gen_tokens, next, hit_max)
    }

    fn samplers() -> Vec<Sampler> {
        let mut all = vec![Sampler::Greedy];
        for (temperature, top_p, top_k) in [(0.8, 1.0, 0), (1.0, 0.9, 0), (1.3, 0.8, 5)] {
            for seed in [0, 7, 42] {
                all.push(Sampler::Sample {
                    temperature,
                    top_p,
                    top_k,
                    seed,
                });
            }
        }
        all
    }

    /// `generate` on the engine emits the pre-engine stream — greedy and seeded stochastic, across
    /// eos ends (an eos the run itself emits), budget ends and the zero budget — and leaves the
    /// caller's cache exactly as the pre-engine loop did (same length, same next-step logits).
    #[test]
    fn generate_on_the_engine_is_the_pre_engine_loop() {
        let (model, ids, idx) = fixture();
        let (mut eos_ends, mut budget_ends) = (0, 0);
        for sampler in samplers() {
            let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
            let free =
                reference_generate(&model, &first, &mut c, t_idx, &[], 12, sampler, None).unwrap();
            for (eos, max) in [
                (vec![], 12),
                (vec![free[3]], 12),
                (vec![free[0]], 12),
                (vec![], 0),
            ] {
                let (mut want_cache, _, _) = prefill(&model, &ids, &idx);
                let want = reference_generate(
                    &model,
                    &first,
                    &mut want_cache,
                    t_idx,
                    &eos,
                    max,
                    sampler,
                    None,
                )
                .unwrap();
                let (mut got_cache, _, _) = prefill(&model, &ids, &idx);
                let got = model
                    .generate(&first, &mut got_cache, t_idx, &eos, max, sampler, None)
                    .unwrap();
                assert_eq!(got, want, "{sampler:?} eos {eos:?} max {max}");
                assert_eq!(got_cache.offset(), want_cache.offset());
                let t = t_idx + want.len() as i32;
                assert_eq!(
                    probe(&model, &mut got_cache, t),
                    probe(&model, &mut want_cache, t)
                );
                if want.len() < max {
                    eos_ends += 1;
                } else if max > 0 {
                    budget_ends += 1;
                }
            }
        }
        assert!(
            eos_ends > 0 && budget_ends > 0,
            "eos {eos_ends}, budget {budget_ends}"
        );
        // A cancel stays typed and is checked before every token, the first included.
        let cancel = CancelFlag::new();
        cancel.cancel();
        for sampler in [Sampler::Greedy, samplers()[1]] {
            let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
            assert!(matches!(
                reference_generate(
                    &model,
                    &first,
                    &mut c,
                    t_idx,
                    &[],
                    4,
                    sampler,
                    Some(&cancel)
                ),
                Err(Error::Canceled)
            ));
            let (mut c, _, _) = prefill(&model, &ids, &idx);
            assert!(matches!(
                model.generate(&first, &mut c, t_idx, &[], 4, sampler, Some(&cancel)),
                Err(Error::Canceled)
            ));
            let (mut c, _, _) = prefill(&model, &ids, &idx);
            assert_eq!(
                model
                    .generate(&first, &mut c, t_idx, &[], 0, sampler, Some(&cancel))
                    .unwrap(),
                Vec::<i32>::new()
            );
        }
    }

    /// The think rollout on the engine is the pre-engine loop: a `</think>` end, an eos end, a
    /// budget end (closed synthetically, or not when its last forward drew eos) and the zero budget,
    /// with the same think ids, temporal index and primed cache.
    #[test]
    fn think_on_the_engine_is_the_pre_engine_loop() {
        let (model, ids, idx) = fixture();
        let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
        let free = reference_generate(
            &model,
            &first,
            &mut c,
            t_idx,
            &[],
            12,
            Sampler::Greedy,
            None,
        )
        .unwrap();
        let append = [11, 13];
        let cases = [
            (free[4], -1, 12),     // `</think>` mid-stream
            (-1, free[3], 12),     // eos mid-stream
            (-1, free[5], 5),      // the budget's last forward draws eos: no synthetic close
            (-1, -1, 6),           // budget end, closed synthetically
            (free[2], free[2], 9), // eos and `</think>` alike: eos wins
            (free[0], -1, 0),      // zero budget
            (-1, free[0], 0),
        ];
        for (think_end, eos, max) in cases {
            let (mut want_cache, _, _) = prefill(&model, &ids, &idx);
            let want = reference_think(
                &model,
                &first,
                &mut want_cache,
                t_idx,
                think_end,
                eos,
                &append,
                max,
                None,
            )
            .unwrap();
            let (mut got_cache, _, _) = prefill(&model, &ids, &idx);
            let got = model
                .generate_think(
                    &first,
                    &mut got_cache,
                    t_idx,
                    think_end,
                    eos,
                    &append,
                    max,
                    None,
                )
                .unwrap();
            let what = format!("think_end {think_end} eos {eos} max {max}");
            assert_eq!(got.think_token_ids, want.think_token_ids, "{what}");
            assert_eq!(got.t_idx, want.t_idx, "{what}");
            assert_eq!(got_cache.offset(), want_cache.offset(), "{what}");
            assert_eq!(
                probe(&model, &mut got_cache, got.t_idx),
                probe(&model, &mut want_cache, want.t_idx),
                "{what}"
            );
        }
        let cancel = CancelFlag::new();
        cancel.cancel();
        let (mut c, _, _) = prefill(&model, &ids, &idx);
        assert!(matches!(
            model.generate_think(&first, &mut c, t_idx, -1, -1, &append, 3, Some(&cancel)),
            Err(Error::Canceled)
        ));
    }

    /// An interleave text segment on the engine (`rollout_budgeted` with the segment's stops and
    /// remaining budget, as `t2i.rs` calls it) emits the pre-engine segment's tokens and ends the
    /// same way: on the same stop, or at the budget (`hit_max`).
    #[test]
    fn interleave_segments_on_the_engine_are_the_pre_engine_loop() {
        let (model, ids, idx) = fixture();
        let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
        let free = reference_generate(
            &model,
            &first,
            &mut c,
            t_idx,
            &[],
            12,
            Sampler::Greedy,
            None,
        )
        .unwrap();
        let (mut stop_ends, mut budget_ends) = (0, 0);
        for (stops, max_new_tokens, total) in [
            ([free[3], -1], 12, 0),
            ([-1, free[2]], 12, 0),
            ([-1, -1], 6, 0),
            ([-1, -1], 9, 4),
            ([free[0], -1], 12, 0),
            ([-1, -1], 0, 0),
        ] {
            let (mut want_cache, _, _) = prefill(&model, &ids, &idx);
            let (mut t_cond, mut want_total) = (t_idx as usize, total);
            let (want, next, hit_max) = reference_segment(
                &model,
                argmax(&first),
                &mut want_cache,
                &mut t_cond,
                stops,
                &mut want_total,
                max_new_tokens,
            );
            let (mut got_cache, _, _) = prefill(&model, &ids, &idx);
            let got = model
                .rollout_budgeted(
                    &first,
                    &mut got_cache,
                    t_idx,
                    &stops,
                    max_new_tokens.saturating_sub(total).max(1),
                    Sampler::Greedy,
                    None,
                    AttentionPlan::UNBOUNDED,
                )
                .unwrap();
            let what = format!("stops {stops:?} max {max_new_tokens} total {total}");
            assert_eq!(got.tokens, want, "{what}");
            assert_eq!(got.stop.is_none(), hit_max, "{what}");
            if !hit_max {
                assert_eq!(got.stop, Some(next), "{what}");
                assert_eq!(got_cache.offset(), want_cache.offset(), "{what}");
                stop_ends += 1;
            } else {
                budget_ends += 1;
            }
        }
        assert!(stop_ends > 0 && budget_ends > 0);
    }

    /// The pipeline's cancel, set mid-rollout, is bridged onto the engine's flag: the engine
    /// finishes `Cancelled` right after the token it was emitting and the rollout returns the typed
    /// [`Error::Canceled`] (greedy and stochastic).
    #[test]
    fn a_mid_run_cancel_ends_the_rollout_typed() {
        let (model, ids, idx) = fixture();
        for sampler in [Sampler::Greedy, samplers()[1]] {
            let (mut c, first, t_idx) = prefill(&model, &ids, &idx);
            let cancel = CancelFlag::new();
            let mut emitted = 0;
            let run = model.rollout_observed(
                &first,
                &mut c,
                t_idx,
                &[],
                12,
                sampler,
                Some(&cancel),
                AttentionPlan::UNBOUNDED,
                &mut |event| {
                    if let StreamEvent::Token { step, .. } = event {
                        emitted = step + 1;
                        if *step == 2 {
                            cancel.cancel();
                        }
                    }
                },
            );
            assert!(matches!(run, Err(Error::Canceled)), "{sampler:?}");
            assert_eq!(
                emitted, 3,
                "{sampler:?}: cancelled right after the third token"
            );
        }
    }
}

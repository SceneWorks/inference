//! The step-model seam (epic sc-24128, story sc-24129).
//!
//! [`StepModel`] is the one thing the fast-decode machinery asks of a model: *run one decode step
//! over N input tokens against a [`DecodeCache`], and give me logits for the last (or every)
//! position*. Everything the epic layers on top — the unified speculative engine (drafts verified in
//! one N-token step, rejected suffixes dropped with [`DecodeCache::rollback_to`]), static KV, the
//! CUDA-graph runner — is written once against this trait; model files only implement it.
//!
//! [`generate_step`] is the walking-skeleton driver: the token-at-a-time loop over the seam. It is
//! deliberately the same algorithm as the reference loop in [`stream`](super::stream) (same prefill,
//! same sampler, same stop / cancel / constraint order) so its output is token-identical to the
//! reference path for the same prompt and config — the parity gate the tiny-config and real-weight
//! tests hold. Unlike the reference loop it returns a [`DecodeRecord`] with measured counters.

use candle_core::{Device, Tensor};

use crate::decode::cancel::CancelFlag;
use crate::decode::record::{DecodePath, DecodeRecord, RequestSpan};
use crate::decode::stream::{
    default_seed, ConstraintMask, FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::{Error, Result};
use crate::primitives::decode_cache::{CacheMemory, DecodeCache};
use crate::primitives::sampler::{sample, SplitMix64};

/// Which positions' logits a step returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitsScope {
    /// Only the last input position: logits `[batch, vocab]`.
    Last,
    /// Every input position: logits `[batch, n, vocab]` (the verify forward of a speculative step).
    All,
}

/// One decode step's inputs.
#[derive(Clone, Copy, Debug)]
pub struct StepRequest<'a> {
    /// The tokens to feed, in order; they occupy positions `cache.len() .. cache.len() + n`.
    pub tokens: &'a [i32],
    /// Which logits to return.
    pub scope: LogitsScope,
    /// Also return the final-normalized hidden states `[batch, n, hidden]` (what a native MTP head
    /// pairs with the next token). `false` skips the extra tensor.
    pub want_hidden: bool,
}

impl<'a> StepRequest<'a> {
    /// Last-position logits over `tokens`, no hidden states — the plain decode step.
    pub fn last(tokens: &'a [i32]) -> Self {
        Self {
            tokens,
            scope: LogitsScope::Last,
            want_hidden: false,
        }
    }

    /// All-position logits over `tokens`, no hidden states — the verify step.
    pub fn all(tokens: &'a [i32]) -> Self {
        Self {
            tokens,
            scope: LogitsScope::All,
            want_hidden: false,
        }
    }
}

/// One decode step's outputs.
#[derive(Clone, Debug)]
pub struct StepOutput {
    /// `[batch, vocab]` for [`LogitsScope::Last`], `[batch, n, vocab]` for [`LogitsScope::All`].
    pub logits: Tensor,
    /// Final-normalized hidden states `[batch, n, hidden]` when requested.
    pub hidden: Option<Tensor>,
}

/// A model the fast-decode machinery can drive.
pub trait StepModel {
    /// The model's per-request state.
    type Cache: DecodeCache;

    /// A fresh, empty cache.
    fn new_cache(&self) -> Self::Cache;

    /// Where input-id tensors must live.
    fn device(&self) -> &Device;

    /// Logit width.
    fn vocab_size(&self) -> usize;

    /// Run one step: feed `request.tokens` at positions `cache.len()..`, advance the cache by
    /// `tokens.len()`, and return the requested logits. An empty token slice is an error.
    ///
    /// Named `forward_step` rather than `step` so a model that also implements the reference
    /// [`Decode`](super::Decode) trait (whose method is `step`) has no ambiguous call site when both
    /// traits are in scope.
    fn forward_step(&self, cache: &mut Self::Cache, request: StepRequest<'_>)
        -> Result<StepOutput>;
}

/// Generate from `prompt_ids` through the [`StepModel`] seam, returning the output and the
/// measured [`DecodeRecord`] (`path == StepModel`). Token-identical to
/// [`generate_with`](super::generate_with) for the same inputs.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference.
pub fn generate_step<M: StepModel>(
    model: &M,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
) -> Result<(GenerationOutput, DecodeRecord)> {
    let (output, record, _) = generate_step_timed(
        model, prompt_ids, config, cancel, on_event, constraint, None,
    )?;
    Ok((output, record))
}

/// [`generate_step`] with an optional prefill boundary callback, invoked once the prompt is in the
/// cache and before the first token is sampled (the timed-provider / bench seam; the callback may
/// synchronize the device). Also returns the **final** cache's [`DecodeCache::memory`] — the state
/// the request actually held at its last step, rollback checkpoints included — so a bench reports
/// what a full-length request costs rather than a fresh cache's.
pub fn generate_step_timed<M: StepModel>(
    model: &M,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    mut constraint: Option<&mut dyn ConstraintMask>,
    mut on_prefill_complete: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<(GenerationOutput, DecodeRecord, CacheMemory)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_step: empty prompt".into()));
    }

    let span = RequestSpan::begin();
    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let mut cache = model.new_cache();
    let mut forwards = 0u64;

    // Prefill the whole prompt at position 0; logits are for the last prompt position.
    let mut logits = model
        .forward_step(&mut cache, StepRequest::last(prompt_ids))?
        .logits;
    forwards += 1;
    if let Some(boundary) = on_prefill_complete.as_mut() {
        boundary()?;
    }

    let mut history: Vec<i32> = prompt_ids.to_vec();
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;

    for step in 0..config.max_new_tokens {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        let next = {
            let mask = constraint.as_mut().map(|c| c.allowed());
            sample(&logits, &history, &config.sampling, &mut rng, mask)?
        };

        if config.stop_tokens.contains(&next) {
            finish = FinishReason::StopToken;
            break;
        }

        if let Some(c) = &mut constraint {
            c.accept(next);
        }

        on_event(StreamEvent::Token { id: next, step });
        generated.push(next);
        history.push(next);

        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        if step + 1 == config.max_new_tokens {
            break; // budget reached; finish stays MaxTokens
        }

        // Feed the new token back at the cache's current length.
        logits = model
            .forward_step(&mut cache, StepRequest::last(&[next]))?
            .logits;
        forwards += 1;
    }

    on_event(StreamEvent::Done {
        reason: finish,
        generated: generated.len(),
    });
    let record = DecodeRecord::plain(
        DecodePath::StepModel,
        forwards,
        generated.len(),
        span.counters(),
    );
    Ok((
        GenerationOutput {
            tokens: generated,
            finish_reason: finish,
        },
        record,
        cache.memory(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// A fixed-logits model whose cache is a bare counter, to exercise the driver's control flow.
    struct Counter {
        device: Device,
        steps: Cell<u64>,
        vocab: usize,
    }

    struct CounterCache(i32);

    impl DecodeCache for CounterCache {
        fn len(&self) -> i32 {
            self.0
        }
        fn rollback_to(&mut self, n: i32) -> Result<()> {
            if n > self.0 || n < 0 {
                return Err(Error::Msg("rollback past end".into()));
            }
            self.0 = n;
            Ok(())
        }
        fn reset(&mut self) {
            self.0 = 0;
        }
        fn memory(&self) -> CacheMemory {
            // Grows with the cache, so a test can tell the final cache from a fresh one.
            CacheMemory {
                live_bytes: self.0 as usize * 4,
                checkpoint_bytes: 1,
            }
        }
    }

    impl StepModel for Counter {
        type Cache = CounterCache;
        fn new_cache(&self) -> CounterCache {
            CounterCache(0)
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn forward_step(
            &self,
            cache: &mut CounterCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            assert!(!request.tokens.is_empty());
            self.steps.set(self.steps.get() + 1);
            cache.0 += request.tokens.len() as i32;
            // Token `(len) % vocab` is the argmax: the sequence is a predictable ramp.
            let mut row = vec![0f32; self.vocab];
            row[(cache.0 as usize) % self.vocab] = 10.0;
            let logits = Tensor::from_vec(row, (1, self.vocab), &self.device)?;
            Ok(StepOutput {
                logits: match request.scope {
                    LogitsScope::Last => logits,
                    LogitsScope::All => logits.unsqueeze(1)?,
                },
                hidden: None,
            })
        }
    }

    fn counter() -> Counter {
        Counter {
            device: Device::Cpu,
            steps: Cell::new(0),
            vocab: 4,
        }
    }

    #[test]
    fn driver_counts_forwards_and_host_syncs_per_token() {
        let model = counter();
        let cfg = GenerationConfig {
            max_new_tokens: 5,
            seed: Some(1),
            ..Default::default()
        };
        let mut events = Vec::new();
        let (out, record) = generate_step(
            &model,
            &[1, 2],
            &cfg,
            &CancelFlag::new(),
            &mut |e| events.push(e),
            None,
        )
        .unwrap();
        assert_eq!(out.finish_reason, FinishReason::MaxTokens);
        // Prompt of 2 → cache 2 → argmax 2; then 3, 0, 1, 2.
        assert_eq!(out.tokens, vec![2, 3, 0, 1, 2]);
        assert_eq!(record.path, DecodePath::StepModel);
        assert_eq!(record.target_forwards, 5, "prefill + 4 single-token steps");
        assert_eq!(model.steps.get(), 5);
        assert_eq!(record.generated_tokens, 5);
        assert_eq!(
            record.host_syncs, 5,
            "one device argmax transfer per sampled token"
        );
        assert_eq!(record.host_syncs_per_token(), Some(1.0));
        assert_eq!(record.acceptance_rate(), None);
        assert!(matches!(
            events.last(),
            Some(StreamEvent::Done {
                reason: FinishReason::MaxTokens,
                generated: 5
            })
        ));
    }

    #[test]
    fn stop_token_and_cancel_follow_the_reference_contract() {
        let model = counter();
        let cfg = GenerationConfig {
            max_new_tokens: 8,
            stop_tokens: vec![0],
            seed: Some(1),
            ..Default::default()
        };
        let (out, record) =
            generate_step(&model, &[1, 2], &cfg, &CancelFlag::new(), &mut |_| {}, None).unwrap();
        assert_eq!(out.tokens, vec![2, 3]);
        assert_eq!(out.finish_reason, FinishReason::StopToken);
        assert_eq!(record.target_forwards, 3);

        let pre = CancelFlag::new();
        pre.cancel();
        assert!(matches!(
            generate_step(&model, &[1], &cfg, &pre, &mut |_| {}, None),
            Err(Error::Canceled)
        ));

        let mid = CancelFlag::new();
        let signal = mid.clone();
        let (out, _) = generate_step(
            &model,
            &[1, 2],
            &cfg,
            &mid,
            &mut |e| {
                if matches!(e, StreamEvent::Token { .. }) {
                    signal.cancel();
                }
            },
            None,
        )
        .unwrap();
        assert_eq!(out.tokens, vec![2]);
        assert_eq!(out.finish_reason, FinishReason::Cancelled);

        assert!(matches!(
            generate_step(&model, &[], &cfg, &CancelFlag::new(), &mut |_| {}, None),
            Err(Error::Msg(_))
        ));
    }

    #[test]
    fn prefill_boundary_fires_once_before_the_first_token() {
        let model = counter();
        let cfg = GenerationConfig {
            max_new_tokens: 3,
            seed: Some(1),
            ..Default::default()
        };
        let boundary_hits = Cell::new(0);
        let tokens_at_boundary = Cell::new(usize::MAX);
        let emitted = Cell::new(0usize);
        let mut boundary = || {
            boundary_hits.set(boundary_hits.get() + 1);
            tokens_at_boundary.set(emitted.get());
            Ok(())
        };
        let (_, _, memory) = generate_step_timed(
            &model,
            &[1],
            &cfg,
            &CancelFlag::new(),
            &mut |e| {
                if matches!(e, StreamEvent::Token { .. }) {
                    emitted.set(emitted.get() + 1);
                }
            },
            None,
            Some(&mut boundary),
        )
        .unwrap();
        assert_eq!(boundary_hits.get(), 1);
        assert_eq!(tokens_at_boundary.get(), 0);
        assert_eq!(emitted.get(), 3);
        // The memory is the final cache's: prompt (1) + two fed-back tokens = 3 positions.
        assert_eq!(
            memory,
            CacheMemory {
                live_bytes: 12,
                checkpoint_bytes: 1
            }
        );
    }
}

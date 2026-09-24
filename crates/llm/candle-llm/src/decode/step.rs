//! The step-model seam (epic sc-24128, story sc-24129).
//!
//! [`StepModel`] is the one thing the fast-decode machinery asks of a model: *run one decode step
//! over N input tokens against a [`DecodeCache`], and give me logits for the last (or every)
//! position*. Everything the epic layers on top — the unified speculative engine (drafts verified in
//! one N-token step, rejected suffixes dropped with [`DecodeCache::rollback_to`]), static KV, the
//! CUDA-graph runner — is written once against this trait; model files only implement it.
//!
//! [`generate_step`] is the token-at-a-time driver over the seam — since sc-24140 a thin wrapper
//! over the one [`engine`](super::engine) loop with no proposer, so the seam has exactly one loop.
//! It keeps the reference loop's algorithm (see [`stream`](super::stream): same prefill, same
//! sampler routing and seeded stream, same stop / cancel / constraint order) so its output is
//! token-identical to the reference path for the same prompt and config — the parity gate the
//! tiny-config and real-weight tests hold. Unlike the reference loop it returns a
//! [`DecodeRecord`] with measured counters, including which KV cache the request ran on
//! (`kv_cache`). The cache is built through [`StepModel::new_cache_for`] with the request's bound
//! (prompt + budget), which is where a preallocated KV cache (story sc-24132) is sized and where
//! an over-budget request fails closed.

use candle_core::{Device, Tensor};

use crate::decode::cancel::CancelFlag;
use crate::decode::engine::RewindableConstraintMask;
use crate::decode::record::{DecodePath, DecodeRecord, RequestSpan};
use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::{
    ConstraintMask, FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::{Error, Result};
use crate::primitives::attention::AttnFormulation;
use crate::primitives::decode_cache::{CacheMemory, DecodeCache};

/// Which positions' logits a step returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LogitsScope {
    /// Only the last input position: logits `[batch, vocab]`.
    Last,
    /// Every input position: logits `[batch, n, vocab]` (the verify forward of a speculative step).
    All,
}

/// The tokens one step feeds: host ids, or ids already on the model's device (story sc-24130).
///
/// A greedy proposer keeps its draft tokens on the device (the argmax tensor of each draft step)
/// and hands the verify step `[cur, drafts…]` as a device tensor, so drafting and verifying issue
/// no device->host transfer of their own; the drafts reach the host once, inside the verify
/// decision's single transfer.
#[derive(Clone, Copy, Debug)]
pub enum StepTokens<'a> {
    /// Token ids on the host.
    Host(&'a [i32]),
    /// A `[1, n]` `u32` id tensor on the model's device.
    Device(&'a Tensor),
}

impl StepTokens<'_> {
    /// How many tokens the step feeds.
    pub fn len(&self) -> Result<usize> {
        Ok(match self {
            StepTokens::Host(tokens) => tokens.len(),
            StepTokens::Device(ids) => ids.dims2()?.1,
        })
    }

    /// `len() == 0`.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// The ids as a `[1, n]` `u32` tensor on `device` (built from host ids, or the device tensor
    /// itself — no copy).
    pub fn ids(&self, device: &Device) -> Result<Tensor> {
        match self {
            StepTokens::Host(tokens) => crate::primitives::input_ids(tokens, device),
            StepTokens::Device(ids) => Ok((*ids).clone()),
        }
    }
}

/// One decode step's inputs.
#[derive(Clone, Copy, Debug)]
pub struct StepRequest<'a> {
    /// The tokens to feed, in order; they occupy positions `cache.len() .. cache.len() + n`.
    pub tokens: StepTokens<'a>,
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
            tokens: StepTokens::Host(tokens),
            scope: LogitsScope::Last,
            want_hidden: false,
        }
    }

    /// All-position logits over `tokens`, no hidden states — the verify step.
    pub fn all(tokens: &'a [i32]) -> Self {
        Self {
            tokens: StepTokens::Host(tokens),
            scope: LogitsScope::All,
            want_hidden: false,
        }
    }

    /// Last-position logits over device-resident ids `[1, n]`.
    pub fn last_ids(ids: &'a Tensor) -> Self {
        Self {
            tokens: StepTokens::Device(ids),
            scope: LogitsScope::Last,
            want_hidden: false,
        }
    }

    /// All-position logits over device-resident ids `[1, n]` — the verify step of a proposer
    /// whose drafts live on the device.
    pub fn all_ids(ids: &'a Tensor) -> Self {
        Self {
            tokens: StepTokens::Device(ids),
            scope: LogitsScope::All,
            want_hidden: false,
        }
    }

    /// The same request, also returning the final-normalized hidden states.
    pub fn with_hidden(mut self, want_hidden: bool) -> Self {
        self.want_hidden = want_hidden;
        self
    }

    /// How many tokens the step feeds.
    pub fn len(&self) -> Result<usize> {
        self.tokens.len()
    }

    /// `len() == 0`.
    pub fn is_empty(&self) -> Result<bool> {
        self.tokens.is_empty()
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

    /// A fresh, empty cache sized for a request that will hold at most `capacity + overshoot`
    /// positions: `capacity` is the prompt plus the generation budget, `overshoot` the positions a
    /// caller may write **past** that budget before it decides what to keep — a speculative verify
    /// step writes `K + 1` positions at once (the current token plus `K` drafts), so it can run
    /// past the budget by up to `K` positions before rolling back, and must ask for them here
    /// (the S2 draft-model story). The token-at-a-time [`generate_step`] passes `0`.
    ///
    /// A model with a preallocated KV cache (story sc-24132) allocates it here, once, for exactly
    /// that bound, and fails closed with [`Error::KvCapacityExceeded`] when the bound exceeds
    /// what it can serve; the default is the unbounded [`new_cache`](Self::new_cache).
    ///
    /// ```
    /// use candle_core::{Device, Tensor};
    /// use candle_llm::decode::{StepModel, StepOutput, StepRequest};
    /// use candle_llm::error::{Error, Result};
    /// use candle_llm::primitives::{CacheMemory, DecodeCache};
    ///
    /// /// A cache that remembers the bound it was built for.
    /// struct Bounded { len: i32, capacity: usize }
    /// impl DecodeCache for Bounded {
    ///     fn len(&self) -> i32 { self.len }
    ///     fn rollback_to(&mut self, n: i32) -> Result<()> { self.len = n; Ok(()) }
    ///     fn reset(&mut self) { self.len = 0; }
    ///     fn memory(&self) -> CacheMemory { CacheMemory { live_bytes: 0, checkpoint_bytes: 0 } }
    /// }
    ///
    /// struct Model;
    /// impl StepModel for Model {
    ///     type Cache = Bounded;
    ///     fn new_cache(&self) -> Bounded { Bounded { len: 0, capacity: usize::MAX } }
    ///     /// The bound is the budget *plus* the overshoot the caller declared.
    ///     fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<Bounded> {
    ///         let capacity = capacity.saturating_add(overshoot);
    ///         if capacity > 1024 {
    ///             return Err(Error::KvCapacityExceeded { requested: capacity, capacity: 1024 });
    ///         }
    ///         Ok(Bounded { len: 0, capacity })
    ///     }
    ///     fn device(&self) -> &Device { &Device::Cpu }
    ///     fn vocab_size(&self) -> usize { 4 }
    ///     fn forward_step(&self, cache: &mut Bounded, request: StepRequest<'_>) -> Result<StepOutput> {
    ///         cache.len += request.len()? as i32;
    ///         Ok(StepOutput { logits: Tensor::zeros((1, 4), candle_core::DType::F32, &Device::Cpu)?, hidden: None })
    ///     }
    /// }
    ///
    /// // A verify step with K = 3 drafts writes 4 positions at once: the request needs 3 positions
    /// // past `prompt + max_new_tokens`, and asks for them as the overshoot.
    /// let (prompt, max_new_tokens, drafts) = (97usize, 256usize, 3usize);
    /// let cache = Model.new_cache_for(prompt + max_new_tokens, drafts).unwrap();
    /// assert_eq!(cache.capacity, 356);
    /// // Past the model's bound the request fails closed before anything is allocated.
    /// assert!(matches!(
    ///     Model.new_cache_for(1000, 25),
    ///     Err(Error::KvCapacityExceeded { requested: 1025, capacity: 1024 })
    /// ));
    /// ```
    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<Self::Cache> {
        let _ = (capacity, overshoot);
        Ok(self.new_cache())
    }

    /// How grouped-query attention was computed for a request running on `cache` (story sc-24132),
    /// stamped on the [`DecodeRecord`] so an evidence row says which arithmetic produced its
    /// tokens — what actually ran on that cache, not merely what was configured (a preallocated
    /// static cache always attends un-expanded, whatever the model's selector says). The default
    /// is [`AttnFormulation::Gqa`] — the un-expanded formulation every path runs since S4.
    fn attn_formulation(&self, cache: &Self::Cache) -> AttnFormulation {
        let _ = cache;
        AttnFormulation::Gqa
    }

    /// Whether this model's step can be captured as a CUDA graph at all (story sc-24134, E5):
    /// `Err` names a known reason the step is not replayable — a device->host read inside the
    /// step (a MoE router that pulls its probabilities to the host), positions or offsets that
    /// only exist as Rust-side scalars. The runner checks this before any capture; the default
    /// is `Ok` and the runner's census of the captured graph is the second gate.
    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        Ok(())
    }

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
/// measured [`DecodeRecord`] (`path == StepModel`, `proposer == none`). Token-identical to
/// [`generate_with`](super::generate_with) for the same inputs.
///
/// A thin wrapper (sc-24140): the token-at-a-time loop over the seam **is** the one speculative
/// engine with no proposer ([`generate_speculative_with`] with [`NoProposer`] and `K = 0`), so the
/// seam has one loop, one sampler routing and one record convention — the engine's (the prefill
/// counted in `prefill_forwards`, every later forward a verify step with its host syncs in
/// `verify_host_syncs`).
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference.
///
/// [`generate_speculative_with`]: super::generate_speculative_with
/// [`NoProposer`]: super::NoProposer
pub fn generate_step<M: StepModel + ?Sized>(
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
/// what a full-length request costs rather than a fresh cache's. The cache is built through
/// [`StepModel::new_cache_for`] with the request's bound (prompt + budget, no overshoot).
pub fn generate_step_timed<M: StepModel + ?Sized>(
    model: &M,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    constraint: Option<&mut dyn ConstraintMask>,
    on_prefill_complete: Option<&mut dyn FnMut() -> Result<()>>,
) -> Result<(GenerationOutput, DecodeRecord, CacheMemory)> {
    let mut commit_only = constraint.map(|inner| CommitOnly { inner, accepted: 0 });
    let run = super::engine::generate_speculative_with(
        model,
        &mut super::engine::NoProposer,
        super::engine::SpeculativePrompt::Tokens(prompt_ids),
        config,
        0,
        cancel,
        on_event,
        commit_only
            .as_mut()
            .map(|c| c as &mut dyn RewindableConstraintMask),
        None,
        on_prefill_complete,
    )?;
    Ok((run.output, run.record, run.memory))
}

/// A plain [`ConstraintMask`] as the engine's [`RewindableConstraintMask`], for a run with **no
/// drafts**: with `K = 0` the engine advances the constraint only by tokens it emits, so there is
/// never a speculative advance to undo. The adapter checks that rather than trusting it — a rewind
/// that would have to undo an advance panics instead of silently leaving the constraint ahead.
struct CommitOnly<'a, 'c> {
    inner: &'a mut (dyn ConstraintMask + 'c),
    accepted: usize,
}

impl ConstraintMask for CommitOnly<'_, '_> {
    fn allowed(&mut self) -> &[bool] {
        self.inner.allowed()
    }

    fn accept(&mut self, token: i32) {
        self.accepted += 1;
        self.inner.accept(token);
    }
}

impl RewindableConstraintMask for CommitOnly<'_, '_> {
    fn checkpoint(&self) -> usize {
        self.accepted
    }

    fn rewind(&mut self, checkpoint: usize) {
        assert_eq!(
            checkpoint, self.accepted,
            "a plain ConstraintMask cannot rewind: the token-at-a-time run advanced it speculatively"
        );
    }
}

/// Decode the continuation of a **caller-prefilled** request through the step seam (sc-24138):
/// the one speculative engine with no proposer — the token-at-a-time loop — over the request's own
/// `cache`, positioned past the prompt by the caller's own prefill (a multimodal splice: LLaVA's
/// image rows, the StarVector conditioning prefix). `logits` are the prefill's last-position row,
/// `history` the effective prompt ids (the repetition-penalty window). Same sampler, stop, cancel
/// and caller-stop contract as [`generate_speculative_with`](super::generate_speculative_with),
/// whose record (`path = StepModel`, `proposer = none`) is returned beside the output.
///
/// A cancel that is already set when this is called — i.e. one that landed during the caller's
/// prefill — is the ordinary mid-stream cancellation here (no tokens,
/// [`FinishReason::Cancelled`]), not [`Error::Canceled`]: the caller has already run inference,
/// and that is what the multimodal providers' own loops reported before they moved onto the seam.
#[allow(clippy::too_many_arguments)]
pub fn generate_step_from_prefill<M: StepModel + ?Sized>(
    model: &M,
    cache: &mut M::Cache,
    logits: Tensor,
    history: &[i32],
    config: &GenerationConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
    should_stop: Option<&dyn Fn() -> bool>,
) -> Result<(GenerationOutput, DecodeRecord)> {
    // Brackets the call so a cancel that lands before the engine's own span still reports the
    // request's measured counters and tallies (the same record convention as a run).
    let span = RequestSpan::begin();
    let run = super::engine::generate_speculative_with(
        model,
        &mut super::engine::NoProposer,
        super::engine::SpeculativePrompt::Prefilled {
            cache: &mut *cache,
            logits,
            hidden: None,
            history,
            position_delta: 0,
            warm_proposer: false,
        },
        config,
        0,
        cancel,
        on_event,
        None,
        should_stop,
        None,
    );
    match run {
        Ok(run) => Ok((run.output, run.record)),
        Err(Error::Canceled) => {
            on_event(StreamEvent::Done {
                reason: FinishReason::Cancelled,
                generated: 0,
            });
            // The caller's prefill is still one of the request's target forwards, counted as the
            // engine counts it on the `Prefilled` path (`prefill_forwards`, sc-24131).
            let prefill = SpeculativeStats {
                forwards: 1,
                prefill_forwards: 1,
                ..SpeculativeStats::default()
            };
            Ok((
                GenerationOutput {
                    tokens: Vec::new(),
                    finish_reason: FinishReason::Cancelled,
                },
                DecodeRecord::speculative(DecodePath::StepModel, prefill, 0, span.counters())
                    .with_kv_cache(cache.kv_kind())
                    .with_attn_formulation(model.attn_formulation(cache))
                    .with_span_tallies(&span),
            ))
        }
        Err(error) => Err(error),
    }
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
            assert!(!request.is_empty().unwrap());
            self.steps.set(self.steps.get() + 1);
            cache.0 += request.len()? as i32;
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

    /// The adapter `generate_step` hands the engine: a plain constraint advanced only by emitted
    /// tokens, so a rewind to the latest checkpoint is a no-op — and a rewind that would have to
    /// undo an advance fails loudly instead of leaving the constraint ahead of the history.
    #[test]
    fn a_commit_only_constraint_refuses_to_rewind_an_advance() {
        struct Log(Vec<i32>);
        impl ConstraintMask for Log {
            fn allowed(&mut self) -> &[bool] {
                &[]
            }
            fn accept(&mut self, token: i32) {
                self.0.push(token);
            }
        }
        let mut log = Log(Vec::new());
        let mut adapter = CommitOnly {
            inner: &mut log,
            accepted: 0,
        };
        adapter.accept(3);
        let checkpoint = adapter.checkpoint();
        adapter.rewind(checkpoint);
        adapter.accept(4);
        let advanced =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| adapter.rewind(checkpoint)));
        assert!(advanced.is_err(), "rewinding past an advance must fail");
        assert_eq!(log.0, vec![3, 4], "every accept reaches the constraint");
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

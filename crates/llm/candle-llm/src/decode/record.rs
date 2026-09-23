//! The structured per-request decode record (epic sc-24128, story sc-24129).
//!
//! Every decode path — the reference `Decode` loop, the [`StepModel`](super::StepModel) driver,
//! native Qwen MTP, prompt-lookup — ends with a [`DecodeRecord`]: which path ran, how many target
//! forwards it took, how many tokens were proposed/accepted, and how many device→host transfers
//! the path issued. The epic's evidence rows (tok/s, acceptance rate, forwards per accepted token,
//! host syncs per token) are all derived from it, and the "which path ran" field is how the
//! reference path stays *visible* rather than merely present once the fast paths land (E2).
//!
//! The counters are **measured**, not derived: forwards are counted by [`CountingDecode`] or by
//! the loop that issued them, host syncs by the thread-local counter in
//! [`primitives::host_sync`](crate::primitives::host_sync) bracketed around the request, and the
//! fused-vs-reference primitive counts (sc-24137) by the thread-local tally in
//! [`primitives::fused`](crate::primitives::fused) bracketed the same way — so a request that
//! took the op-chain path for a leaf shows `reference` / `mixed` with the reason, never silently.

use std::cell::Cell;

use candle_core::{Device, Tensor};
use core_llm::ProposerKind;

use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::Decode;
use crate::error::Result;
use crate::primitives::attention::AttnFormulation;
use crate::primitives::fused::{fused_tally, FusedTally};
use crate::primitives::host_sync::host_sync_count;
use crate::primitives::kv_cache::{KvCache, KvCacheKind};

/// Which decode implementation produced a request's tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    /// The pre-epic token-at-a-time loop over the [`Decode`] trait — the parity oracle.
    Reference,
    /// The token-at-a-time loop over the [`StepModel`](super::StepModel) seam.
    StepModel,
    /// Native Qwen3.8 multi-token prediction with `drafts` proposals per verify forward (the
    /// unified engine over [`StepModel`](super::StepModel) with the MTP proposer, sc-24130).
    Mtp {
        /// Draft tokens requested per target verification pass.
        drafts: u32,
    },
    /// Prompt-lookup (n-gram) speculation: the unified engine with the n-gram proposer, or the
    /// pre-epic loop over a `CausalLm`.
    PromptLookup,
    /// Draft-model speculation: the unified engine with the draft-model proposer, or the pre-epic
    /// loop over a `CausalLm` pair.
    DraftModel,
}

impl DecodePath {
    /// Stable lower-case label for logs and evidence rows (`reference`, `step_model`, `mtp`, …).
    pub fn label(&self) -> &'static str {
        match self {
            DecodePath::Reference => "reference",
            DecodePath::StepModel => "step_model",
            DecodePath::Mtp { .. } => "mtp",
            DecodePath::PromptLookup => "prompt_lookup",
            DecodePath::DraftModel => "draft_model",
        }
    }
}

/// Measured counters for one generation request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeRecord {
    /// Which path ran.
    pub path: DecodePath,
    /// Target-model forward passes, including the prompt prefill.
    pub target_forwards: u64,
    /// Draft tokens proposed (0 on non-speculative paths).
    pub proposed_tokens: u64,
    /// Draft tokens accepted by target verification (0 on non-speculative paths).
    pub accepted_tokens: u64,
    /// Tokens emitted to the caller.
    pub generated_tokens: u64,
    /// Device→host transfers this crate issued while generating (see `primitives::host_sync`).
    pub host_syncs: u64,
    /// Which KV cache implementation the request ran on (story sc-24132): the growing reference
    /// cache or the preallocated static one. Reported from the cache itself
    /// ([`DecodeCache::kv_kind`](crate::primitives::DecodeCache::kv_kind)), so the row says what
    /// actually ran, not what was configured.
    pub kv_cache: KvCacheKind,
    /// How grouped-query attention was computed (story sc-24132): the un-expanded `gqa`
    /// formulation every path runs since S4, or the pre-S4 `expanded` (`repeat_kv`) arithmetic
    /// selected for a comparison row. Reported from the model, which owns the selector.
    pub attn_formulation: AttnFormulation,
    /// Which proposer ran (sc-24130): `none` on the token-at-a-time paths — including a request
    /// whose [`MtpMode::Auto`](core_llm::MtpMode::Auto) resolved to no proposer, which is thereby
    /// visible rather than a silent downgrade — else `mtp` / `ngram` / `draft`.
    pub proposer: ProposerKind,
    /// Verify steps the speculative engine took (0 on non-speculative paths).
    pub verify_steps: u64,
    /// Verify steps whose partial acceptance was recovered by a direct cache rollback into the
    /// verify step (sc-24131) — no extra target forward.
    pub direct_rollbacks: u64,
    /// Verify steps the cache could not roll back into, recovered by a rollback to the step start
    /// plus a replay forward of the kept prefix — one extra target forward each. `0` on a cache
    /// with per-token checkpoints (the `Qwen35Cache` since S3); the E2 "replay fallback" count.
    pub replay_fallbacks: u64,
    /// Device->host transfers issued inside those verify steps (proposing, verifying, deciding and
    /// committing), so `verify_host_syncs / verify_steps` is the engine's per-step sync cost — the
    /// AC2 figure, exactly `1.0` for a greedy run with device-resident drafts.
    pub verify_host_syncs: u64,
    /// Fused-vs-reference primitive leaf runs while generating (see `primitives::fused`): how many
    /// RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel, how many the op chain, and why
    /// the last op-chain run happened. `FusedTally::label` gives `fused` / `reference` / `mixed`.
    pub fused_primitives: FusedTally,
}

impl DecodeRecord {
    /// A record for a non-speculative path.
    pub fn plain(
        path: DecodePath,
        target_forwards: u64,
        generated: usize,
        host_syncs: u64,
    ) -> Self {
        Self {
            path,
            target_forwards,
            proposed_tokens: 0,
            accepted_tokens: 0,
            generated_tokens: generated as u64,
            host_syncs,
            kv_cache: KvCacheKind::Growing,
            attn_formulation: AttnFormulation::Gqa,
            proposer: ProposerKind::None,
            verify_steps: 0,
            direct_rollbacks: 0,
            replay_fallbacks: 0,
            verify_host_syncs: 0,
            fused_primitives: FusedTally::default(),
        }
    }

    /// The same record with `proposer` set — the engine stamps the proposer it ran.
    pub fn with_proposer(mut self, proposer: ProposerKind) -> Self {
        self.proposer = proposer;
        self
    }

    /// The same record with the verify-step counters set (see
    /// [`host_syncs_per_verify_step`](Self::host_syncs_per_verify_step)).
    pub fn with_verify_syncs(mut self, verify_host_syncs: u64) -> Self {
        self.verify_host_syncs = verify_host_syncs;
        self
    }

    /// The same record with `kv_cache` set — the step driver stamps the cache's own
    /// [`DecodeCache::kv_kind`](crate::primitives::DecodeCache::kv_kind) on it.
    pub fn with_kv_cache(mut self, kv_cache: KvCacheKind) -> Self {
        self.kv_cache = kv_cache;
        self
    }

    /// The same record with `attn_formulation` set — stamped from the model's own selector
    /// ([`StepModel::attn_formulation`](super::StepModel::attn_formulation)).
    pub fn with_attn_formulation(mut self, attn_formulation: AttnFormulation) -> Self {
        self.attn_formulation = attn_formulation;
        self
    }

    /// The same record with its fused-primitive tally filled in (from
    /// [`RequestSpan::fused_primitives`]).
    pub fn with_fused_primitives(mut self, tally: FusedTally) -> Self {
        self.fused_primitives = tally;
        self
    }

    /// A record from a speculative run's [`SpeculativeStats`].
    pub fn speculative(
        path: DecodePath,
        stats: SpeculativeStats,
        generated: usize,
        host_syncs: u64,
    ) -> Self {
        Self {
            path,
            target_forwards: stats.forwards as u64,
            proposed_tokens: stats.proposed as u64,
            accepted_tokens: stats.accepted as u64,
            generated_tokens: generated as u64,
            host_syncs,
            kv_cache: KvCacheKind::Growing,
            attn_formulation: AttnFormulation::Gqa,
            proposer: ProposerKind::None,
            verify_steps: stats.verify_steps as u64,
            direct_rollbacks: stats.direct_rollbacks as u64,
            replay_fallbacks: stats.replay_fallbacks as u64,
            verify_host_syncs: 0,
            fused_primitives: FusedTally::default(),
        }
    }

    /// Host syncs per verify step (`verify_host_syncs / verify_steps`), or `None` when no verify
    /// step ran.
    pub fn host_syncs_per_verify_step(&self) -> Option<f64> {
        (self.verify_steps > 0).then(|| self.verify_host_syncs as f64 / self.verify_steps as f64)
    }

    /// Target forwards per verify step — the verify forward plus any replay fallback:
    /// `(verify_steps + replay_fallbacks) / verify_steps` — or `None` when no verify step ran.
    /// Exactly `1.0` on a cache with per-token checkpoints (sc-24131 AC2).
    pub fn target_forwards_per_verify_step(&self) -> Option<f64> {
        (self.verify_steps > 0)
            .then(|| (self.verify_steps + self.replay_fallbacks) as f64 / self.verify_steps as f64)
    }

    /// `accepted / proposed`, or `None` when nothing was proposed (non-speculative paths).
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.proposed_tokens > 0)
            .then(|| self.accepted_tokens as f64 / self.proposed_tokens as f64)
    }

    /// Target forwards per emitted token (`< 1.0` means speculation paid off), or `None` when
    /// nothing was generated.
    pub fn forwards_per_generated_token(&self) -> Option<f64> {
        (self.generated_tokens > 0)
            .then(|| self.target_forwards as f64 / self.generated_tokens as f64)
    }

    /// Host syncs per emitted token, or `None` when nothing was generated.
    pub fn host_syncs_per_token(&self) -> Option<f64> {
        (self.generated_tokens > 0).then(|| self.host_syncs as f64 / self.generated_tokens as f64)
    }
}

/// Brackets a request on the current thread: constructed before the first forward, `finish`ed after
/// the last, so the host-sync delta is exactly that request's transfers.
#[derive(Debug)]
pub struct RequestSpan {
    host_syncs_at_start: u64,
    fused_at_start: FusedTally,
}

impl Default for RequestSpan {
    fn default() -> Self {
        Self::begin()
    }
}

impl RequestSpan {
    /// Mark the start of a request on this thread.
    pub fn begin() -> Self {
        Self {
            host_syncs_at_start: host_sync_count(),
            fused_at_start: fused_tally(),
        }
    }

    /// Host syncs recorded on this thread since [`begin`](Self::begin).
    pub fn host_syncs(&self) -> u64 {
        host_sync_count().wrapping_sub(self.host_syncs_at_start)
    }

    /// Fused-vs-reference primitive runs on this thread since [`begin`](Self::begin).
    pub fn fused_primitives(&self) -> FusedTally {
        fused_tally().since(&self.fused_at_start)
    }
}

/// A [`Decode`] adapter that counts the forwards it is asked for — how the reference loop's
/// `target_forwards` are *measured* rather than inferred from the token count.
pub struct CountingDecode<'a> {
    inner: &'a dyn Decode,
    forwards: Cell<u64>,
}

impl<'a> CountingDecode<'a> {
    /// Wrap `inner`; the count starts at zero.
    pub fn new(inner: &'a dyn Decode) -> Self {
        Self {
            inner,
            forwards: Cell::new(0),
        }
    }

    /// Forwards issued through this adapter so far.
    pub fn forwards(&self) -> u64 {
        self.forwards.get()
    }

    /// Count a forward the caller issued **directly** on the inner decoder (e.g. a multimodal
    /// prefill that bypasses [`Decode::step`]).
    pub fn note_external_forward(&self) {
        self.forwards.set(self.forwards.get() + 1);
    }
}

impl Decode for CountingDecode<'_> {
    fn make_cache(&self) -> Box<dyn KvCache> {
        self.inner.make_cache()
    }

    fn device(&self) -> &Device {
        self.inner.device()
    }

    fn step(&self, input_ids: &Tensor, cache: &mut dyn KvCache, offset: i32) -> Result<Tensor> {
        self.forwards.set(self.forwards.get() + 1);
        self.inner.step(input_ids, cache, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::kv_cache::ContiguousKvCache;

    struct Stub(Device);
    impl Decode for Stub {
        fn make_cache(&self) -> Box<dyn KvCache> {
            Box::new(ContiguousKvCache::new(0))
        }
        fn device(&self) -> &Device {
            &self.0
        }
        fn step(&self, _: &Tensor, _: &mut dyn KvCache, _: i32) -> Result<Tensor> {
            Ok(Tensor::new(&[[0f32, 1.]], &self.0)?)
        }
    }

    #[test]
    fn counting_decode_measures_forwards() {
        let stub = Stub(Device::Cpu);
        let counted = CountingDecode::new(&stub);
        let mut cache = counted.make_cache();
        let ids = Tensor::new(&[[1u32]], &Device::Cpu).unwrap();
        counted.step(&ids, cache.as_mut(), 0).unwrap();
        counted.step(&ids, cache.as_mut(), 1).unwrap();
        counted.note_external_forward();
        assert_eq!(counted.forwards(), 3);
    }

    #[test]
    fn ratios_are_none_without_denominators_and_exact_otherwise() {
        let plain = DecodeRecord::plain(DecodePath::Reference, 0, 0, 0);
        assert_eq!(plain.acceptance_rate(), None);
        assert_eq!(plain.forwards_per_generated_token(), None);
        assert_eq!(plain.host_syncs_per_token(), None);
        assert_eq!(plain.path.label(), "reference");
        assert_eq!(plain.kv_cache, KvCacheKind::Growing);
        assert_eq!(plain.attn_formulation, AttnFormulation::Gqa);
        assert_eq!(plain.attn_formulation.label(), "gqa");
        assert_eq!(plain.proposer, ProposerKind::None);
        assert_eq!(plain.proposer.label(), "none");
        assert_eq!(plain.host_syncs_per_verify_step(), None);
        let stamped = plain
            .with_kv_cache(KvCacheKind::Static)
            .with_attn_formulation(AttnFormulation::Expanded)
            .with_proposer(ProposerKind::Ngram);
        assert_eq!(stamped.kv_cache, KvCacheKind::Static);
        assert_eq!(stamped.attn_formulation, AttnFormulation::Expanded);
        assert_eq!(stamped.attn_formulation.label(), "expanded");
        assert_eq!(stamped.proposer, ProposerKind::Ngram);

        let spec = DecodeRecord::speculative(
            DecodePath::Mtp { drafts: 3 },
            SpeculativeStats {
                forwards: 6,
                proposed: 12,
                accepted: 6,
                verify_steps: 4,
                direct_rollbacks: 2,
                replay_fallbacks: 1,
            },
            10,
            20,
        )
        .with_proposer(ProposerKind::Mtp)
        .with_verify_syncs(4);
        assert_eq!(spec.acceptance_rate(), Some(0.5));
        assert_eq!(spec.forwards_per_generated_token(), Some(0.6));
        assert_eq!(spec.host_syncs_per_token(), Some(2.0));
        assert_eq!(spec.host_syncs_per_verify_step(), Some(1.0));
        assert_eq!((spec.direct_rollbacks, spec.replay_fallbacks), (2, 1));
        assert_eq!(spec.target_forwards_per_verify_step(), Some(1.25));
        assert_eq!(plain.target_forwards_per_verify_step(), None);
        assert_eq!(spec.path.label(), "mtp");
        assert_eq!(spec.proposer.label(), "mtp");
    }

    #[test]
    fn request_span_brackets_this_threads_host_syncs() {
        let span = RequestSpan::begin();
        crate::primitives::note_host_sync();
        assert_eq!(span.host_syncs(), 1);
    }

    #[test]
    fn request_span_brackets_this_threads_fused_primitive_runs() {
        let span = RequestSpan::begin();
        crate::primitives::fused::note_reference("shape");
        crate::primitives::fused::note_fused();
        let tally = span.fused_primitives();
        assert_eq!(tally.fused, 1);
        assert_eq!(tally.reference, 1);
        assert_eq!(tally.reference_reason, Some("shape"));
        assert_eq!(tally.label(), "mixed");
        let record =
            DecodeRecord::plain(DecodePath::Reference, 1, 1, 0).with_fused_primitives(tally);
        assert_eq!(record.fused_primitives, tally);
        assert_eq!(
            DecodeRecord::plain(DecodePath::Reference, 1, 1, 0)
                .fused_primitives
                .label(),
            "none"
        );
    }
}

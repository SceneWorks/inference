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
//! [`primitives::host_sync`](crate::primitives::host_sync) bracketed around the request.
//!
//! Story sc-24133 adds the sampler's half ([`SamplerTelemetry`]): which sampler path served the
//! request (`device`, or `host` with the reason), how many draws each path made, and how many whole
//! logits rows were copied to the host.

use std::cell::Cell;

use candle_core::{Device, Tensor};

use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::Decode;
use crate::error::Result;
use crate::primitives::host_sync::{
    host_sync_count, last_host_reason, sampler_counters, SamplerCounters,
};
use crate::primitives::kv_cache::KvCache;
use crate::primitives::sampler::SamplerPath;

/// Which decode implementation produced a request's tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodePath {
    /// The pre-epic token-at-a-time loop over the [`Decode`] trait — the parity oracle.
    Reference,
    /// The token-at-a-time loop over the [`StepModel`](super::StepModel) seam.
    StepModel,
    /// Native Qwen3.8 multi-token prediction with `drafts` proposals per verify forward.
    Mtp {
        /// Draft tokens requested per target verification pass.
        drafts: u32,
    },
    /// Prompt-lookup (n-gram) speculation over a `CausalLm`.
    PromptLookup,
    /// Draft-model speculation over a `CausalLm` pair.
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

/// Which sampler served a request and what it cost (story sc-24133). Measured by the thread-local
/// sampler counters bracketed around the request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerTelemetry {
    /// The request's sampler path: `Host(reason)` if any draw (or speculative distribution read)
    /// ran on the host — with the most recent host reason — else `Device`; `None` if nothing was
    /// sampled.
    pub path: Option<SamplerPath>,
    /// Tokens drawn on the device (greedy argmax or the device sampler).
    pub device_draws: u64,
    /// Host sampling decisions (host draws and speculative distribution reads).
    pub host_draws: u64,
    /// Whole logits rows copied to the host (a vocabulary-wide `to_vec1`).
    pub logits_to_host: u64,
}

impl SamplerTelemetry {
    /// `device`, `host:<reason>` (e.g. `host:penalty`), or `none` — the evidence-row label.
    pub fn label(&self) -> String {
        self.path
            .map_or_else(|| "none".to_string(), |path| path.to_string())
    }
}

/// A request's measured transfer counters: host syncs plus the sampler's telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpanCounters {
    /// Device→host transfers this crate issued.
    pub host_syncs: u64,
    /// The sampler's path and counters.
    pub sampler: SamplerTelemetry,
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
    /// Which sampler path served the request (`device` | `host` + reason) and its counters.
    pub sampler: SamplerTelemetry,
}

impl DecodeRecord {
    /// A record for a non-speculative path.
    pub fn plain(
        path: DecodePath,
        target_forwards: u64,
        generated: usize,
        counters: SpanCounters,
    ) -> Self {
        Self {
            path,
            target_forwards,
            proposed_tokens: 0,
            accepted_tokens: 0,
            generated_tokens: generated as u64,
            host_syncs: counters.host_syncs,
            sampler: counters.sampler,
        }
    }

    /// A record from a speculative run's [`SpeculativeStats`].
    pub fn speculative(
        path: DecodePath,
        stats: SpeculativeStats,
        generated: usize,
        counters: SpanCounters,
    ) -> Self {
        Self {
            path,
            target_forwards: stats.forwards as u64,
            proposed_tokens: stats.proposed as u64,
            accepted_tokens: stats.accepted as u64,
            generated_tokens: generated as u64,
            host_syncs: counters.host_syncs,
            sampler: counters.sampler,
        }
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

    /// Whole logits rows copied to the host per emitted token, or `None` when nothing was
    /// generated. `0.0` on the device sampler path.
    pub fn logits_to_host_per_token(&self) -> Option<f64> {
        (self.generated_tokens > 0)
            .then(|| self.sampler.logits_to_host as f64 / self.generated_tokens as f64)
    }
}

/// Brackets a request on the current thread: constructed before the first forward, `finish`ed after
/// the last, so the host-sync delta is exactly that request's transfers.
#[derive(Debug)]
pub struct RequestSpan {
    host_syncs_at_start: u64,
    sampler_at_start: SamplerCounters,
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
            sampler_at_start: sampler_counters(),
        }
    }

    /// Host syncs recorded on this thread since [`begin`](Self::begin).
    pub fn host_syncs(&self) -> u64 {
        host_sync_count().wrapping_sub(self.host_syncs_at_start)
    }

    /// The sampler's telemetry on this thread since [`begin`](Self::begin).
    pub fn sampler(&self) -> SamplerTelemetry {
        let now = sampler_counters();
        let start = self.sampler_at_start;
        let device_draws = now.device_draws.wrapping_sub(start.device_draws);
        let host_draws = now.host_draws.wrapping_sub(start.host_draws);
        let path = if host_draws > 0 {
            last_host_reason().map(SamplerPath::Host)
        } else if device_draws > 0 {
            Some(SamplerPath::Device)
        } else {
            None
        };
        SamplerTelemetry {
            path,
            device_draws,
            host_draws,
            logits_to_host: now.logits_to_host.wrapping_sub(start.logits_to_host),
        }
    }

    /// Everything measured on this thread since [`begin`](Self::begin).
    pub fn counters(&self) -> SpanCounters {
        SpanCounters {
            host_syncs: self.host_syncs(),
            sampler: self.sampler(),
        }
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
        let plain = DecodeRecord::plain(DecodePath::Reference, 0, 0, SpanCounters::default());
        assert_eq!(plain.acceptance_rate(), None);
        assert_eq!(plain.forwards_per_generated_token(), None);
        assert_eq!(plain.host_syncs_per_token(), None);
        assert_eq!(plain.logits_to_host_per_token(), None);
        assert_eq!(plain.sampler.label(), "none");
        assert_eq!(plain.path.label(), "reference");

        let spec = DecodeRecord::speculative(
            DecodePath::Mtp { drafts: 3 },
            SpeculativeStats {
                forwards: 5,
                proposed: 12,
                accepted: 6,
            },
            10,
            SpanCounters {
                host_syncs: 20,
                sampler: SamplerTelemetry {
                    path: Some(SamplerPath::Device),
                    device_draws: 10,
                    host_draws: 0,
                    logits_to_host: 0,
                },
            },
        );
        assert_eq!(spec.acceptance_rate(), Some(0.5));
        assert_eq!(spec.forwards_per_generated_token(), Some(0.5));
        assert_eq!(spec.host_syncs_per_token(), Some(2.0));
        assert_eq!(spec.path.label(), "mtp");
        assert_eq!(spec.logits_to_host_per_token(), Some(0.0));
        assert_eq!(spec.sampler.label(), "device");
    }

    #[test]
    fn request_span_brackets_this_threads_host_syncs() {
        let span = RequestSpan::begin();
        crate::primitives::note_host_sync();
        assert_eq!(span.host_syncs(), 1);
    }

    #[test]
    fn request_span_names_the_sampler_path_and_reason() {
        use crate::primitives::{note_logits_to_host, note_sampler_path, HostSampleReason};
        let span = RequestSpan::begin();
        assert_eq!(span.sampler().path, None, "nothing sampled yet");
        note_sampler_path(SamplerPath::Device);
        assert_eq!(span.sampler().path, Some(SamplerPath::Device));
        note_sampler_path(SamplerPath::Host(HostSampleReason::Constraint));
        note_logits_to_host();
        let telemetry = span.counters().sampler;
        assert_eq!(
            telemetry.path,
            Some(SamplerPath::Host(HostSampleReason::Constraint)),
            "any host draw makes the request a host request, with its reason"
        );
        assert_eq!(telemetry.label(), "host:constraint");
        assert_eq!(
            (
                telemetry.device_draws,
                telemetry.host_draws,
                telemetry.logits_to_host
            ),
            (1, 1, 1)
        );
        // A later span on the same thread starts clean.
        let next = RequestSpan::begin();
        note_sampler_path(SamplerPath::Device);
        assert_eq!(next.sampler().path, Some(SamplerPath::Device));
    }
}

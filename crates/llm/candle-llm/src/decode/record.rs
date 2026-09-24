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
//!
//! Story sc-24137 adds the fused-vs-reference primitive counts, by the thread-local tally in
//! [`primitives::fused`](crate::primitives::fused) bracketed the same way — so a request that
//! took the op-chain path for a leaf shows `reference` / `mixed` with the reason, never silently.

use std::cell::Cell;

use candle_core::{Device, Tensor};
use core_llm::ProposerKind;

use crate::decode::graph::{graph_tally, GraphTally};
use crate::decode::speculative::SpeculativeStats;
use crate::decode::stream::Decode;
use crate::error::Result;
use crate::primitives::attention::AttnFormulation;
use crate::primitives::fused::{fused_tally, FusedTally};
use crate::primitives::host_sync::{
    host_sync_count, last_host_reason, sampler_counters, SamplerCounters,
};
use crate::primitives::kv_cache::{KvCache, KvCacheKind};
use crate::primitives::nvfp4_path::{nvfp4_path_tally, Nvfp4PathTally};
use crate::primitives::sampler::SamplerPath;

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
    /// The prefill forwards among `target_forwards`, as the speculative engine counted them
    /// ([`SpeculativeStats::prefill_forwards`]: one whenever the prompt was prefilled, by the
    /// engine or by its caller). `0` on a non-speculative record, which keeps its prefill inside
    /// `target_forwards` without splitting it out and has no verify step to divide by.
    pub prefill_forwards: u64,
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
    /// Verify steps that fell back from a direct rollback to a step-start rollback plus a replay
    /// forward (sc-24130, E2): the engine's `RollbackUnavailable` → replay recovery made visible.
    /// `0` on non-speculative paths and on a cache with per-position rollback (the `Qwen35Cache`
    /// since its per-token checkpoint ring, sc-24131); on the S1 hybrid cache one per rejected
    /// verify step. Each is one of `target_forwards`.
    pub replay_forwards: u64,
    /// Device->host transfers issued inside those verify steps (proposing, verifying, deciding and
    /// committing), so `verify_host_syncs / verify_steps` is the engine's per-step sync cost — the
    /// AC2 figure, exactly `1.0` for a greedy run with device-resident drafts.
    pub verify_host_syncs: u64,
    /// Fused-vs-reference primitive leaf runs while generating (see `primitives::fused`): how many
    /// RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel, how many the op chain, and why
    /// the last op-chain run happened. `FusedTally::label` gives `fused` / `reference` / `mixed`.
    pub fused_primitives: FusedTally,
    /// CUDA-graph runner steps while generating (story sc-24134, see `decode::graph`): how many
    /// steps replayed a captured graph, how many ran eager, how many graphs were captured, and —
    /// when an eager step was a fallback — the named reason (`graph: … fallback=<reason>` in
    /// [`GraphTally::describe`]). `none` when no runner was involved (the reference paths).
    pub cuda_graphs: GraphTally,
    /// NVFP4 projection calls by path while generating (see `primitives::nvfp4_path`, sc-24136):
    /// how many ran the fused decode GEMV, how many the cuBLASLt W4A4 GEMM, and why the last
    /// cuBLASLt run happened (`rows` for a prefill, `disabled` with the switch off, …).
    /// `Nvfp4PathTally::label` gives `gemv` / `cublaslt` / `mixed` / `none` (a non-NVFP4 model).
    pub nvfp4_projections: Nvfp4PathTally,
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
            prefill_forwards: 0,
            proposed_tokens: 0,
            accepted_tokens: 0,
            generated_tokens: generated as u64,
            host_syncs: counters.host_syncs,
            sampler: counters.sampler,
            kv_cache: KvCacheKind::Growing,
            attn_formulation: AttnFormulation::Gqa,
            proposer: ProposerKind::None,
            verify_steps: 0,
            direct_rollbacks: 0,
            replay_forwards: 0,
            verify_host_syncs: 0,
            fused_primitives: FusedTally::default(),
            cuda_graphs: GraphTally::default(),
            nvfp4_projections: Nvfp4PathTally::default(),
        }
    }

    /// The same record with its CUDA-graph tally filled in (from [`RequestSpan::cuda_graphs`]).
    pub fn with_cuda_graphs(mut self, tally: GraphTally) -> Self {
        self.cuda_graphs = tally;
        self
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

    /// The same record with its NVFP4 projection path tally filled in (from
    /// [`RequestSpan::nvfp4_projections`]).
    pub fn with_nvfp4_projections(mut self, tally: Nvfp4PathTally) -> Self {
        self.nvfp4_projections = tally;
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

    /// The same record with the host-side counters and the fused / CUDA-graph / NVFP4 tallies of
    /// the whole request measured by `span` (sc-24139) — for a caller that prefills before handing
    /// the engine a [`Prefilled`](super::SpeculativePrompt::Prefilled) prompt, whose own span
    /// starts after that prefill. The engine-measured fields (path, forwards, cache, proposer)
    /// are kept.
    pub fn with_request_span(self, span: &RequestSpan) -> Self {
        let counters = span.counters();
        Self {
            host_syncs: counters.host_syncs,
            sampler: counters.sampler,
            fused_primitives: span.fused_primitives(),
            cuda_graphs: span.cuda_graphs(),
            nvfp4_projections: span.nvfp4_projections(),
            ..self
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
            prefill_forwards: stats.prefill_forwards as u64,
            proposed_tokens: stats.proposed as u64,
            accepted_tokens: stats.accepted as u64,
            generated_tokens: generated as u64,
            host_syncs: counters.host_syncs,
            sampler: counters.sampler,
            kv_cache: KvCacheKind::Growing,
            attn_formulation: AttnFormulation::Gqa,
            proposer: ProposerKind::None,
            verify_steps: stats.verify_steps as u64,
            direct_rollbacks: stats.direct_rollbacks as u64,
            replay_forwards: stats.replays as u64,
            verify_host_syncs: 0,
            fused_primitives: FusedTally::default(),
            cuda_graphs: GraphTally::default(),
            nvfp4_projections: Nvfp4PathTally::default(),
        }
    }

    /// Host syncs per verify step (`verify_host_syncs / verify_steps`), or `None` when no verify
    /// step ran.
    pub fn host_syncs_per_verify_step(&self) -> Option<f64> {
        (self.verify_steps > 0).then(|| self.verify_host_syncs as f64 / self.verify_steps as f64)
    }

    /// Target forwards per verify step, from the **measured** forwards:
    /// `(target_forwards - prefill_forwards) / verify_steps` — the verify forward, plus any replay
    /// fallback, plus any other target forward the run spent — or `None` when no verify step ran.
    /// Exactly `1.0` on a cache with per-token checkpoints (sc-24131 AC2); a forward that is
    /// neither a verify step nor a counted replay shows up here rather than disappearing.
    pub fn target_forwards_per_verify_step(&self) -> Option<f64> {
        (self.verify_steps > 0).then(|| {
            self.target_forwards.saturating_sub(self.prefill_forwards) as f64
                / self.verify_steps as f64
        })
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

    /// The backend-neutral report a product renders (sc-24139): the same labels as the evidence
    /// rows. `cuda_graphs_enabled` is the graph switch the request ran under (the loaded model's
    /// `LoadSpec::cuda_graphs`, else the process switch at load), which the tally alone cannot
    /// say: a request that never reached the runner reads `none` either way.
    pub fn report(&self, cuda_graphs_enabled: bool) -> core_llm::DecodeReport {
        let draft_tokens = match self.path {
            DecodePath::Mtp { drafts } => Some(drafts),
            _ => None,
        };
        core_llm::DecodeReport {
            path: self.path.label().to_string(),
            proposer: self.proposer,
            draft_tokens,
            sampler: self.sampler.label(),
            kv_cache: self.kv_cache.label().to_string(),
            attention: self.attn_formulation.label().to_string(),
            cuda_graphs: core_llm::CudaGraphsReport {
                enabled: cuda_graphs_enabled,
                path: self.cuda_graphs.label().to_string(),
                replayed: self.cuda_graphs.replayed,
                eager: self.cuda_graphs.eager,
                captured: self.cuda_graphs.captured,
                fallback_reason: self.cuda_graphs.fallback_reason.map(str::to_string),
            },
            nvfp4_projections: core_llm::PathReport {
                path: self.nvfp4_projections.label().to_string(),
                reason: self.nvfp4_projections.cublaslt_reason.map(str::to_string),
            },
            fused_primitives: core_llm::PathReport {
                path: self.fused_primitives.label().to_string(),
                reason: self.fused_primitives.reference_reason.map(str::to_string),
            },
            target_forwards: self.target_forwards,
            proposed_tokens: self.proposed_tokens,
            accepted_tokens: self.accepted_tokens,
            replay_forwards: self.replay_forwards,
        }
    }
}

/// Brackets a request on the current thread: constructed before the first forward, `finish`ed after
/// the last, so the host-sync delta is exactly that request's transfers.
#[derive(Debug)]
pub struct RequestSpan {
    host_syncs_at_start: u64,
    sampler_at_start: SamplerCounters,
    fused_at_start: FusedTally,
    graphs_at_start: GraphTally,
    nvfp4_at_start: Nvfp4PathTally,
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
            fused_at_start: fused_tally(),
            graphs_at_start: graph_tally(),
            nvfp4_at_start: nvfp4_path_tally(),
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

    /// Fused-vs-reference primitive runs on this thread since [`begin`](Self::begin).
    pub fn fused_primitives(&self) -> FusedTally {
        fused_tally().since(&self.fused_at_start)
    }

    /// CUDA-graph runner steps on this thread since [`begin`](Self::begin).
    pub fn cuda_graphs(&self) -> GraphTally {
        graph_tally().since(&self.graphs_at_start)
    }

    /// NVFP4 projection calls by path on this thread since [`begin`](Self::begin).
    pub fn nvfp4_projections(&self) -> Nvfp4PathTally {
        nvfp4_path_tally().since(&self.nvfp4_at_start)
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
    fn the_report_carries_every_path_label_and_names_the_fallbacks() {
        let record = DecodeRecord::speculative(
            DecodePath::Mtp { drafts: 3 },
            SpeculativeStats {
                forwards: 5,
                proposed: 9,
                accepted: 6,
                verify_steps: 3,
                replays: 1,
                ..SpeculativeStats::default()
            },
            8,
            SpanCounters {
                host_syncs: 4,
                sampler: SamplerTelemetry {
                    path: Some(SamplerPath::Host(core_llm::HostSampleReason::Penalty)),
                    device_draws: 0,
                    host_draws: 8,
                    logits_to_host: 8,
                },
            },
        )
        .with_proposer(ProposerKind::Mtp)
        .with_kv_cache(KvCacheKind::Static)
        .with_cuda_graphs(GraphTally {
            replayed: 0,
            eager: 7,
            captured: 0,
            fallback_reason: Some("deltanet_state_unstable"),
        })
        .with_nvfp4_projections(Nvfp4PathTally {
            gemv: 40,
            cublaslt: 2,
            cublaslt_reason: Some("rows"),
        })
        .with_fused_primitives(FusedTally {
            fused: 10,
            reference: 0,
            reference_reason: None,
        });
        let report = record.report(true);
        assert_eq!(report.path, "mtp");
        assert_eq!(report.proposer, ProposerKind::Mtp);
        assert_eq!(report.draft_tokens, Some(3));
        assert_eq!(report.sampler, "host:penalty");
        assert_eq!(report.kv_cache, "static");
        assert_eq!(report.attention, "gqa");
        assert!(report.cuda_graphs.enabled);
        assert_eq!(report.cuda_graphs.path, "eager");
        assert_eq!(report.cuda_graphs.eager, 7);
        assert_eq!(
            report.cuda_graphs.fallback_reason.as_deref(),
            Some("deltanet_state_unstable")
        );
        assert_eq!(report.nvfp4_projections.path, "mixed");
        assert_eq!(report.nvfp4_projections.reason.as_deref(), Some("rows"));
        assert_eq!(report.fused_primitives.path, "fused");
        assert_eq!(report.fused_primitives.reason, None);
        assert_eq!(
            (
                report.target_forwards,
                report.proposed_tokens,
                report.accepted_tokens
            ),
            (5, 9, 6)
        );
        assert_eq!(report.replay_forwards, 1);

        // A reference run with the switch off: no proposer, no graph step, and the report says
        // the switch was off rather than leaving `none` ambiguous.
        let plain = DecodeRecord::plain(DecodePath::Reference, 3, 2, SpanCounters::default());
        let report = plain.report(false);
        assert_eq!(report.path, "reference");
        assert_eq!(report.proposer, ProposerKind::None);
        assert_eq!(report.draft_tokens, None);
        assert_eq!(report.sampler, "none");
        assert!(!report.cuda_graphs.enabled);
        assert_eq!(report.cuda_graphs.path, "none");
        assert_eq!(report.nvfp4_projections.path, "none");
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
                prefill_forwards: 1,
                proposed: 12,
                accepted: 6,
                verify_steps: 4,
                direct_rollbacks: 2,
                replays: 1,
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
        )
        .with_proposer(ProposerKind::Mtp)
        .with_verify_syncs(4);
        assert_eq!(spec.acceptance_rate(), Some(0.5));
        assert_eq!(spec.forwards_per_generated_token(), Some(0.6));
        assert_eq!(spec.host_syncs_per_token(), Some(2.0));
        assert_eq!(spec.host_syncs_per_verify_step(), Some(1.0));
        assert_eq!((spec.direct_rollbacks, spec.replay_forwards), (2, 1));
        assert_eq!(
            spec.replay_forwards, 1,
            "the replay fallback is on the record"
        );
        // 6 forwards = 1 prefill + 4 verify steps + 1 replay.
        assert_eq!(spec.prefill_forwards, 1);
        assert_eq!(spec.target_forwards_per_verify_step(), Some(1.25));
        assert_eq!(plain.target_forwards_per_verify_step(), None);
        // The ratio is derived from the measured forwards: 9 target forwards against 4 verify
        // steps + 1 replay + 1 prefill — three forwards that are neither a verify step nor a
        // counted replay — raise it, where the counters alone
        // (`(verify_steps + replay_forwards) / verify_steps`) would still say 1.25.
        let uncounted = DecodeRecord {
            target_forwards: 9,
            ..spec
        };
        assert_eq!(uncounted.target_forwards_per_verify_step(), Some(2.0));
        assert_eq!(spec.path.label(), "mtp");
        assert_eq!(spec.proposer.label(), "mtp");
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
    fn request_span_brackets_this_threads_cuda_graph_steps() {
        let span = RequestSpan::begin();
        assert_eq!(span.cuda_graphs(), GraphTally::default());
        assert_eq!(span.cuda_graphs().label(), "none");
        let record = DecodeRecord::plain(DecodePath::Reference, 1, 1, SpanCounters::default())
            .with_cuda_graphs(GraphTally {
                replayed: 3,
                eager: 1,
                captured: 1,
                fallback_reason: None,
            });
        assert_eq!(record.cuda_graphs.label(), "mixed");
        assert_eq!(
            record.cuda_graphs.describe(),
            "graph: mixed replayed=3 eager=1 captured=1"
        );
        assert_eq!(
            DecodeRecord::plain(DecodePath::Reference, 1, 1, SpanCounters::default())
                .cuda_graphs
                .label(),
            "none"
        );
    }

    /// sc-24139: a caller that prefills before the engine overlays its whole-request span on the
    /// engine's record — the span's counters and tallies replace the engine's, and the
    /// engine-measured fields stay.
    #[test]
    fn with_request_span_takes_the_whole_requests_counters() {
        let span = RequestSpan::begin();
        // Work before the engine's own span (the caller's prefill) ...
        crate::primitives::note_host_sync();
        crate::primitives::nvfp4_path::note_gemv();
        // ... which the engine's record, measured after it, does not see.
        let engine = DecodeRecord::plain(DecodePath::StepModel, 5, 4, SpanCounters::default())
            .with_kv_cache(KvCacheKind::Static)
            .with_proposer(ProposerKind::Ngram);
        let record = engine.with_request_span(&span);
        assert_eq!(record.host_syncs, 1);
        assert_eq!(record.nvfp4_projections.gemv, 1);
        assert_eq!(record.nvfp4_projections, span.nvfp4_projections());
        assert_eq!(record.fused_primitives, span.fused_primitives());
        assert_eq!(record.cuda_graphs, span.cuda_graphs());
        assert_eq!(
            (
                record.path,
                record.target_forwards,
                record.generated_tokens,
                record.kv_cache,
                record.proposer
            ),
            (
                DecodePath::StepModel,
                5,
                4,
                KvCacheKind::Static,
                ProposerKind::Ngram
            )
        );
    }

    #[test]
    fn request_span_brackets_this_threads_nvfp4_projection_paths() {
        let span = RequestSpan::begin();
        crate::primitives::nvfp4_path::note_cublaslt("rows");
        crate::primitives::nvfp4_path::note_gemv();
        crate::primitives::nvfp4_path::note_gemv();
        let tally = span.nvfp4_projections();
        assert_eq!((tally.gemv, tally.cublaslt), (2, 1));
        assert_eq!(tally.cublaslt_reason, Some("rows"));
        assert_eq!(tally.label(), "mixed");
        let record = DecodeRecord::plain(DecodePath::StepModel, 1, 1, SpanCounters::default())
            .with_nvfp4_projections(tally);
        assert_eq!(record.nvfp4_projections, tally);
        assert_eq!(
            DecodeRecord::plain(DecodePath::Reference, 1, 1, SpanCounters::default())
                .nvfp4_projections
                .label(),
            "none"
        );
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
        let record = DecodeRecord::plain(DecodePath::Reference, 1, 1, SpanCounters::default())
            .with_fused_primitives(tally);
        assert_eq!(record.fused_primitives, tally);
        assert_eq!(
            DecodeRecord::plain(DecodePath::Reference, 1, 1, SpanCounters::default())
                .fused_primitives
                .label(),
            "none"
        );
    }
}

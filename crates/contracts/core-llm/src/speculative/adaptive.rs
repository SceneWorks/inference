//! Adaptive speculative decoding policy: candidate pool resolution, per-step cost accounting,
//! dynamic depth adaptation, and bounded mid-request proposer switching (feature SW-F357,
//! story SW-7237).
//!
//! Both native engines (`mlx-llm` and `candle-llm`) drive this one backend-neutral controller
//! under [`Speculative::Auto`](crate::Speculative::Auto):
//!
//! * **Candidate pool ([`resolve_auto_candidates`])**: evaluates plain decode (`none`), `mtp`,
//!   `prompt_lookup`, compatible resident `draft_model` (within its context window), and eligible
//!   installed `block_draft` (DFlash2 / DSpine) from [`TextLlmCapabilities`](crate::TextLlmCapabilities).
//!   Never selects an unadvertised proposer or an out-of-context draft model.
//! * **Cost-aware decisions ([`StepCost`], [`AdaptiveSpeculativeController`])**: accounts for
//!   proposal, verification, rollback, and synchronization costs (`total_ns / committed_tokens`),
//!   skipping per-shape warm-up steps ([`SHAPE_WARMUP_STEPS`]) and filtering host stalls
//!   ([`STALL_STEP_FACTOR`]), while keeping MLX's pipelined-plain guard ([`PlainDecode::MlxPipelined`]).
//! * **Bounded adaptation without oscillation**: resizes proposal depth within advertised and
//!   provisioned bounds (`1..=max_depth`), switches proposers via bounded probing
//!   ([`ADAPTIVE_PROBE_STEPS`]) with improvement hysteresis ([`ADAPTIVE_SWITCH_IMPROVEMENT_MARGIN`]),
//!   enforces cooldown ([`ADAPTIVE_COOLDOWN_VERIFIES`]), caps total depth changes and switches per
//!   request, remembers ruled-out arms, and caps regressed depth growth so oscillation cannot occur.
//! * **Authoritative explicit controls**: explicit `off` and `{proposer, depth}` never resize,
//!   probe, switch, or demote, while still producing a structured [`AdaptiveSpeculativeReport`].

use super::{
    demotion_threshold, ngram_propose, DemotionBasis, MonitorDecision, PlainDecode, ProposerKind,
    Stretch, TimedStep, ACCEPTANCE_PROBE_VERIFIES, AUTO_BLOCK_DRAFT_NOT_ADMITTED, CLEAR_LOSS_GAIN,
    CLEAR_LOSS_MAX_TIMED_STEPS, CLEAR_LOSS_MIN_TIMED_STEPS, MEASURED_GAIN_MARGIN,
    MIN_TIMED_WINDOW_STEPS, PLAIN_PROBE_MAX_STEPS, PLAIN_PROBE_STEPS, SHAPE_WARMUP_STEPS,
};

/// Fallback named when `auto`'s compatible draft model did not fit this request's memory and
/// `auto` resolved again without it (SW-7237): a request `auto` would admit without a draft
/// model is never refused for it.
pub const AUTO_DRAFT_MODEL_NOT_ADMITTED: &str = "speculative: auto's draft model did not fit this \
     request's memory; resolved without it";

/// Verify steps in a bounded alternate-proposer probe: shape warm-up plus four timed steps.
pub const ADAPTIVE_PROBE_STEPS: u32 = SHAPE_WARMUP_STEPS + 4;

/// Minimum post-warmup timed steps required for a timed probe comparison to decide.
pub const ADAPTIVE_PROBE_MIN_TIMED_STEPS: u32 = 2;

/// Minimum verify steps between mid-request depth changes or proposer switches (anti-oscillation).
pub const ADAPTIVE_COOLDOWN_VERIFIES: u32 = 8;

/// Hard cap on mid-request proposal depth changes per request (anti-oscillation).
pub const ADAPTIVE_MAX_DEPTH_CHANGES: u32 = 6;

/// Hard cap on mid-request proposer switches per request, including fallback to plain.
pub const ADAPTIVE_MAX_SWITCHES: u32 = 4;

/// Acceptance ratio (`accepted / proposed`) and depth saturation (`accepted / (verifies * depth)`)
/// at or above which depth may grow toward `max_depth`.
pub const ADAPTIVE_GROW_ACCEPTANCE_RATIO: f64 = 0.75;

/// Minimum fraction of the depth window (`proposed / (verifies * depth)`) that must be filled
/// before depth may grow. Fixed-width proposers always have fill ratio `1.0`; for variable-width
/// `prompt_lookup`, this prevents growing depth when matches only partially fill the current depth.
pub const ADAPTIVE_GROW_FILL_RATIO: f64 = 0.80;

/// Acceptance ratio (`accepted / proposed`) below which depth shrinks (when `accepted > 0`).
pub const ADAPTIVE_SHRINK_ACCEPTANCE_RATIO: f64 = 0.35;

/// Relative reduction in `ns / committed_token` an alternate proposer probe must achieve over
/// the incumbent to commit a mid-request switch (8% hysteresis margin).
pub const ADAPTIVE_SWITCH_IMPROVEMENT_MARGIN: f64 = 0.08;

/// Relative increase in `tokens / verify` required on an untimed run for an alternate proposer
/// probe to commit a switch over the incumbent.
pub const ADAPTIVE_UNTIMED_SWITCH_GAIN: f64 = 1.25;

/// Consecutive steps where CPU `prompt_lookup` peek finds a match challenging the incumbent depth
/// before triggering a bounded probe from a GPU proposer to `prompt_lookup`.
pub const ADAPTIVE_LOOKUP_TRIGGER_MATCHES: u32 = 3;

/// Minimum incumbent verify steps observed before a GPU proposer may be interrupted to probe
/// `prompt_lookup`.
pub const ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES: u32 = 4;

/// Incumbent acceptance ratio (`accepted / proposed`) at or above which a GPU proposer is
/// considered near saturation and will not be interrupted to probe `prompt_lookup` on MLX.
pub const ADAPTIVE_LOOKUP_MAX_INCUMBENT_ACCEPT_RATIO: f64 = 0.70;

/// Stricter incumbent acceptance-ratio ceiling on Candle/CUDA, where probing variable-width
/// `prompt_lookup` incurs eager CUDA graph warm-ups and captures for new verify shapes.
pub const ADAPTIVE_LOOKUP_MAX_INCUMBENT_ACCEPT_RATIO_CANDLE: f64 = 0.55;

/// Consecutive zero-acceptance steps under `prompt_lookup` before probing an eligible synced
/// GPU proposer (if one is available and not ruled out).
pub const ADAPTIVE_LOOKUP_DRY_STEPS: u32 = 4;

/// Maximum trailing history tokens scanned by [`ngram_peek_match_len`] so CPU match peeking
/// stays O(1) bounded even on long prompts.
pub const NGRAM_PEEK_WINDOW_TOKENS: usize = 2048;

/// Maximum tokens buffered while an inactive `mtp` proposer waits for a switch-back before the
/// engine retires it to keep catch-up memory bounded.
pub const MAX_MTP_CATCHUP_TOKENS: usize = 128;

/// Maximum tokens committed while an inactive `draft_model` proposer shadows decode before the
/// engine retires it to bound inactive draft forward work.
pub const MAX_DRAFT_CATCHUP_TOKENS: usize = 256;

/// Zero-cost-on-GPU CPU check: how many draft tokens `prompt_lookup` would propose at the end of
/// `tokens`, scanning at most the trailing `max_window` tokens (0 = full history).
pub fn ngram_peek_match_len(
    tokens: &[i32],
    max_ngram: usize,
    max_proposal: usize,
    max_window: usize,
) -> usize {
    let slice = if max_window > 0 && tokens.len() > max_window {
        &tokens[tokens.len() - max_window..]
    } else {
        tokens
    };
    ngram_propose(slice, max_ngram, max_proposal).len()
}

/// Whether a request's memory refusal `error` withdraws `auto`'s draft model: the request asked
/// for `auto`, the route priced `draft_model`, and the refusal is request memory exhaustion.
pub fn withdraws_auto_draft_model(
    mode: crate::Speculative,
    route: ProposerKind,
    error: &crate::Error,
) -> bool {
    mode == crate::Speculative::Auto
        && route == ProposerKind::DraftModel
        && matches!(error, crate::Error::RequestResourceExhausted(_))
}

/// One eligible proposer candidate in an [`AutoSpeculativePlan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AutoCandidate {
    /// The proposer (`none` for plain decode).
    pub proposer: ProposerKind,
    /// Starting / recommended draft depth (`0` for plain decode).
    pub recommended_depth: u32,
    /// Maximum advertised / eligible draft depth (`0` for plain decode).
    pub max_depth: u32,
}

impl AutoCandidate {
    /// Plain token-at-a-time decode candidate.
    pub const fn plain() -> Self {
        Self {
            proposer: ProposerKind::None,
            recommended_depth: 0,
            max_depth: 0,
        }
    }
}

/// Request-scoped inputs to [`resolve_auto_candidates`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AutoCandidateContext {
    /// Whether recorded evidence prefers a resident block draft ahead of MTP (SW-7200).
    pub auto_prefers_block_draft: bool,
    /// Whether a resident block draft was withheld because request memory admission refused it.
    pub block_draft_withheld: bool,
    /// Whether a resident draft model was withheld because request memory admission refused it.
    pub draft_model_withheld: bool,
    /// Resident draft model's context window (`0` = unbounded).
    pub draft_context: usize,
    /// Request prompt token count (used to check `draft_model` context eligibility).
    pub prompt_tokens: usize,
    /// Requested maximum generated tokens (used to check `draft_model` context eligibility).
    pub max_new_tokens: u32,
}

/// Resolved candidate pool for [`Speculative::Auto`](crate::Speculative::Auto): all eligible
/// advertised proposers in preference order, terminated by plain decode (`none`), plus any
/// diagnostic reasons for excluded candidates or staying plain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AutoSpeculativePlan {
    /// Eligible candidates in priority order; always ends with [`AutoCandidate::plain`].
    pub candidates: Vec<AutoCandidate>,
    /// Diagnostic fallback reasons (e.g. withheld drafts, out-of-context draft model, or no
    /// advertised proposer).
    pub fallbacks: Vec<String>,
}

impl AutoSpeculativePlan {
    /// The highest-priority eligible candidate (`AutoCandidate::plain()` if none).
    pub fn primary(&self) -> AutoCandidate {
        self.candidates
            .first()
            .copied()
            .unwrap_or_else(AutoCandidate::plain)
    }

    /// Look up an eligible candidate by proposer kind.
    pub fn candidate(&self, kind: ProposerKind) -> Option<AutoCandidate> {
        self.candidates.iter().copied().find(|c| c.proposer == kind)
    }

    /// The candidate kinds in priority order.
    pub fn kinds(&self) -> Vec<ProposerKind> {
        self.candidates.iter().map(|c| c.proposer).collect()
    }

    /// Maximum depth across all non-plain candidates (`0` when only plain decode is eligible).
    pub fn max_depth(&self) -> u32 {
        self.candidates
            .iter()
            .map(|c| c.max_depth)
            .max()
            .unwrap_or(0)
    }

    /// Construct a single-proposer plan (plus plain decode) for direct engine callers that did
    /// not pre-resolve a full capability pool.
    pub fn single(proposer: ProposerKind, recommended_depth: u32, max_depth: u32) -> Self {
        if proposer == ProposerKind::None || recommended_depth == 0 {
            return Self {
                candidates: vec![AutoCandidate::plain()],
                fallbacks: Vec::new(),
            };
        }
        let max_depth = max_depth.max(recommended_depth).max(1);
        let recommended_depth = recommended_depth.clamp(1, max_depth);
        Self {
            candidates: vec![
                AutoCandidate {
                    proposer,
                    recommended_depth,
                    max_depth,
                },
                AutoCandidate::plain(),
            ],
            fallbacks: Vec::new(),
        }
    }

    /// Return a copy of this plan preserving all candidates and clamping every non-plain
    /// candidate's `max_depth` (and `recommended_depth`) to `cap.max(1)`.
    pub fn clamped_to(mut self, cap: u32) -> Self {
        for c in &mut self.candidates {
            if c.proposer != ProposerKind::None {
                let bound = cap.max(1);
                c.max_depth = c.max_depth.min(bound).max(1);
                c.recommended_depth = c.recommended_depth.clamp(1, c.max_depth);
            }
        }
        self
    }
}

/// Resolve all eligible `auto` candidates from `capabilities` and `ctx` (SW-7237 E1/AC1).
///
/// Considers:
/// * `block_draft` (when advertised and not withheld; ranked before `mtp` when
///   `ctx.auto_prefers_block_draft`, otherwise after `mtp`);
/// * `mtp` (when advertised);
/// * `draft_model` (when advertised, not withheld, and fits `ctx.draft_context` for
///   `prompt_tokens + max_new_tokens + depth + 1`);
/// * `prompt_lookup` (when advertised — zero GPU memory, always eligible alongside GPU proposers);
/// * `none` (plain decode baseline / fallback).
///
/// Never includes any unadvertised proposer.
pub fn resolve_auto_candidates(
    capabilities: &crate::TextLlmCapabilities,
    ctx: AutoCandidateContext,
) -> AutoSpeculativePlan {
    use crate::SpeculativeProposer;

    let mut candidates = Vec::new();
    let mut fallbacks = Vec::new();

    if ctx.block_draft_withheld {
        fallbacks.push(AUTO_BLOCK_DRAFT_NOT_ADMITTED.to_string());
    }
    if ctx.draft_model_withheld {
        fallbacks.push(AUTO_DRAFT_MODEL_NOT_ADMITTED.to_string());
    }

    let push_advertised = |candidates: &mut Vec<AutoCandidate>, proposer: SpeculativeProposer| {
        if let Some(cap) = capabilities.proposer(proposer) {
            let max_depth = cap.max_depth.max(1);
            let recommended_depth = cap.recommended_depth.clamp(1, max_depth);
            candidates.push(AutoCandidate {
                proposer: proposer.into(),
                recommended_depth,
                max_depth,
            });
        }
    };

    if ctx.auto_prefers_block_draft && !ctx.block_draft_withheld {
        push_advertised(&mut candidates, SpeculativeProposer::BlockDraft);
    }
    push_advertised(&mut candidates, SpeculativeProposer::Mtp);
    if !ctx.auto_prefers_block_draft && !ctx.block_draft_withheld {
        push_advertised(&mut candidates, SpeculativeProposer::BlockDraft);
    }

    // Compatible resident draft model: eligible only when advertised, admitted, and within its
    // own context window.
    if !ctx.draft_model_withheld {
        if let Some(cap) = capabilities.proposer(SpeculativeProposer::DraftModel) {
            let base = ctx
                .prompt_tokens
                .saturating_add(ctx.max_new_tokens as usize)
                .saturating_add(1);
            let max_depth = cap.max_depth.max(1);
            let rec_depth = cap.recommended_depth.clamp(1, max_depth);
            let fits = |depth: u32| {
                ctx.draft_context == 0 || base.saturating_add(depth as usize) <= ctx.draft_context
            };
            if fits(1) {
                let eligible_max = if ctx.draft_context == 0 {
                    max_depth
                } else {
                    let room = ctx.draft_context.saturating_sub(base);
                    max_depth
                        .min(u32::try_from(room).unwrap_or(u32::MAX))
                        .max(1)
                };
                candidates.push(AutoCandidate {
                    proposer: ProposerKind::DraftModel,
                    recommended_depth: rec_depth.min(eligible_max).max(1),
                    max_depth: eligible_max,
                });
            } else {
                fallbacks.push(format!(
                    "speculative: `draft_model` excluded from `auto`: prompt ({} tokens) + \
                     requested generation ({}) + verify step exceeds draft context window {}",
                    ctx.prompt_tokens, ctx.max_new_tokens, ctx.draft_context
                ));
            }
        }
    }

    push_advertised(&mut candidates, SpeculativeProposer::PromptLookup);

    if candidates.is_empty() {
        fallbacks.push(format!(
            "{} ({})",
            super::AUTO_FOUND_NO_PROPOSER,
            super::NO_PROPOSER_ADVERTISED
        ));
    }

    candidates.push(AutoCandidate::plain());
    AutoSpeculativePlan {
        candidates,
        fallbacks,
    }
}

/// Measured cost components for one decode step (nanoseconds + host sync count).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StepCost {
    /// Time spent proposing drafts (nanoseconds).
    pub proposal_ns: u64,
    /// Time spent running the target verify forward (nanoseconds).
    pub verify_ns: u64,
    /// Time spent recovering cache / replaying rejected prefixes (nanoseconds).
    pub rollback_ns: u64,
    /// Time spent in host synchronization and decision readback (nanoseconds).
    pub sync_ns: u64,
    /// End-to-end step wall time (nanoseconds), encompassing proposal, verify, sync, rollback,
    /// and proposer commit.
    pub total_ns: u64,
    /// Device->host synchronizations executed during this step.
    pub host_syncs: u64,
}

impl StepCost {
    /// Construct a `StepCost` when only total step wall time is available.
    pub const fn from_total(total_ns: u64) -> Self {
        Self {
            proposal_ns: 0,
            verify_ns: 0,
            rollback_ns: 0,
            sync_ns: 0,
            total_ns,
            host_syncs: 0,
        }
    }

    /// Effective total cost in nanoseconds: `total_ns` if populated, else the component sum.
    pub const fn effective_total_ns(self) -> u64 {
        if self.total_ns > 0 {
            self.total_ns
        } else {
            self.proposal_ns
                .saturating_add(self.verify_ns)
                .saturating_add(self.rollback_ns)
                .saturating_add(self.sync_ns)
        }
    }
}

/// Aggregate step cost breakdown across a request's measured decode steps (SW-7237 AC1/AC3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveCostSummary {
    /// Summed proposal time (nanoseconds).
    pub proposal_ns: u64,
    /// Summed target verify forward time (nanoseconds).
    pub verify_ns: u64,
    /// Summed rollback and replay recovery time (nanoseconds).
    pub rollback_ns: u64,
    /// Summed host synchronization / readback time (nanoseconds).
    pub sync_ns: u64,
    /// Summed end-to-end decode step time (nanoseconds).
    pub total_ns: u64,
    /// Summed device->host synchronizations across measured steps.
    pub host_syncs: u64,
    /// Total tokens committed across measured steps.
    pub committed_tokens: u64,
}

impl AdaptiveCostSummary {
    /// Record one step's costs and committed tokens.
    pub fn record(&mut self, cost: StepCost, committed_tokens: u64) {
        self.proposal_ns = self.proposal_ns.saturating_add(cost.proposal_ns);
        self.verify_ns = self.verify_ns.saturating_add(cost.verify_ns);
        self.rollback_ns = self.rollback_ns.saturating_add(cost.rollback_ns);
        self.sync_ns = self.sync_ns.saturating_add(cost.sync_ns);
        self.total_ns = self.total_ns.saturating_add(cost.effective_total_ns());
        self.host_syncs = self.host_syncs.saturating_add(cost.host_syncs);
        self.committed_tokens = self.committed_tokens.saturating_add(committed_tokens);
    }

    /// Summed proposal time in milliseconds.
    pub fn proposal_ms(&self) -> f64 {
        self.proposal_ns as f64 / 1e6
    }

    /// Summed verification time in milliseconds.
    pub fn verify_ms(&self) -> f64 {
        self.verify_ns as f64 / 1e6
    }

    /// Summed rollback / replay recovery time in milliseconds.
    pub fn rollback_ms(&self) -> f64 {
        self.rollback_ns as f64 / 1e6
    }

    /// Summed synchronization / readback time in milliseconds.
    pub fn sync_ms(&self) -> f64 {
        self.sync_ns as f64 / 1e6
    }

    /// Summed total step time in milliseconds.
    pub fn total_ms(&self) -> f64 {
        self.total_ns as f64 / 1e6
    }

    /// Milliseconds per committed token, or `None` when no tokens were committed.
    pub fn ms_per_committed_token(&self) -> Option<f64> {
        (self.committed_tokens > 0 && self.total_ns > 0)
            .then(|| self.total_ms() / self.committed_tokens as f64)
    }
}

/// Kind of decision recorded in [`AdaptiveSpeculativeReport::events`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveEventKind {
    /// Initial proposer and depth selected at request start.
    Selected,
    /// `auto` stayed on plain decoding (reason names why).
    StayedPlain,
    /// Bounded probe of an alternate proposer started.
    ProbeStarted,
    /// Alternate proposer probe beat incumbent (or incumbent failed); switched active proposer.
    ProposerSwitched,
    /// Alternate proposer probe did not beat incumbent; reverted and ruled out alternate.
    ProbeReverted,
    /// Proposal depth increased within advertised limits.
    DepthIncreased,
    /// Proposal depth decreased within advertised limits.
    DepthDecreased,
    /// Demoted / fell back to plain decoding.
    FallbackToPlain,
}

impl AdaptiveEventKind {
    /// Stable lower-case label for JSON and UI rendering.
    pub fn label(self) -> &'static str {
        match self {
            AdaptiveEventKind::Selected => "selected",
            AdaptiveEventKind::StayedPlain => "stayed_plain",
            AdaptiveEventKind::ProbeStarted => "probe_started",
            AdaptiveEventKind::ProposerSwitched => "proposer_switched",
            AdaptiveEventKind::ProbeReverted => "probe_reverted",
            AdaptiveEventKind::DepthIncreased => "depth_increased",
            AdaptiveEventKind::DepthDecreased => "depth_decreased",
            AdaptiveEventKind::FallbackToPlain => "fallback_to_plain",
        }
    }
}

/// One measured adaptive decision during a generation request (SW-7237 AC3/E4).
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveSpeculativeEvent {
    /// Generated tokens committed so far when the event occurred.
    pub token_offset: u64,
    /// Verify steps completed when the event occurred.
    pub verify_step: u64,
    /// Event kind.
    pub kind: AdaptiveEventKind,
    /// Effective proposer before this event.
    pub from_proposer: ProposerKind,
    /// Effective depth before this event (`0` for plain).
    pub from_depth: u32,
    /// Effective proposer after this event.
    pub to_proposer: ProposerKind,
    /// Effective depth after this event (`0` for plain).
    pub to_depth: u32,
    /// Human- and machine-readable reason for the decision.
    pub reason: String,
}

/// Per-request adaptive speculative telemetry carried on [`DecodeReport`](crate::DecodeReport).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AdaptiveSpeculativeReport {
    /// `true` when `Speculative::Auto` drove adaptive policy; `false` for explicit `off` or
    /// explicit `{proposer, depth}`.
    pub adaptive: bool,
    /// Eligible candidate proposers considered for this request (in preference order).
    pub candidates: Vec<ProposerKind>,
    /// Effective proposer at decode start.
    pub initial_proposer: ProposerKind,
    /// Effective depth at decode start (`0` for plain).
    pub initial_depth: u32,
    /// Effective proposer at decode end.
    pub final_proposer: ProposerKind,
    /// Effective depth at decode end (`0` for plain).
    pub final_depth: u32,
    /// Number of mid-request depth changes.
    pub depth_changes: u32,
    /// Number of mid-request proposer switches (including fallback to plain).
    pub proposer_switches: u32,
    /// Ordered decision trail for every selection, probe, resize, switch, and fallback.
    pub events: Vec<AdaptiveSpeculativeEvent>,
    /// Reason why the request ran plain or fell back to plain, if applicable.
    pub plain_reason: Option<String>,
    /// Aggregate cost summary across measured steps, if timed.
    pub cost: Option<AdaptiveCostSummary>,
}

impl AdaptiveSpeculativeReport {
    /// Report for an explicit non-adaptive request (`Speculative::Off` or explicit
    /// `Speculative::Proposer`), optionally carrying a `plain_reason` (e.g. `off` or fallback)
    /// and measured cost summary.
    pub fn explicit(
        proposer: ProposerKind,
        depth: u32,
        plain_reason: Option<String>,
        cost: Option<AdaptiveCostSummary>,
    ) -> Self {
        let effective_depth = if proposer == ProposerKind::None {
            0
        } else {
            depth
        };
        let reason = if proposer == ProposerKind::None {
            plain_reason
                .clone()
                .unwrap_or_else(|| "explicitly decoded without a proposer".into())
        } else {
            format!("explicit `{}` at depth {effective_depth}", proposer.label())
        };
        let kind = if proposer == ProposerKind::None {
            AdaptiveEventKind::StayedPlain
        } else {
            AdaptiveEventKind::Selected
        };
        Self {
            adaptive: false,
            candidates: vec![proposer],
            initial_proposer: proposer,
            initial_depth: effective_depth,
            final_proposer: proposer,
            final_depth: effective_depth,
            depth_changes: 0,
            proposer_switches: 0,
            events: vec![AdaptiveSpeculativeEvent {
                token_offset: 0,
                verify_step: 0,
                kind,
                from_proposer: proposer,
                from_depth: effective_depth,
                to_proposer: proposer,
                to_depth: effective_depth,
                reason,
            }],
            plain_reason: (proposer == ProposerKind::None)
                .then_some(plain_reason.unwrap_or_else(|| "speculative: off".into())),
            cost,
        }
    }

    /// Report for an `Auto` request that stayed on plain decoding before the decode loop began.
    pub fn auto_stayed_plain(reason: impl Into<String>, candidates: Vec<ProposerKind>) -> Self {
        let reason = reason.into();
        let candidates = if candidates.is_empty() {
            vec![ProposerKind::None]
        } else {
            candidates
        };
        Self {
            adaptive: true,
            candidates,
            initial_proposer: ProposerKind::None,
            initial_depth: 0,
            final_proposer: ProposerKind::None,
            final_depth: 0,
            depth_changes: 0,
            proposer_switches: 0,
            events: vec![AdaptiveSpeculativeEvent {
                token_offset: 0,
                verify_step: 0,
                kind: AdaptiveEventKind::StayedPlain,
                from_proposer: ProposerKind::None,
                from_depth: 0,
                to_proposer: ProposerKind::None,
                to_depth: 0,
                reason: reason.clone(),
            }],
            plain_reason: Some(reason),
            cost: None,
        }
    }
}

/// Inputs fed to [`AdaptiveSpeculativeController::observe_step`] after each decode step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdaptiveStepInput {
    /// Drafts accepted on this step.
    pub accepted: usize,
    /// Drafts verified on this step (`0` for plain probe or empty proposal).
    pub drafts: usize,
    /// Tokens committed on this step (`1 + accepted` normally).
    pub committed: usize,
    /// Total tokens generated so far (including this step's committed tokens).
    pub generated_tokens: u64,
    /// Measured cost breakdown for this step.
    pub cost: StepCost,
    /// Zero-cost CPU hint: match length `prompt_lookup` would propose at current history end.
    pub lookup_peek_len: usize,
}

/// What the engine should run on the upcoming decode step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptiveStepPlan {
    /// Proposer to run (`none` during initial plain probe or after fallback to plain).
    pub proposer: ProposerKind,
    /// Draft depth (`0` during initial plain probe or plain decode).
    pub depth: u32,
    /// `true` while collecting initial plain-step baseline timings.
    pub probing_plain: bool,
    /// `true` once permanently demoted to plain decode.
    pub demoted_to_plain: bool,
}

/// Action returned by [`AdaptiveSpeculativeController::observe_step`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveAction {
    /// Keep current proposer and depth.
    Continue,
    /// Resize proposal depth within advertised limits.
    ResizeDepth { from: u32, to: u32 },
    /// Switch active proposer (starting a bounded probe, committing an alternate, or reverting).
    SwitchProposer {
        from: ProposerKind,
        to: ProposerKind,
        depth: u32,
    },
    /// Permanently demote to plain decoding (triggers MLX pipelined handoff when applicable).
    DemoteToPlain,
}

#[derive(Clone, Debug)]
struct ProbeState {
    incumbent_proposer: ProposerKind,
    incumbent_depth: u32,
    incumbent_ns_per_token: Option<f64>,
    incumbent_tokens_per_verify: f64,
    probe_proposer: ProposerKind,
    probe_depth: u32,
    steps: u32,
    accepted: u64,
    committed: u64,
    timed: Vec<TimedStep>,
}

/// Backend-neutral adaptive speculative controller shared by MLX and Candle (SW-7237).
#[derive(Clone, Debug)]
pub struct AdaptiveSpeculativeController {
    adaptive: bool,
    plain: PlainDecode,
    timed: bool,
    /// All eligible candidates reported on `AdaptiveSpeculativeReport::candidates`.
    candidates: Vec<AutoCandidate>,
    /// Subset of `candidates` that the engine can actually execute mid-request (e.g. primary +
    /// optional `PromptLookup` + `None`).
    switchable: Vec<ProposerKind>,
    /// Proposers that lost a probe, were retired, or failed a window and must not be re-entered.
    ruled_out: Vec<ProposerKind>,
    /// Per-proposer dynamic max depth cap (reduced if growing past a depth regressed cost).
    depth_caps: Vec<(ProposerKind, u32)>,

    initial_proposer: ProposerKind,
    initial_depth: u32,
    active_proposer: ProposerKind,
    active_depth: u32,

    probe_steps: u32,
    plain_ns: Vec<u64>,
    shapes: Vec<(ProposerKind, usize, u32)>,

    total_verifies: u64,
    windows: u32,
    window_verifies: u32,
    window_proposed: u64,
    window_accepted: u64,
    window_committed: u64,
    window_timed: Vec<TimedStep>,

    clear_verifies: u32,
    clear_accepted: u64,
    clear_timed: Vec<TimedStep>,
    measured_window_seen: bool,

    incumbent_ns_per_token: Option<f64>,
    incumbent_tokens_per_verify: f64,
    incumbent_accept_ratio: Option<f64>,
    prev_depth_baseline: Option<(ProposerKind, u32, f64)>,

    active_probe: Option<ProbeState>,
    lookup_match_streak: u32,
    last_lookup_peek_len: usize,
    lookup_dry_streak: u32,

    cooldown_remaining: u32,
    depth_changes: u32,
    proposer_switches: u32,
    demoted: bool,
    last_monitor: Option<MonitorDecision>,
    events: Vec<AdaptiveSpeculativeEvent>,
    plain_reason: Option<String>,
    cost: AdaptiveCostSummary,
    has_timed_cost: bool,
}

impl AdaptiveSpeculativeController {
    /// Build a controller for a request running `mode`.
    ///
    /// * `Speculative::Off` / `Speculative::Proposer`: authoritative non-adaptive mode (never
    ///   resizes, switches, or demotes; records `adaptive = false` and aggregate costs).
    /// * `Speculative::Auto`: adaptive mode over `plan`, allowing mid-request switching across
    ///   `switchable` (which must be a subset of `plan.candidates`). Every candidate's `max_depth`
    ///   is clamped to `provisioned_max_depth` so depth growth never exceeds allocated cache
    ///   overshoot or rollback ring capacity.
    #[allow(clippy::too_many_arguments)]
    pub fn for_request(
        mode: crate::Speculative,
        plan: AutoSpeculativePlan,
        switchable: Vec<ProposerKind>,
        plain: PlainDecode,
        timed: bool,
        provisioned_max_depth: u32,
        explicit_proposer: ProposerKind,
        explicit_depth: u32,
        explicit_plain_reason: Option<String>,
    ) -> Self {
        if mode != crate::Speculative::Auto {
            let depth = if explicit_proposer == ProposerKind::None {
                0
            } else {
                explicit_depth
            };
            let plain_reason = if explicit_proposer == ProposerKind::None {
                Some(explicit_plain_reason.unwrap_or_else(|| "speculative: off".into()))
            } else {
                None
            };
            let init_event = AdaptiveSpeculativeEvent {
                token_offset: 0,
                verify_step: 0,
                kind: if explicit_proposer == ProposerKind::None {
                    AdaptiveEventKind::StayedPlain
                } else {
                    AdaptiveEventKind::Selected
                },
                from_proposer: explicit_proposer,
                from_depth: depth,
                to_proposer: explicit_proposer,
                to_depth: depth,
                reason: plain_reason.clone().unwrap_or_else(|| {
                    format!("explicit `{}` at depth {depth}", explicit_proposer.label())
                }),
            };
            return Self {
                adaptive: false,
                plain,
                timed,
                candidates: vec![AutoCandidate {
                    proposer: explicit_proposer,
                    recommended_depth: depth,
                    max_depth: depth,
                }],
                switchable: vec![explicit_proposer],
                ruled_out: Vec::new(),
                depth_caps: vec![(explicit_proposer, depth)],
                initial_proposer: explicit_proposer,
                initial_depth: depth,
                active_proposer: explicit_proposer,
                active_depth: depth,
                probe_steps: 0,
                plain_ns: Vec::new(),
                shapes: Vec::new(),
                total_verifies: 0,
                windows: 0,
                window_verifies: 0,
                window_proposed: 0,
                window_accepted: 0,
                window_committed: 0,
                window_timed: Vec::new(),
                clear_verifies: 0,
                clear_accepted: 0,
                clear_timed: Vec::new(),
                measured_window_seen: false,
                incumbent_ns_per_token: None,
                incumbent_tokens_per_verify: 1.0,
                incumbent_accept_ratio: None,
                prev_depth_baseline: None,
                active_probe: None,
                lookup_match_streak: 0,
                last_lookup_peek_len: 0,
                lookup_dry_streak: 0,
                cooldown_remaining: 0,
                depth_changes: 0,
                proposer_switches: 0,
                demoted: explicit_proposer == ProposerKind::None,
                last_monitor: None,
                events: vec![init_event],
                plain_reason,
                cost: AdaptiveCostSummary::default(),
                has_timed_cost: false,
            };
        }

        let plan = if provisioned_max_depth > 0 {
            plan.clamped_to(provisioned_max_depth)
        } else {
            plan
        };
        let primary = plan.primary();
        let depth_caps: Vec<(ProposerKind, u32)> = plan
            .candidates
            .iter()
            .map(|c| (c.proposer, c.max_depth))
            .collect();
        let switchable: Vec<ProposerKind> = switchable
            .into_iter()
            .filter(|k| *k == ProposerKind::None || plan.candidate(*k).is_some())
            .collect();

        if primary.proposer == ProposerKind::None || primary.recommended_depth == 0 {
            let reason = plan
                .fallbacks
                .first()
                .cloned()
                .unwrap_or_else(|| "speculative: auto found no eligible proposer".into());
            let event = AdaptiveSpeculativeEvent {
                token_offset: 0,
                verify_step: 0,
                kind: AdaptiveEventKind::StayedPlain,
                from_proposer: ProposerKind::None,
                from_depth: 0,
                to_proposer: ProposerKind::None,
                to_depth: 0,
                reason: reason.clone(),
            };
            return Self {
                adaptive: true,
                plain,
                timed,
                candidates: plan.candidates,
                switchable: vec![ProposerKind::None],
                ruled_out: Vec::new(),
                depth_caps,
                initial_proposer: ProposerKind::None,
                initial_depth: 0,
                active_proposer: ProposerKind::None,
                active_depth: 0,
                probe_steps: 0,
                plain_ns: Vec::new(),
                shapes: Vec::new(),
                total_verifies: 0,
                windows: 0,
                window_verifies: 0,
                window_proposed: 0,
                window_accepted: 0,
                window_committed: 0,
                window_timed: Vec::new(),
                clear_verifies: 0,
                clear_accepted: 0,
                clear_timed: Vec::new(),
                measured_window_seen: false,
                incumbent_ns_per_token: None,
                incumbent_tokens_per_verify: 1.0,
                incumbent_accept_ratio: None,
                prev_depth_baseline: None,
                active_probe: None,
                lookup_match_streak: 0,
                last_lookup_peek_len: 0,
                lookup_dry_streak: 0,
                cooldown_remaining: 0,
                depth_changes: 0,
                proposer_switches: 0,
                demoted: true,
                last_monitor: None,
                events: vec![event],
                plain_reason: Some(reason),
                cost: AdaptiveCostSummary::default(),
                has_timed_cost: false,
            };
        }

        let pool_labels: Vec<&'static str> =
            plan.candidates.iter().map(|c| c.proposer.label()).collect();
        let select_reason = format!(
            "auto selected `{}` at depth {} (max {}); candidates: [{}]",
            primary.proposer.label(),
            primary.recommended_depth,
            primary.max_depth,
            pool_labels.join(", ")
        );
        let init_event = AdaptiveSpeculativeEvent {
            token_offset: 0,
            verify_step: 0,
            kind: AdaptiveEventKind::Selected,
            from_proposer: primary.proposer,
            from_depth: primary.recommended_depth,
            to_proposer: primary.proposer,
            to_depth: primary.recommended_depth,
            reason: select_reason,
        };

        Self {
            adaptive: true,
            plain,
            timed,
            candidates: plan.candidates,
            switchable,
            ruled_out: Vec::new(),
            depth_caps,
            initial_proposer: primary.proposer,
            initial_depth: primary.recommended_depth,
            active_proposer: primary.proposer,
            active_depth: primary.recommended_depth,
            probe_steps: 0,
            plain_ns: Vec::new(),
            shapes: Vec::new(),
            total_verifies: 0,
            windows: 0,
            window_verifies: 0,
            window_proposed: 0,
            window_accepted: 0,
            window_committed: 0,
            window_timed: Vec::new(),
            clear_verifies: 0,
            clear_accepted: 0,
            clear_timed: Vec::new(),
            measured_window_seen: false,
            incumbent_ns_per_token: None,
            incumbent_tokens_per_verify: 1.0,
            incumbent_accept_ratio: None,
            prev_depth_baseline: None,
            active_probe: None,
            lookup_match_streak: 0,
            last_lookup_peek_len: 0,
            lookup_dry_streak: 0,
            cooldown_remaining: 0,
            depth_changes: 0,
            proposer_switches: 0,
            demoted: false,
            last_monitor: None,
            events: vec![init_event],
            plain_reason: None,
            cost: AdaptiveCostSummary::default(),
            has_timed_cost: false,
        }
    }

    /// Whether this controller is running under adaptive `Auto` mode.
    pub fn is_adaptive(&self) -> bool {
        self.adaptive
    }

    /// Whether the next step is part of the initial plain-step baseline probe.
    pub fn probing_plain(&self) -> bool {
        self.adaptive
            && self.timed
            && !self.demoted
            && self.active_proposer != ProposerKind::None
            && self.plain_ns.len() < PLAIN_PROBE_STEPS as usize
            && self.probe_steps < PLAIN_PROBE_MAX_STEPS
    }

    /// Whether the controller can still switch to `kind` later in this request.
    pub fn can_switch_to(&self, kind: ProposerKind) -> bool {
        self.adaptive
            && !self.demoted
            && kind != ProposerKind::None
            && self.switchable.contains(&kind)
            && !self.ruled_out.contains(&kind)
    }

    /// Mark a proposer unavailable (e.g. its catch-up buffer exceeded its memory bound while
    /// inactive) so the controller will never attempt to switch back to it.
    pub fn retire_proposer(&mut self, kind: ProposerKind) {
        if kind != ProposerKind::None && !self.ruled_out.contains(&kind) {
            self.ruled_out.push(kind);
        }
    }

    /// Plan for the next decode step.
    pub fn next_step(&self) -> AdaptiveStepPlan {
        if self.demoted || self.active_proposer == ProposerKind::None {
            return AdaptiveStepPlan {
                proposer: ProposerKind::None,
                depth: 0,
                probing_plain: false,
                demoted_to_plain: self.demoted,
            };
        }
        if self.probing_plain() {
            return AdaptiveStepPlan {
                proposer: ProposerKind::None,
                depth: 0,
                probing_plain: true,
                demoted_to_plain: false,
            };
        }
        AdaptiveStepPlan {
            proposer: self.active_proposer,
            depth: self.active_depth,
            probing_plain: false,
            demoted_to_plain: false,
        }
    }

    /// Last monitor window decision (populates `DecodeReport::speculative_monitor` for full
    /// backwards compatibility with `AcceptanceMonitor`).
    pub fn last_monitor_decision(&self) -> Option<MonitorDecision> {
        self.last_monitor
    }

    /// Whether the controller has permanently demoted to plain decode.
    pub fn is_demoted(&self) -> bool {
        self.demoted
    }

    /// Build the final [`AdaptiveSpeculativeReport`] for [`DecodeReport`].
    pub fn report(&self) -> AdaptiveSpeculativeReport {
        AdaptiveSpeculativeReport {
            adaptive: self.adaptive,
            candidates: self.candidates.iter().map(|c| c.proposer).collect(),
            initial_proposer: self.initial_proposer,
            initial_depth: self.initial_depth,
            final_proposer: if self.demoted {
                ProposerKind::None
            } else {
                self.active_proposer
            },
            final_depth: if self.demoted { 0 } else { self.active_depth },
            depth_changes: self.depth_changes,
            proposer_switches: self.proposer_switches,
            events: self.events.clone(),
            plain_reason: self.plain_reason.clone(),
            cost: self.has_timed_cost.then_some(self.cost),
        }
    }

    /// Observe one completed engine step and decide whether to continue, resize depth, switch
    /// proposer, or fall back to plain decode.
    pub fn observe_step(&mut self, input: AdaptiveStepInput) -> AdaptiveAction {
        let total_ns = input.cost.effective_total_ns();
        if total_ns > 0 || input.cost.host_syncs > 0 {
            self.cost.record(input.cost, input.committed as u64);
            if total_ns > 0 {
                self.has_timed_cost = true;
            }
        }
        if !self.adaptive || self.demoted {
            return AdaptiveAction::Continue;
        }

        let probing_plain = self.probing_plain();
        let width = 1 + input.drafts;
        let shape_proposer = if probing_plain {
            ProposerKind::None
        } else {
            self.active_proposer
        };
        let seen = self.bump_shape(shape_proposer, width);
        let elapsed_ns =
            (self.timed && total_ns > 0 && seen >= SHAPE_WARMUP_STEPS).then_some(total_ns);
        if input.drafts == 0 {
            self.plain_ns.extend(elapsed_ns);
        }
        if probing_plain {
            self.probe_steps += 1;
            return AdaptiveAction::Continue;
        }

        self.total_verifies += 1;
        self.cooldown_remaining = self.cooldown_remaining.saturating_sub(1);

        // Track CPU prompt-lookup match streaks for repeated-context <-> open-ended switching.
        self.last_lookup_peek_len = input.lookup_peek_len;
        let lookup_min_peek = self.lookup_probe_min_peek();
        if input.lookup_peek_len >= lookup_min_peek {
            self.lookup_match_streak = self.lookup_match_streak.saturating_add(1);
        } else {
            self.lookup_match_streak = 0;
        }
        if self.active_proposer == ProposerKind::PromptLookup && input.accepted == 0 {
            self.lookup_dry_streak = self.lookup_dry_streak.saturating_add(1);
        } else {
            self.lookup_dry_streak = 0;
        }

        let timed_step = elapsed_ns.map(|ns| TimedStep {
            tokens: input.committed.max(1) as u64,
            width: width as u64,
            ns,
        });

        // 1. If a bounded alternate-proposer probe is in flight, drive it to completion.
        if self.active_probe.is_some() {
            return self.step_active_probe(input, timed_step);
        }

        // 2. Accumulate rolling window and clear-loss counters for the incumbent.
        self.window_verifies += 1;
        self.window_proposed += input.drafts as u64;
        self.window_accepted += input.accepted as u64;
        self.window_committed += input.committed.max(1) as u64;
        self.clear_verifies += 1;
        self.clear_accepted += input.accepted as u64;
        if let Some(ts) = timed_step {
            self.window_timed.push(ts);
            if !self.measured_window_seen {
                if self.clear_timed.len() == CLEAR_LOSS_MAX_TIMED_STEPS {
                    self.clear_timed.remove(0);
                }
                self.clear_timed.push(ts);
            }
        }

        // 3. Check clear-loss fast demotion / fallback before waiting for full window.
        if let Some(clear_decision) = self.clear_loss() {
            return self.handle_losing_decision(clear_decision, input.generated_tokens, true);
        }

        // 4. Mid-request bounded probe trigger (only when cooldown expired and switch budget allows).
        if self.cooldown_remaining == 0 && self.proposer_switches < ADAPTIVE_MAX_SWITCHES {
            if let Some(action) = self.maybe_start_probe(input.generated_tokens) {
                return action;
            }
        }

        // 5. Rolling window evaluation every ACCEPTANCE_PROBE_VERIFIES steps.
        if self.window_verifies < ACCEPTANCE_PROBE_VERIFIES {
            return AdaptiveAction::Continue;
        }

        let decision = self.evaluate_window();
        if decision.basis == DemotionBasis::Measured {
            self.measured_window_seen = true;
            self.clear_timed.clear();
        }
        let window_proposed = self.window_proposed;
        let window_accepted = self.window_accepted;
        let window_committed = self.window_committed;
        let window_verifies = self.window_verifies;
        let window_ns_per_tok = (decision.timed_steps >= MIN_TIMED_WINDOW_STEPS
            && decision.timed_tokens > 0
            && decision.timed_ns > 0)
            .then(|| decision.timed_ns as f64 / decision.timed_tokens as f64);

        if let Some(ns_per_tok) = window_ns_per_tok {
            self.incumbent_ns_per_token = Some(ns_per_tok);
        }
        self.incumbent_tokens_per_verify =
            window_committed as f64 / f64::from(window_verifies.max(1));
        if window_proposed > 0 {
            self.incumbent_accept_ratio = Some(window_accepted as f64 / window_proposed as f64);
        }

        self.reset_window();

        if decision.demoted {
            return self.handle_losing_window(
                decision,
                window_accepted,
                window_verifies,
                input.generated_tokens,
            );
        }

        self.last_monitor = Some(decision);

        // Passing window: check whether depth should adapt (grow, shrink, or revert regressed growth).
        if self.cooldown_remaining == 0 && self.depth_changes < ADAPTIVE_MAX_DEPTH_CHANGES {
            if let Some(action) = self.maybe_adapt_depth(
                decision,
                window_proposed,
                window_accepted,
                window_verifies,
                window_ns_per_tok,
                input.generated_tokens,
            ) {
                return action;
            }
        }

        AdaptiveAction::Continue
    }

    fn bump_shape(&mut self, proposer: ProposerKind, width: usize) -> u32 {
        match self
            .shapes
            .iter_mut()
            .find(|(p, w, _)| *p == proposer && *w == width)
        {
            Some((_, _, n)) => {
                *n += 1;
                *n - 1
            }
            None => {
                self.shapes.push((proposer, width, 1));
                0
            }
        }
    }

    fn plain_step_ns(&self) -> Option<u64> {
        (self.plain_ns.len() >= PLAIN_PROBE_STEPS as usize).then(|| {
            let mut sorted = self.plain_ns.clone();
            sorted.sort_unstable();
            sorted[sorted.len() / 2]
        })
    }

    fn max_depth_for(&self, proposer: ProposerKind) -> u32 {
        self.depth_caps
            .iter()
            .find(|(p, _)| *p == proposer)
            .map_or(0, |(_, cap)| *cap)
    }

    fn cap_depth_for(&mut self, proposer: ProposerKind, cap: u32) {
        if let Some((_, existing)) = self.depth_caps.iter_mut().find(|(p, _)| *p == proposer) {
            *existing = (*existing).min(cap.max(1));
        }
    }

    fn candidate_recommended_depth(&self, proposer: ProposerKind) -> u32 {
        let cap = self.max_depth_for(proposer).max(1);
        self.candidates
            .iter()
            .find(|c| c.proposer == proposer)
            .map_or(1, |c| c.recommended_depth)
            .clamp(1, cap)
    }

    fn reset_window(&mut self) {
        self.window_verifies = 0;
        self.window_proposed = 0;
        self.window_accepted = 0;
        self.window_committed = 0;
        self.window_timed.clear();
    }

    fn reset_clear_loss(&mut self) {
        self.clear_verifies = 0;
        self.clear_accepted = 0;
        self.clear_timed.clear();
        self.measured_window_seen = false;
    }

    fn clear_loss(&mut self) -> Option<MonitorDecision> {
        if self.measured_window_seen || self.clear_timed.len() < CLEAR_LOSS_MIN_TIMED_STEPS as usize
        {
            return None;
        }
        let plain_step_ns = self.plain_step_ns()?;
        let stretch = Stretch::of(&self.clear_timed);
        if stretch.steps < CLEAR_LOSS_MIN_TIMED_STEPS {
            return None;
        }
        let decision = MonitorDecision {
            window: self.windows + 1,
            verifies: self.clear_verifies,
            accepted: self.clear_accepted,
            timed_steps: stretch.steps,
            timed_tokens: stretch.tokens,
            timed_ns: stretch.ns,
            plain_step_ns: Some(plain_step_ns),
            basis: DemotionBasis::Measured,
            demoted: true,
        };
        let gain = decision.gain()?;
        let optimistic = gain * stretch.optimistic_tokens_factor();
        if optimistic.is_nan() || optimistic >= CLEAR_LOSS_GAIN {
            return None;
        }
        self.windows += 1;
        Some(decision)
    }

    fn evaluate_window(&mut self) -> MonitorDecision {
        self.windows += 1;
        let plain_step_ns = self.plain_step_ns();
        let stretch = Stretch::of(&self.window_timed);
        let mut decision = MonitorDecision {
            window: self.windows,
            verifies: self.window_verifies,
            accepted: self.window_accepted,
            timed_steps: stretch.steps,
            timed_tokens: stretch.tokens,
            timed_ns: stretch.ns,
            plain_step_ns,
            basis: DemotionBasis::Static,
            demoted: false,
        };
        let measured_loss = decision
            .gain()
            .filter(|_| stretch.steps >= MIN_TIMED_WINDOW_STEPS)
            .map(|gain| gain < 1.0 - MEASURED_GAIN_MARGIN);
        let threshold = demotion_threshold(self.active_proposer, self.active_depth, self.plain);
        let static_loss = threshold
            .is_some_and(|t| (self.window_accepted as f64) < t * f64::from(self.window_verifies));
        (decision.demoted, decision.basis) = match (self.plain, measured_loss) {
            (PlainDecode::MlxPipelined, Some(true)) => (true, DemotionBasis::Measured),
            (PlainDecode::MlxPipelined, measured) if static_loss || measured.is_none() => {
                (static_loss, DemotionBasis::Static)
            }
            (_, Some(loss)) => (loss, DemotionBasis::Measured),
            (_, None) => (static_loss, DemotionBasis::Static),
        };
        decision
    }

    fn step_active_probe(
        &mut self,
        input: AdaptiveStepInput,
        timed_step: Option<TimedStep>,
    ) -> AdaptiveAction {
        let mut probe = self.active_probe.take().expect("checked is_some");
        probe.steps += 1;
        probe.accepted += input.accepted as u64;
        probe.committed += input.committed.max(1) as u64;
        if let Some(ts) = timed_step {
            probe.timed.push(ts);
        }

        // Fast abort if alternate proposer accepts nothing over its first 3 probe steps.
        let early_fail = probe.steps >= 3 && probe.accepted == 0;
        if !early_fail && probe.steps < ADAPTIVE_PROBE_STEPS {
            self.active_probe = Some(probe);
            return AdaptiveAction::Continue;
        }

        let stretch = Stretch::of(&probe.timed);
        let probe_ns_per_tok = (stretch.steps >= ADAPTIVE_PROBE_MIN_TIMED_STEPS
            && stretch.tokens > 0
            && stretch.ns > 0)
            .then(|| stretch.ns as f64 / stretch.tokens as f64);
        let probe_tok_per_verify = probe.committed as f64 / f64::from(probe.steps.max(1));

        let plain_ok = match (probe_ns_per_tok, self.plain_step_ns()) {
            (Some(p_ns), Some(plain)) if plain > 0 => {
                p_ns <= (plain as f64) * (1.0 + MEASURED_GAIN_MARGIN)
            }
            _ => true,
        };

        let (wins, reason) = if early_fail {
            (
                false,
                format!(
                    "probe `{}` aborted after {} steps with 0 accepted drafts; reverted to `{}`",
                    probe.probe_proposer.label(),
                    probe.steps,
                    probe.incumbent_proposer.label()
                ),
            )
        } else if let (Some(p_ns), Some(inc_ns)) = (probe_ns_per_tok, probe.incumbent_ns_per_token)
        {
            let target_ns = inc_ns * (1.0 - ADAPTIVE_SWITCH_IMPROVEMENT_MARGIN);
            let beats = p_ns < target_ns && plain_ok;
            (
                beats,
                format!(
                    "probe `{}` {:.2} ms/tok vs incumbent `{}` {:.2} ms/tok ({})",
                    probe.probe_proposer.label(),
                    p_ns / 1e6,
                    probe.incumbent_proposer.label(),
                    inc_ns / 1e6,
                    if beats { "switched" } else { "reverted" }
                ),
            )
        } else {
            let beats = probe.accepted > 0
                && probe_tok_per_verify
                    > probe.incumbent_tokens_per_verify * ADAPTIVE_UNTIMED_SWITCH_GAIN;
            (
                beats,
                format!(
                    "probe `{}` {:.2} tok/verify vs incumbent `{}` {:.2} tok/verify ({})",
                    probe.probe_proposer.label(),
                    probe_tok_per_verify,
                    probe.incumbent_proposer.label(),
                    probe.incumbent_tokens_per_verify,
                    if beats { "switched" } else { "reverted" }
                ),
            )
        };

        self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
        self.incumbent_accept_ratio = None;
        self.reset_window();
        self.reset_clear_loss();

        if wins {
            self.active_proposer = probe.probe_proposer;
            self.active_depth = probe.probe_depth;
            self.proposer_switches += 1;
            if let Some(p_ns) = probe_ns_per_tok {
                self.incumbent_ns_per_token = Some(p_ns);
            }
            self.incumbent_tokens_per_verify = probe_tok_per_verify;
            self.prev_depth_baseline = None;
            self.events.push(AdaptiveSpeculativeEvent {
                token_offset: input.generated_tokens,
                verify_step: self.total_verifies,
                kind: AdaptiveEventKind::ProposerSwitched,
                from_proposer: probe.incumbent_proposer,
                from_depth: probe.incumbent_depth,
                to_proposer: probe.probe_proposer,
                to_depth: probe.probe_depth,
                reason,
            });
            AdaptiveAction::Continue
        } else {
            if !self.ruled_out.contains(&probe.probe_proposer) {
                self.ruled_out.push(probe.probe_proposer);
            }
            if self.ruled_out.contains(&probe.incumbent_proposer) {
                self.demoted = true;
                self.active_proposer = ProposerKind::None;
                self.active_depth = 0;
                self.proposer_switches += 1;
                let demote_reason = format!(
                    "{reason}; incumbent `{}` retired while probe was active; falling back to plain decode",
                    probe.incumbent_proposer.label()
                );
                self.plain_reason = Some(demote_reason.clone());
                self.events.push(AdaptiveSpeculativeEvent {
                    token_offset: input.generated_tokens,
                    verify_step: self.total_verifies,
                    kind: AdaptiveEventKind::FallbackToPlain,
                    from_proposer: probe.probe_proposer,
                    from_depth: probe.probe_depth,
                    to_proposer: ProposerKind::None,
                    to_depth: 0,
                    reason: demote_reason,
                });
                return AdaptiveAction::DemoteToPlain;
            }
            self.active_proposer = probe.incumbent_proposer;
            self.active_depth = probe.incumbent_depth;
            self.events.push(AdaptiveSpeculativeEvent {
                token_offset: input.generated_tokens,
                verify_step: self.total_verifies,
                kind: AdaptiveEventKind::ProbeReverted,
                from_proposer: probe.probe_proposer,
                from_depth: probe.probe_depth,
                to_proposer: probe.incumbent_proposer,
                to_depth: probe.incumbent_depth,
                reason,
            });
            AdaptiveAction::SwitchProposer {
                from: probe.probe_proposer,
                to: probe.incumbent_proposer,
                depth: probe.incumbent_depth,
            }
        }
    }

    /// Minimum `prompt_lookup` peek match length required to count toward a bounded probe streak.
    ///
    /// When the incumbent is a GPU proposer (`mtp`, `block_draft`, `draft_model`), require the
    /// peeked match to meaningfully challenge the incumbent's current depth (at least 3 tokens,
    /// and at least `active_depth`, capped at `prompt_lookup`'s own probe depth so deep lagging
    /// GPU proposers can still be challenged within the engine's peek horizon).
    fn lookup_probe_min_peek(&self) -> usize {
        if is_gpu_proposer(self.active_proposer) {
            let probe_depth = self.candidate_recommended_depth(ProposerKind::PromptLookup) as usize;
            (self.active_depth as usize).max(3).min(probe_depth.max(2))
        } else {
            2
        }
    }

    fn max_incumbent_accept_ratio_for_lookup(&self) -> f64 {
        match self.plain {
            PlainDecode::Candle => ADAPTIVE_LOOKUP_MAX_INCUMBENT_ACCEPT_RATIO_CANDLE,
            PlainDecode::MlxPipelined | PlainDecode::MlxUnpipelined => {
                ADAPTIVE_LOOKUP_MAX_INCUMBENT_ACCEPT_RATIO
            }
        }
    }

    /// Current incumbent acceptance ratio (`accepted / proposed`) once at least
    /// [`ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES`] verify steps have been observed for the
    /// incumbent, or `None` if insufficient samples exist yet.
    fn current_incumbent_accept_ratio(&self) -> Option<f64> {
        if self.window_verifies >= ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES
            && self.window_proposed > 0
        {
            Some(self.window_accepted as f64 / self.window_proposed as f64)
        } else if self.window_verifies > 0 && self.window_proposed > 0 {
            let prev = self.incumbent_accept_ratio?;
            let partial = self.window_accepted as f64 / self.window_proposed as f64;
            Some(prev.max(partial))
        } else {
            self.incumbent_accept_ratio
        }
    }

    fn maybe_start_probe(&mut self, generated_tokens: u64) -> Option<AdaptiveAction> {
        let have_baseline =
            self.incumbent_ns_per_token.is_some() || (!self.timed && self.total_verifies >= 4);
        let (probe_proposer, reason, is_dry_recovery) = if is_gpu_proposer(self.active_proposer)
            && self.lookup_match_streak >= ADAPTIVE_LOOKUP_TRIGGER_MATCHES
            && self.can_switch_to(ProposerKind::PromptLookup)
            && (have_baseline || self.window_verifies >= ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES)
        {
            let max_inc_ratio = self.max_incumbent_accept_ratio_for_lookup();
            let inc_ratio = self.current_incumbent_accept_ratio()?;
            if inc_ratio >= max_inc_ratio {
                return None;
            }
            let min_peek = self.lookup_probe_min_peek();
            (
                ProposerKind::PromptLookup,
                format!(
                    "prompt_lookup matched >={min_peek} tokens for {} consecutive steps while `{}` \
                     acceptance {:.0}% < {:.0}%; starting bounded probe",
                    self.lookup_match_streak,
                    self.active_proposer.label(),
                    inc_ratio * 100.0,
                    max_inc_ratio * 100.0,
                ),
                false,
            )
        } else if self.active_proposer == ProposerKind::PromptLookup
            && self.lookup_dry_streak >= ADAPTIVE_LOOKUP_DRY_STEPS
        {
            let gpu = self.switchable.iter().copied().find(|&k| {
                k != ProposerKind::None
                    && k != ProposerKind::PromptLookup
                    && !self.ruled_out.contains(&k)
            })?;
            (
                gpu,
                format!(
                    "prompt_lookup accepted 0 drafts for {} consecutive steps; probing `{}`",
                    self.lookup_dry_streak,
                    gpu.label()
                ),
                true,
            )
        } else {
            return None;
        };

        let probe_depth = self.candidate_recommended_depth(probe_proposer);
        let inc_proposer = self.active_proposer;
        let inc_depth = self.active_depth;

        // Snapshot current incumbent cost/token. When recovering from a dry `prompt_lookup`
        // streak, compare against the dry streak's own recent cost (or plain baseline) rather
        // than a stale repeated-context rate from earlier in the request.
        let inc_ns_per_tok = if is_dry_recovery {
            let dry_tail: Vec<TimedStep> = self
                .window_timed
                .iter()
                .rev()
                .take(self.lookup_dry_streak as usize)
                .take_while(|ts| ts.tokens <= 1)
                .copied()
                .collect();
            let stretch = Stretch::of(&dry_tail);
            (stretch.steps >= 2 && stretch.tokens > 0 && stretch.ns > 0)
                .then(|| stretch.ns as f64 / stretch.tokens as f64)
                .or_else(|| self.plain_step_ns().map(|p| p as f64))
        } else {
            let recent = {
                let stretch = Stretch::of(&self.window_timed);
                (stretch.steps >= 2 && stretch.tokens > 0 && stretch.ns > 0)
                    .then(|| stretch.ns as f64 / stretch.tokens as f64)
            };
            recent.or(self.incumbent_ns_per_token)
        };
        let inc_tok_per_verify = if is_dry_recovery {
            // A dry lookup streak accepted 0 drafts -> 1 committed token per verify step.
            1.0
        } else if self.window_verifies > 0 {
            self.window_committed as f64 / f64::from(self.window_verifies)
        } else {
            self.incumbent_tokens_per_verify
        };

        self.active_probe = Some(ProbeState {
            incumbent_proposer: inc_proposer,
            incumbent_depth: inc_depth,
            incumbent_ns_per_token: inc_ns_per_tok,
            incumbent_tokens_per_verify: inc_tok_per_verify,
            probe_proposer,
            probe_depth,
            steps: 0,
            accepted: 0,
            committed: 0,
            timed: Vec::new(),
        });
        self.active_proposer = probe_proposer;
        self.active_depth = probe_depth;
        self.lookup_match_streak = 0;
        self.last_lookup_peek_len = 0;
        self.lookup_dry_streak = 0;
        self.events.push(AdaptiveSpeculativeEvent {
            token_offset: generated_tokens,
            verify_step: self.total_verifies,
            kind: AdaptiveEventKind::ProbeStarted,
            from_proposer: inc_proposer,
            from_depth: inc_depth,
            to_proposer: probe_proposer,
            to_depth: probe_depth,
            reason,
        });
        Some(AdaptiveAction::SwitchProposer {
            from: inc_proposer,
            to: probe_proposer,
            depth: probe_depth,
        })
    }

    fn handle_losing_window(
        &mut self,
        decision: MonitorDecision,
        window_accepted: u64,
        window_verifies: u32,
        generated_tokens: u64,
    ) -> AdaptiveAction {
        let severe_loss = decision.gain().is_some_and(|g| g < CLEAR_LOSS_GAIN);

        // When drafts are partially accepted (window_accepted > 0), loss is marginal (not severe),
        // and we have room to shrink depth, shrink depth first before giving up on the proposer.
        if window_accepted > 0
            && !severe_loss
            && self.active_depth > 1
            && self.depth_changes < ADAPTIVE_MAX_DEPTH_CHANGES
            && self.cooldown_remaining == 0
        {
            let from = self.active_depth;
            let mal = window_accepted as f64 / f64::from(window_verifies.max(1));
            let to = ((from / 2).max(1)).min((mal.ceil() as u32).max(1));
            let to = to.clamp(1, from - 1);
            self.active_depth = to;
            self.cap_depth_for(self.active_proposer, to);
            self.depth_changes += 1;
            self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
            self.prev_depth_baseline = None;
            self.incumbent_accept_ratio = None;
            self.reset_clear_loss();
            // Record the window as non-demoting so `speculative_demoted_at` and
            // `speculative_monitor.demoted` remain consistent.
            self.last_monitor = Some(MonitorDecision {
                demoted: false,
                ..decision
            });
            let reason = match decision.gain() {
                Some(g) => format!(
                    "window {} measured gain {:.2}x < {:.2}x with mal {:.2}; shrunk depth {} -> {}",
                    decision.window,
                    g,
                    1.0 - MEASURED_GAIN_MARGIN,
                    mal,
                    from,
                    to
                ),
                None => format!(
                    "window {} mal {:.2} below break-even; shrunk depth {} -> {}",
                    decision.window, mal, from, to
                ),
            };
            self.events.push(AdaptiveSpeculativeEvent {
                token_offset: generated_tokens,
                verify_step: self.total_verifies,
                kind: AdaptiveEventKind::DepthDecreased,
                from_proposer: self.active_proposer,
                from_depth: from,
                to_proposer: self.active_proposer,
                to_depth: to,
                reason,
            });
            return AdaptiveAction::ResizeDepth { from, to };
        }

        self.handle_losing_decision(decision, generated_tokens, false)
    }

    fn handle_losing_decision(
        &mut self,
        decision: MonitorDecision,
        generated_tokens: u64,
        clear_loss: bool,
    ) -> AdaptiveAction {
        let losing_proposer = self.active_proposer;
        let losing_depth = self.active_depth;
        if !self.ruled_out.contains(&losing_proposer) {
            self.ruled_out.push(losing_proposer);
        }

        // Before demoting permanently to plain, see if an eligible non-ruled-out alternate proposer
        // makes sense (for `PromptLookup`, only when a repeated-context match is actually present).
        if self.proposer_switches < ADAPTIVE_MAX_SWITCHES {
            let alternate = self.switchable.iter().copied().find(|&k| {
                if k == ProposerKind::None || k == losing_proposer || self.ruled_out.contains(&k) {
                    return false;
                }
                if k == ProposerKind::PromptLookup {
                    self.lookup_match_streak >= 1 || self.last_lookup_peek_len >= 2
                } else {
                    true
                }
            });
            if let Some(next_proposer) = alternate {
                let next_depth = self.candidate_recommended_depth(next_proposer);
                self.active_proposer = next_proposer;
                self.active_depth = next_depth;
                self.proposer_switches += 1;
                self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
                self.prev_depth_baseline = None;
                self.incumbent_ns_per_token = None;
                self.incumbent_accept_ratio = None;
                self.lookup_match_streak = 0;
                self.last_lookup_peek_len = 0;
                self.reset_window();
                self.reset_clear_loss();
                self.last_monitor = Some(MonitorDecision {
                    demoted: false,
                    ..decision
                });
                let reason = format!(
                    "`{}` ruled out ({}); switched to `{}` at depth {}",
                    losing_proposer.label(),
                    describe_loss(&decision, clear_loss),
                    next_proposer.label(),
                    next_depth
                );
                self.events.push(AdaptiveSpeculativeEvent {
                    token_offset: generated_tokens,
                    verify_step: self.total_verifies,
                    kind: AdaptiveEventKind::ProposerSwitched,
                    from_proposer: losing_proposer,
                    from_depth: losing_depth,
                    to_proposer: next_proposer,
                    to_depth: next_depth,
                    reason,
                });
                return AdaptiveAction::SwitchProposer {
                    from: losing_proposer,
                    to: next_proposer,
                    depth: next_depth,
                };
            }
        }

        // Permanent fallback to plain decode.
        self.last_monitor = Some(decision);
        self.demoted = true;
        self.active_proposer = ProposerKind::None;
        self.active_depth = 0;
        self.proposer_switches += 1;
        let reason = format!(
            "demoted `{}` (depth {}) to plain decode: {}",
            losing_proposer.label(),
            losing_depth,
            describe_loss(&decision, clear_loss)
        );
        self.plain_reason = Some(reason.clone());
        self.events.push(AdaptiveSpeculativeEvent {
            token_offset: generated_tokens,
            verify_step: self.total_verifies,
            kind: AdaptiveEventKind::FallbackToPlain,
            from_proposer: losing_proposer,
            from_depth: losing_depth,
            to_proposer: ProposerKind::None,
            to_depth: 0,
            reason,
        });
        AdaptiveAction::DemoteToPlain
    }

    fn maybe_adapt_depth(
        &mut self,
        decision: MonitorDecision,
        window_proposed: u64,
        window_accepted: u64,
        window_verifies: u32,
        window_ns_per_tok: Option<f64>,
        generated_tokens: u64,
    ) -> Option<AdaptiveAction> {
        let proposer = self.active_proposer;
        let depth = self.active_depth;
        let max_depth = self.max_depth_for(proposer);

        // 1. If we previously grew depth and this measured window has worse ms/token than the
        //    previous depth, revert and cap max_depth so we never oscillate back up.
        if let (Some((prev_p, prev_d, prev_ns)), Some(curr_ns)) =
            (self.prev_depth_baseline, window_ns_per_tok)
        {
            if prev_p == proposer && prev_d < depth {
                if curr_ns > prev_ns * (1.0 + MEASURED_GAIN_MARGIN) {
                    self.active_depth = prev_d;
                    self.cap_depth_for(proposer, prev_d);
                    self.depth_changes += 1;
                    self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
                    self.prev_depth_baseline = None;
                    self.incumbent_accept_ratio = None;
                    let reason = format!(
                        "depth {depth} cost {:.2} ms/tok > prior depth {prev_d} cost {:.2} ms/tok; \
                         reverted and capped max_depth={prev_d}",
                        curr_ns / 1e6,
                        prev_ns / 1e6
                    );
                    self.events.push(AdaptiveSpeculativeEvent {
                        token_offset: generated_tokens,
                        verify_step: self.total_verifies,
                        kind: AdaptiveEventKind::DepthDecreased,
                        from_proposer: proposer,
                        from_depth: depth,
                        to_proposer: proposer,
                        to_depth: prev_d,
                        reason,
                    });
                    return Some(AdaptiveAction::ResizeDepth {
                        from: depth,
                        to: prev_d,
                    });
                }
                // Growth held or improved cost/token: clear baseline so further growth is allowed.
                self.prev_depth_baseline = None;
            }
        }

        if window_proposed == 0 {
            return None;
        }
        let acceptance_ratio = window_accepted as f64 / window_proposed as f64;
        let depth_capacity = (u64::from(window_verifies.max(1)) * u64::from(depth.max(1))) as f64;
        let depth_fill_ratio = window_proposed as f64 / depth_capacity;
        let depth_saturation = window_accepted as f64 / depth_capacity;
        let mal = window_accepted as f64 / f64::from(window_verifies.max(1));

        // 2. Grow depth only when the current depth window is genuinely saturated (both fill ratio
        //    and accepted/capacity meet threshold — equivalent to `acceptance_ratio >= 0.75` for
        //    fixed-width proposers, while preventing variable-width `prompt_lookup` from growing
        //    depth when matches only partially fill the current depth) and (if timed) gain exceeds
        //    break-even + margin.
        let gain_allows_grow = match decision.gain() {
            Some(g) => g >= 1.0 + MEASURED_GAIN_MARGIN,
            None => true,
        };
        if depth < max_depth
            && acceptance_ratio >= ADAPTIVE_GROW_ACCEPTANCE_RATIO
            && depth_fill_ratio >= ADAPTIVE_GROW_FILL_RATIO
            && depth_saturation >= ADAPTIVE_GROW_ACCEPTANCE_RATIO
            && gain_allows_grow
        {
            let step =
                if self.plain == PlainDecode::Candle && proposer == ProposerKind::PromptLookup {
                    // Variable-width `prompt_lookup` on Candle pays CUDA graph warm-up/capture for
                    // each new verify shape; grow conservatively by +1 per saturated window.
                    1
                } else {
                    (depth / 2).clamp(1, 2)
                };
            let next_depth = (depth + step).min(max_depth);
            if next_depth > depth {
                if let Some(curr_ns) = window_ns_per_tok {
                    self.prev_depth_baseline = Some((proposer, depth, curr_ns));
                }
                self.active_depth = next_depth;
                self.depth_changes += 1;
                self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
                self.incumbent_accept_ratio = None;
                let reason = format!(
                    "window {} acceptance {:.0}% (fill {:.0}%, saturation {:.0}%, mal {:.2}/{}) >= {:.0}%; grew depth {} -> {} (cap {})",
                    decision.window,
                    acceptance_ratio * 100.0,
                    depth_fill_ratio * 100.0,
                    depth_saturation * 100.0,
                    mal,
                    depth,
                    ADAPTIVE_GROW_ACCEPTANCE_RATIO * 100.0,
                    depth,
                    next_depth,
                    max_depth
                );
                self.events.push(AdaptiveSpeculativeEvent {
                    token_offset: generated_tokens,
                    verify_step: self.total_verifies,
                    kind: AdaptiveEventKind::DepthIncreased,
                    from_proposer: proposer,
                    from_depth: depth,
                    to_proposer: proposer,
                    to_depth: next_depth,
                    reason,
                });
                return Some(AdaptiveAction::ResizeDepth {
                    from: depth,
                    to: next_depth,
                });
            }
        }

        // 3. Shrink depth when wasting most drafts (`acceptance_ratio < 0.35`, `depth > 1`).
        if depth > 1 && window_accepted > 0 && acceptance_ratio < ADAPTIVE_SHRINK_ACCEPTANCE_RATIO {
            let next_depth = (mal.ceil() as u32).clamp(1, depth - 1);
            if next_depth < depth {
                self.active_depth = next_depth;
                self.depth_changes += 1;
                self.cooldown_remaining = ADAPTIVE_COOLDOWN_VERIFIES;
                self.prev_depth_baseline = None;
                self.incumbent_accept_ratio = None;
                let reason = format!(
                    "window {} acceptance {:.0}% (mal {:.2}/{}) < {:.0}%; shrunk depth {} -> {}",
                    decision.window,
                    acceptance_ratio * 100.0,
                    mal,
                    depth,
                    ADAPTIVE_SHRINK_ACCEPTANCE_RATIO * 100.0,
                    depth,
                    next_depth
                );
                self.events.push(AdaptiveSpeculativeEvent {
                    token_offset: generated_tokens,
                    verify_step: self.total_verifies,
                    kind: AdaptiveEventKind::DepthDecreased,
                    from_proposer: proposer,
                    from_depth: depth,
                    to_proposer: proposer,
                    to_depth: next_depth,
                    reason,
                });
                return Some(AdaptiveAction::ResizeDepth {
                    from: depth,
                    to: next_depth,
                });
            }
        }

        None
    }
}

fn is_gpu_proposer(kind: ProposerKind) -> bool {
    matches!(
        kind,
        ProposerKind::Mtp | ProposerKind::BlockDraft | ProposerKind::DraftModel
    )
}

fn describe_loss(decision: &MonitorDecision, clear_loss: bool) -> String {
    let tag = if clear_loss {
        "clear loss"
    } else {
        "window loss"
    };
    match decision.gain() {
        Some(gain) => format!(
            "{tag} on window {} (basis={}, gain={gain:.2}x, mal={:.2})",
            decision.window,
            decision.basis.label(),
            decision.mean_accepted_length()
        ),
        None => format!(
            "{tag} on window {} (basis={}, mal={:.2})",
            decision.window,
            decision.basis.label(),
            decision.mean_accepted_length()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        MtpCapabilities, ProposerCapabilities, Speculative, SpeculativeProposer,
        TextLlmCapabilities,
    };

    fn caps_all() -> TextLlmCapabilities {
        TextLlmCapabilities {
            mtp: Some(MtpCapabilities {
                max_draft_tokens: 5,
                recommended_draft_tokens: 3,
            }),
            speculative: vec![
                ProposerCapabilities {
                    proposer: SpeculativeProposer::BlockDraft,
                    max_depth: 8,
                    recommended_depth: 4,
                },
                ProposerCapabilities {
                    proposer: SpeculativeProposer::DraftModel,
                    max_depth: 6,
                    recommended_depth: 4,
                },
                ProposerCapabilities {
                    proposer: SpeculativeProposer::PromptLookup,
                    max_depth: 7,
                    recommended_depth: 4,
                },
            ],
            ..Default::default()
        }
    }

    fn step_cost(prop: u64, ver: u64, rb: u64, sync: u64, host_syncs: u64) -> StepCost {
        StepCost {
            proposal_ns: prop,
            verify_ns: ver,
            rollback_ns: rb,
            sync_ns: sync,
            total_ns: prop + ver + rb + sync,
            host_syncs,
        }
    }

    fn run_plain_probe(ctrl: &mut AdaptiveSpeculativeController, plain_ns: u64, tokens: &mut u64) {
        while ctrl.probing_plain() {
            let plan = ctrl.next_step();
            assert!(plan.probing_plain);
            assert_eq!(plan.proposer, ProposerKind::None);
            assert_eq!(plan.depth, 0);
            *tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 0,
                committed: 1,
                generated_tokens: *tokens,
                cost: step_cost(0, plain_ns * 9 / 10, 0, plain_ns / 10, 1),
                lookup_peek_len: 0,
            });
            assert_eq!(action, AdaptiveAction::Continue);
        }
    }

    #[test]
    fn resolve_auto_candidates_evaluates_all_five_paths_and_respects_context_and_preference() {
        let caps = caps_all();

        // Preferred BlockDraft: BlockDraft > Mtp > DraftModel > PromptLookup > None
        let preferred = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: true,
                draft_context: 1024,
                prompt_tokens: 100,
                max_new_tokens: 128,
                ..Default::default()
            },
        );
        assert_eq!(
            preferred.kinds(),
            vec![
                ProposerKind::BlockDraft,
                ProposerKind::Mtp,
                ProposerKind::DraftModel,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ]
        );
        assert!(preferred.fallbacks.is_empty());

        // Non-preferred BlockDraft: Mtp > BlockDraft > DraftModel > PromptLookup > None
        let unpreferred = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: false,
                draft_context: 1024,
                prompt_tokens: 100,
                max_new_tokens: 128,
                ..Default::default()
            },
        );
        assert_eq!(
            unpreferred.kinds(),
            vec![
                ProposerKind::Mtp,
                ProposerKind::BlockDraft,
                ProposerKind::DraftModel,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ]
        );

        // Out-of-context DraftModel is excluded with diagnostic fallback.
        let tight = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: false,
                draft_context: 120,
                prompt_tokens: 100,
                max_new_tokens: 32,
                ..Default::default()
            },
        );
        assert!(!tight.kinds().contains(&ProposerKind::DraftModel));
        assert!(
            tight
                .fallbacks
                .iter()
                .any(|f| f.contains("draft_model") && f.contains("context window 120")),
            "{:?}",
            tight.fallbacks
        );

        // Tight-but-fitting DraftModel clamps its max_depth to remaining room.
        let room_3 = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: false,
                draft_context: 134, // 100 + 30 + 1 + 3 = 134
                prompt_tokens: 100,
                max_new_tokens: 30,
                ..Default::default()
            },
        );
        let dm = room_3.candidate(ProposerKind::DraftModel).unwrap();
        assert_eq!(dm.max_depth, 3);
        assert_eq!(dm.recommended_depth, 3);

        // Withheld BlockDraft and DraftModel leave Mtp, PromptLookup, None + fallbacks.
        let withheld = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: true,
                block_draft_withheld: true,
                draft_model_withheld: true,
                ..Default::default()
            },
        );
        assert_eq!(
            withheld.kinds(),
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ]
        );
        assert_eq!(withheld.fallbacks.len(), 2);

        // No advertised proposers stays plain with reason.
        let empty = resolve_auto_candidates(&TextLlmCapabilities::default(), Default::default());
        assert_eq!(empty.kinds(), vec![ProposerKind::None]);
        assert_eq!(empty.fallbacks.len(), 1);
    }

    #[test]
    fn depth_grows_on_high_acceptance_and_reverts_with_cap_if_cost_regresses() {
        let plan = AutoSpeculativePlan::single(ProposerKind::Mtp, 3, 6);
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![ProposerKind::Mtp, ProposerKind::None],
            PlainDecode::Candle,
            true,
            6,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut tokens = 1u64;
        // Plain baseline: 10ms per token.
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // Window 1 at depth 3: 3/3 accepted every step, total cost 14ms -> 3.5ms/tok (gain 2.86x).
        let mut grew = false;
        for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
            let p = ctrl.next_step();
            assert_eq!((p.proposer, p.depth), (ProposerKind::Mtp, 3));
            tokens += 4;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 3,
                drafts: 3,
                committed: 4,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 8_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 0,
            });
            if let AdaptiveAction::ResizeDepth { from, to } = action {
                assert_eq!((from, to), (3, 4));
                grew = true;
            }
        }
        assert!(grew, "expected depth growth 3 -> 4");

        // Window 2 at depth 4: expensive (28ms per step, 4 tokens -> 7ms/tok, still beats plain
        // 10ms/tok so passing, but worse than depth 3's 3.5ms/tok!). Should revert to 3 and cap.
        let mut reverted = false;
        for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
            let p = ctrl.next_step();
            assert_eq!(p.depth, 4);
            tokens += 4;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 3,
                drafts: 4,
                committed: 4,
                generated_tokens: tokens,
                cost: step_cost(10_000_000, 15_000_000, 1_000_000, 2_000_000, 1),
                lookup_peek_len: 0,
            });
            if let AdaptiveAction::ResizeDepth { from, to } = action {
                assert_eq!((from, to), (4, 3));
                reverted = true;
            }
        }
        assert!(reverted, "expected revert 4 -> 3 after higher ms/tok");

        // Window 3 at depth 3: even with 100% acceptance, depth must NOT oscillate back to 4!
        for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
            let p = ctrl.next_step();
            assert_eq!(p.depth, 3);
            tokens += 4;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 3,
                drafts: 3,
                committed: 4,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 8_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 0,
            });
            assert_eq!(
                action,
                AdaptiveAction::Continue,
                "must not oscillate back up"
            );
        }

        let report = ctrl.report();
        assert!(report.adaptive);
        assert_eq!(report.initial_depth, 3);
        assert_eq!(report.final_depth, 3);
        assert_eq!(report.depth_changes, 2);
        let cost = report.cost.expect("timed cost summary");
        assert!(
            cost.proposal_ns > 0 && cost.verify_ns > 0 && cost.rollback_ns > 0 && cost.sync_ns > 0
        );
        assert!(cost.ms_per_committed_token().unwrap() > 0.0);
    }

    #[test]
    fn mid_request_probe_switches_to_prompt_lookup_and_back_to_gpu_without_oscillation() {
        let caps = caps_all();
        let plan = resolve_auto_candidates(
            &caps,
            AutoCandidateContext {
                auto_prefers_block_draft: false,
                ..Default::default()
            },
        );
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::Candle,
            true,
            7,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // Establish lagging MTP baseline (avg 1.5/3 = 50% acceptance, ~6ms/tok: 15ms for 2.5
        // tokens), then trigger repeated context.
        let mut probe_started = false;
        for step in 0..ACCEPTANCE_PROBE_VERIFIES + 4 {
            let p = ctrl.next_step();
            if p.proposer == ProposerKind::PromptLookup {
                probe_started = true;
                break;
            }
            let accepted = if step % 2 == 0 { 1 } else { 2 };
            let committed = 1 + accepted;
            tokens += committed as u64;
            let peek = if step >= ACCEPTANCE_PROBE_VERIFIES {
                4
            } else {
                0
            };
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted,
                drafts: 3,
                committed,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 9_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: peek,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ) {
                probe_started = true;
                break;
            }
        }
        assert!(probe_started, "expected bounded probe of prompt_lookup");

        // During bounded probe, PromptLookup accepts 4 drafts at 12ms total (2.4ms/tok << 6ms/tok).
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            let p = ctrl.next_step();
            assert_eq!(p.proposer, ProposerKind::PromptLookup);
            tokens += 5;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 4,
                drafts: 4,
                committed: 5,
                generated_tokens: tokens,
                cost: step_cost(10_000, 10_500_000, 200_000, 1_290_000, 1),
                lookup_peek_len: 4,
            });
            assert_eq!(action, AdaptiveAction::Continue);
        }
        assert_eq!(ctrl.next_step().proposer, ProposerKind::PromptLookup);
        assert_eq!(ctrl.report().proposer_switches, 1);

        // Now cooldown passes, then repeated context ends -> PromptLookup dries up -> probes back to MTP!
        for _ in 0..ADAPTIVE_COOLDOWN_VERIFIES {
            tokens += 5;
            let _ = ctrl.observe_step(AdaptiveStepInput {
                accepted: 4,
                drafts: 4,
                committed: 5,
                generated_tokens: tokens,
                cost: step_cost(10_000, 10_500_000, 200_000, 1_290_000, 1),
                lookup_peek_len: 4,
            });
        }
        let mut back_to_mtp_probe = false;
        for _ in 0..ADAPTIVE_LOOKUP_DRY_STEPS {
            tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 0,
                committed: 1,
                generated_tokens: tokens,
                cost: step_cost(10_000, 9_000_000, 0, 990_000, 1),
                lookup_peek_len: 0,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    from: ProposerKind::PromptLookup,
                    to: ProposerKind::Mtp,
                    ..
                }
            ) {
                back_to_mtp_probe = true;
            }
        }
        assert!(
            back_to_mtp_probe,
            "expected probe back to MTP when lookup dried up"
        );
        assert_eq!(ctrl.next_step().proposer, ProposerKind::Mtp);

        // MTP probe succeeds (2 accepted, 18ms -> 6ms/tok vs dry lookup's 10ms/tok? Actually
        // incumbent_ns_per_token was last measured window; verify clean probe completion).
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            tokens += 3;
            let _ = ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: tokens,
                cost: step_cost(500_000, 4_000_000, 200_000, 300_000, 1),
                lookup_peek_len: 0,
            });
        }
        assert_eq!(ctrl.next_step().proposer, ProposerKind::Mtp);
        let report = ctrl.report();
        assert_eq!(report.initial_proposer, ProposerKind::Mtp);
        assert_eq!(report.final_proposer, ProposerKind::Mtp);
        assert!(report.proposer_switches <= ADAPTIVE_MAX_SWITCHES);
        assert!(report
            .events
            .iter()
            .any(|e| e.kind == AdaptiveEventKind::ProposerSwitched
                && e.to_proposer == ProposerKind::PromptLookup));
    }

    #[test]
    fn failed_probe_rules_out_alternate_and_prevents_oscillation() {
        let caps = caps_all();
        let plan = resolve_auto_candidates(&caps, Default::default());
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::Candle,
            true,
            7,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // 4 steps with a lagging MTP incumbent (1/3 = 33% < 55%) and lookup_peek_len >= 3
        // trigger a bounded probe of PromptLookup.
        let mut started = false;
        for _ in 0..6 {
            tokens += 2;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 3,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(3_000_000, 8_000_000, 500_000, 500_000, 1),
                lookup_peek_len: 3,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ) {
                started = true;
                break;
            }
        }
        assert!(started);

        // Probe accepts 0 drafts for 3 steps -> fast abort, revert to MTP, rule out PromptLookup!
        let mut reverted = false;
        for _ in 0..3 {
            tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 3,
                committed: 1,
                generated_tokens: tokens,
                cost: step_cost(10_000, 12_000_000, 200_000, 500_000, 1),
                lookup_peek_len: 3,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    from: ProposerKind::PromptLookup,
                    to: ProposerKind::Mtp,
                    ..
                }
            ) {
                reverted = true;
            }
        }
        assert!(reverted);
        assert!(!ctrl.can_switch_to(ProposerKind::PromptLookup));

        // Even with endless lookup_peek_len >= 2 after cooldown, PromptLookup is NEVER probed again!
        for _ in 0..32 {
            tokens += 3;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: tokens,
                cost: step_cost(3_000_000, 8_000_000, 500_000, 500_000, 1),
                lookup_peek_len: 4,
            });
            assert!(!matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ));
        }
        let report = ctrl.report();
        assert_eq!(report.proposer_switches, 0);
        assert!(report
            .events
            .iter()
            .any(|e| e.kind == AdaptiveEventKind::ProbeReverted));
    }

    #[test]
    fn explicit_modes_remain_strictly_authoritative_and_never_adapt() {
        let off = AdaptiveSpeculativeController::for_request(
            Speculative::Off,
            AutoSpeculativePlan::default(),
            vec![ProposerKind::None],
            PlainDecode::MlxPipelined,
            true,
            0,
            ProposerKind::None,
            0,
            None,
        );
        assert!(!off.is_adaptive());
        assert_eq!(off.next_step().proposer, ProposerKind::None);
        assert_eq!(off.next_step().depth, 0);
        assert_eq!(
            off.report().plain_reason.as_deref(),
            Some("speculative: off")
        );

        let mut explicit = AdaptiveSpeculativeController::for_request(
            Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
            AutoSpeculativePlan::single(ProposerKind::PromptLookup, 4, 7),
            vec![ProposerKind::PromptLookup],
            PlainDecode::MlxPipelined,
            true,
            7,
            ProposerKind::PromptLookup,
            4,
            None,
        );
        assert!(!explicit.is_adaptive());
        assert!(!explicit.probing_plain());
        for i in 1..=40u64 {
            let p = explicit.next_step();
            assert_eq!((p.proposer, p.depth), (ProposerKind::PromptLookup, 4));
            let action = explicit.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 4,
                committed: 1,
                generated_tokens: i,
                cost: step_cost(10_000, 50_000_000, 1_000_000, 1_000_000, 1),
                lookup_peek_len: 0,
            });
            assert_eq!(action, AdaptiveAction::Continue);
        }
        let report = explicit.report();
        assert!(!report.adaptive);
        assert_eq!(report.initial_proposer, ProposerKind::PromptLookup);
        assert_eq!(report.final_proposer, ProposerKind::PromptLookup);
        assert_eq!(report.initial_depth, 4);
        assert_eq!(report.final_depth, 4);
        assert_eq!(report.depth_changes, 0);
        assert_eq!(report.proposer_switches, 0);
        assert_eq!(explicit.last_monitor_decision(), None);
    }

    #[test]
    fn mlx_pipelined_guard_demotes_on_static_loss_even_when_unpipelined_probe_looks_fast() {
        let plan = AutoSpeculativePlan::single(ProposerKind::PromptLookup, 4, 7);
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![ProposerKind::PromptLookup, ProposerKind::None],
            PlainDecode::MlxPipelined,
            true,
            7,
            ProposerKind::PromptLookup,
            4,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // 0 accepted drafts, same 10ms cost as unpipelined plain probe -> measured gain = 1.0
        // (passes measured check), but MLX pipelined static floor (0.4) MUST demote!
        let mut demoted = false;
        for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
            tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 4,
                committed: 1,
                generated_tokens: tokens,
                cost: step_cost(10_000, 9_500_000, 200_000, 290_000, 1),
                lookup_peek_len: 0,
            });
            if action == AdaptiveAction::DemoteToPlain {
                demoted = true;
            }
        }
        assert!(demoted, "MlxPipelined must demote on static loss");
        assert!(ctrl.is_demoted());
        assert_eq!(
            ctrl.last_monitor_decision().map(|d| (d.demoted, d.basis)),
            Some((true, DemotionBasis::Static))
        );
    }

    #[test]
    fn adaptive_report_roundtrips_json_cleanly() {
        let mut summary = AdaptiveCostSummary::default();
        summary.record(step_cost(1_000_000, 5_000_000, 250_000, 750_000, 2), 4);
        let report = AdaptiveSpeculativeReport {
            adaptive: true,
            candidates: vec![
                ProposerKind::BlockDraft,
                ProposerKind::Mtp,
                ProposerKind::DraftModel,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            initial_proposer: ProposerKind::BlockDraft,
            initial_depth: 4,
            final_proposer: ProposerKind::PromptLookup,
            final_depth: 6,
            depth_changes: 1,
            proposer_switches: 1,
            events: vec![AdaptiveSpeculativeEvent {
                token_offset: 32,
                verify_step: 10,
                kind: AdaptiveEventKind::ProposerSwitched,
                from_proposer: ProposerKind::BlockDraft,
                from_depth: 4,
                to_proposer: ProposerKind::PromptLookup,
                to_depth: 4,
                reason: "lookup faster".into(),
            }],
            plain_reason: None,
            cost: Some(summary),
        };
        let json = serde_json::to_string(&report).expect("serialize");
        let parsed: AdaptiveSpeculativeReport = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, report);
    }

    #[test]
    fn marginal_loss_shrinks_depth_before_demoting_and_auto_stayed_plain_records_reason() {
        // Marginal loss with partial acceptance (1/4 accepted at 2.1x plain cost -> gain 0.95 ->
        // wait: 2 tokens / 22ms vs 10ms plain -> gain 0.909, between CLEAR_LOSS_GAIN=0.8 and 0.95):
        // shrinks depth first instead of immediately demoting.
        let plan = AutoSpeculativePlan::single(ProposerKind::Mtp, 4, 4);
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![ProposerKind::Mtp, ProposerKind::None],
            PlainDecode::Candle,
            true,
            4,
            ProposerKind::Mtp,
            4,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        let mut shrunk = None;
        for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
            tokens += 2;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 4,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(6_000_000, 14_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 0,
            });
            if let AdaptiveAction::ResizeDepth { from, to } = action {
                shrunk = Some((from, to));
            }
        }
        assert_eq!(shrunk, Some((4, 1)));
        assert!(!ctrl.is_demoted());
        assert_eq!(ctrl.last_monitor_decision().map(|d| d.demoted), Some(false));

        // Auto with no eligible proposers stays plain with StayedPlain event + plain_reason.
        let empty = resolve_auto_candidates(&TextLlmCapabilities::default(), Default::default());
        let plain_ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            empty,
            vec![ProposerKind::None],
            PlainDecode::MlxPipelined,
            true,
            0,
            ProposerKind::None,
            0,
            None,
        );
        let report = plain_ctrl.report();
        assert!(report.adaptive);
        assert_eq!(report.initial_proposer, ProposerKind::None);
        assert_eq!(report.final_proposer, ProposerKind::None);
        assert!(report.plain_reason.is_some());
        assert_eq!(report.events.len(), 1);
        assert_eq!(report.events[0].kind, AdaptiveEventKind::StayedPlain);
    }

    #[test]
    fn failed_probe_demotes_to_plain_when_incumbent_retired_during_probe() {
        let plan = AutoSpeculativePlan {
            candidates: vec![
                AutoCandidate {
                    proposer: ProposerKind::Mtp,
                    recommended_depth: 2,
                    max_depth: 4,
                },
                AutoCandidate {
                    proposer: ProposerKind::PromptLookup,
                    recommended_depth: 4,
                    max_depth: 6,
                },
                AutoCandidate::plain(),
            ],
            fallbacks: Vec::new(),
        };
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::Candle,
            true,
            6,
            ProposerKind::Mtp,
            2,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // Establish a lagging MTP baseline (1/2 = 50% < 55%) and trigger a bounded probe to
        // PromptLookup.
        let mut started_probe = false;
        for _ in 0..8 {
            tokens += 2;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 2,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(1_000_000, 10_000_000, 100_000, 200_000, 1),
                lookup_peek_len: 4,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    from: ProposerKind::Mtp,
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ) {
                started_probe = true;
                break;
            }
        }
        assert!(started_probe, "expected bounded probe to PromptLookup");
        assert_eq!(ctrl.next_step().proposer, ProposerKind::PromptLookup);

        // Retire the incumbent MTP while the PromptLookup probe is in flight.
        ctrl.retire_proposer(ProposerKind::Mtp);

        // Run the probe with poor performance so the probe fails; it must demote cleanly to plain
        // rather than reverting to the retired MTP proposer.
        let mut final_action = AdaptiveAction::Continue;
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 4,
                committed: 1,
                generated_tokens: tokens,
                cost: step_cost(100_000, 14_000_000, 200_000, 200_000, 1),
                lookup_peek_len: 0,
            });
            if action != AdaptiveAction::Continue {
                final_action = action;
                break;
            }
        }
        assert_eq!(final_action, AdaptiveAction::DemoteToPlain);
        assert!(ctrl.is_demoted());
        assert_eq!(ctrl.next_step().proposer, ProposerKind::None);
        assert_eq!(ctrl.next_step().depth, 0);
        let report = ctrl.report();
        assert_eq!(report.final_proposer, ProposerKind::None);
        assert_eq!(report.final_depth, 0);
        assert!(report.plain_reason.is_some());
        assert!(
            report
                .events
                .iter()
                .any(|e| e.kind == AdaptiveEventKind::FallbackToPlain),
            "expected FallbackToPlain event: {:?}",
            report.events
        );
        assert!(
            !report
                .events
                .iter()
                .any(|e| e.kind == AdaptiveEventKind::ProbeReverted),
            "must not emit ProbeReverted to retired incumbent: {:?}",
            report.events
        );
    }

    #[test]
    fn saturated_gpu_proposer_does_not_probe_prompt_lookup_on_short_or_repeated_streaks() {
        let caps = caps_all();
        for plain in [PlainDecode::Candle, PlainDecode::MlxPipelined] {
            let plan = resolve_auto_candidates(&caps, Default::default());
            let mut ctrl = AdaptiveSpeculativeController::for_request(
                Speculative::Auto,
                plan,
                vec![
                    ProposerKind::Mtp,
                    ProposerKind::PromptLookup,
                    ProposerKind::None,
                ],
                plain,
                true,
                3,
                ProposerKind::Mtp,
                3,
                None,
            );
            let mut tokens = 1u64;
            run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

            // Saturated MTP@3 (3/3 accepted = 100%, 6.18ms/tok): neither short match=2 streaks
            // nor sustained match=4 streaks may interrupt it for a wasteful PromptLookup probe.
            for step in 0..32u32 {
                let p = ctrl.next_step();
                assert_eq!((p.proposer, p.depth), (ProposerKind::Mtp, 3));
                tokens += 4;
                let peek = if step < 16 { 2 } else { 4 };
                let action = ctrl.observe_step(AdaptiveStepInput {
                    accepted: 3,
                    drafts: 3,
                    committed: 4,
                    generated_tokens: tokens,
                    cost: step_cost(1_200_000, 22_000_000, 0, 1_520_000, 0),
                    lookup_peek_len: peek,
                });
                assert_eq!(
                    action,
                    AdaptiveAction::Continue,
                    "saturated MTP@3 on {plain:?} must not probe PromptLookup at step {step} (peek={peek})"
                );
            }
            let report = ctrl.report();
            assert_eq!(report.final_proposer, ProposerKind::Mtp);
            assert_eq!(report.proposer_switches, 0);
            assert!(
                !report
                    .events
                    .iter()
                    .any(|e| e.kind == AdaptiveEventKind::ProbeStarted),
                "unexpected ProbeStarted on saturated MTP@3 ({plain:?}): {:?}",
                report.events
            );
        }

        // On Candle (strict 0.55 ceiling), even 2/3 (67%) MTP must not trigger a variable-width
        // PromptLookup probe, whereas on MLX (0.70 ceiling) 2/3 DOES probe on match>=3 (not match=2).
        let plan = resolve_auto_candidates(&caps, Default::default());
        let mut candle_ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan.clone(),
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::Candle,
            true,
            4,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut candle_ctrl, 10_000_000, &mut tokens);
        for _ in 0..12 {
            tokens += 3;
            let action = candle_ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: tokens,
                cost: step_cost(2_000_000, 14_000_000, 200_000, 800_000, 1),
                lookup_peek_len: 4,
            });
            assert_eq!(
                action,
                AdaptiveAction::Continue,
                "Candle MTP@3 at 67% acceptance (>= 55%) must not probe PromptLookup"
            );
        }

        let mut mlx_ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::MlxPipelined,
            true,
            4,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut mlx_tokens = 1u64;
        run_plain_probe(&mut mlx_ctrl, 10_000_000, &mut mlx_tokens);
        // Short match=2 streak must not trigger even on MLX at 67%.
        for _ in 0..6 {
            mlx_tokens += 3;
            let action = mlx_ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: mlx_tokens,
                cost: step_cost(2_000_000, 14_000_000, 200_000, 800_000, 1),
                lookup_peek_len: 2,
            });
            assert_eq!(action, AdaptiveAction::Continue);
        }
        // Challenging match=3 streak on MLX at 67% (< 70%) DOES trigger a probe.
        let mut mlx_probed = false;
        for _ in 0..ADAPTIVE_LOOKUP_TRIGGER_MATCHES {
            mlx_tokens += 3;
            let action = mlx_ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: mlx_tokens,
                cost: step_cost(2_000_000, 14_000_000, 200_000, 800_000, 1),
                lookup_peek_len: 3,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ) {
                mlx_probed = true;
            }
        }
        assert!(
            mlx_probed,
            "MLX MTP@3 at 67% (< 70%) should probe on match=3 streak"
        );
    }

    #[test]
    fn lagging_gpu_proposer_probes_and_switches_to_prompt_lookup_and_back_when_dry() {
        let caps = caps_all();
        let plan = resolve_auto_candidates(&caps, Default::default());
        let mut ctrl = AdaptiveSpeculativeController::for_request(
            Speculative::Auto,
            plan,
            vec![
                ProposerKind::Mtp,
                ProposerKind::PromptLookup,
                ProposerKind::None,
            ],
            PlainDecode::Candle,
            true,
            7,
            ProposerKind::Mtp,
            3,
            None,
        );
        let mut tokens = 1u64;
        run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

        // First 3 steps (< ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES = 4): even with low acceptance
        // and lookup_peek_len = 4, wait until at least 4 incumbent verifies are observed.
        for step in 0..(ADAPTIVE_LOOKUP_MIN_INCUMBENT_VERIFIES - 1) {
            tokens += 2;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 3,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 10_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 4,
            });
            assert_eq!(
                action,
                AdaptiveAction::Continue,
                "must wait for minimum incumbent verifies before probing (step {step})"
            );
        }

        // Step 4 with short peek=2 resets the streak and still does not probe.
        tokens += 2;
        assert_eq!(
            ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 3,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 10_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 2,
            }),
            AdaptiveAction::Continue
        );

        // Now 3 consecutive steps with peek=4 on a lagging incumbent (1/3 = 33% < 55%) start
        // a bounded PromptLookup probe.
        let mut probe_started = false;
        for _ in 0..ADAPTIVE_LOOKUP_TRIGGER_MATCHES {
            tokens += 2;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 1,
                drafts: 3,
                committed: 2,
                generated_tokens: tokens,
                cost: step_cost(4_000_000, 10_000_000, 500_000, 1_500_000, 1),
                lookup_peek_len: 4,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    from: ProposerKind::Mtp,
                    to: ProposerKind::PromptLookup,
                    ..
                }
            ) {
                probe_started = true;
            }
        }
        assert!(
            probe_started,
            "lagging MTP should trigger PromptLookup probe on peek=4 streak"
        );

        // PromptLookup wins the bounded probe (4/4 accepted at 12ms -> 2.4ms/tok vs 8ms/tok).
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            tokens += 5;
            assert_eq!(
                ctrl.observe_step(AdaptiveStepInput {
                    accepted: 4,
                    drafts: 4,
                    committed: 5,
                    generated_tokens: tokens,
                    cost: step_cost(10_000, 10_500_000, 200_000, 1_290_000, 1),
                    lookup_peek_len: 4,
                }),
                AdaptiveAction::Continue
            );
        }
        assert_eq!(ctrl.next_step().proposer, ProposerKind::PromptLookup);

        // Burn cooldown, then PromptLookup goes dry -> probes and switches back to MTP.
        for _ in 0..ADAPTIVE_COOLDOWN_VERIFIES {
            tokens += 5;
            let _ = ctrl.observe_step(AdaptiveStepInput {
                accepted: 4,
                drafts: 4,
                committed: 5,
                generated_tokens: tokens,
                cost: step_cost(10_000, 10_500_000, 200_000, 1_290_000, 1),
                lookup_peek_len: 4,
            });
        }
        let mut switched_back = false;
        for _ in 0..ADAPTIVE_LOOKUP_DRY_STEPS {
            tokens += 1;
            let action = ctrl.observe_step(AdaptiveStepInput {
                accepted: 0,
                drafts: 0,
                committed: 1,
                generated_tokens: tokens,
                cost: step_cost(10_000, 9_500_000, 0, 490_000, 1),
                lookup_peek_len: 0,
            });
            if matches!(
                action,
                AdaptiveAction::SwitchProposer {
                    from: ProposerKind::PromptLookup,
                    to: ProposerKind::Mtp,
                    ..
                }
            ) {
                switched_back = true;
            }
        }
        assert!(switched_back, "dry PromptLookup must probe back to MTP");
        for _ in 0..ADAPTIVE_PROBE_STEPS {
            tokens += 3;
            let _ = ctrl.observe_step(AdaptiveStepInput {
                accepted: 2,
                drafts: 3,
                committed: 3,
                generated_tokens: tokens,
                cost: step_cost(2_000_000, 10_000_000, 200_000, 800_000, 1),
                lookup_peek_len: 0,
            });
        }
        assert_eq!(ctrl.next_step().proposer, ProposerKind::Mtp);
        assert_eq!(ctrl.report().proposer_switches, 2);
    }

    #[test]
    fn prompt_lookup_depth_growth_requires_depth_saturation_not_just_proposed_ratio() {
        for (plain, expected_next_depth) in [
            (PlainDecode::Candle, 5u32),
            (PlainDecode::MlxPipelined, 6u32),
        ] {
            let plan = AutoSpeculativePlan::single(ProposerKind::PromptLookup, 4, 7);
            let mut ctrl = AdaptiveSpeculativeController::for_request(
                Speculative::Auto,
                plan,
                vec![ProposerKind::PromptLookup, ProposerKind::None],
                plain,
                true,
                7,
                ProposerKind::PromptLookup,
                4,
                None,
            );
            let mut tokens = 1u64;
            run_plain_probe(&mut ctrl, 10_000_000, &mut tokens);

            // Window 1: proposed = 2, accepted = 2 (100% of proposed, mean_accepted = 2.0/4 = 50%
            // depth saturation and 50% fill) -> MUST NOT grow depth!
            for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
                let p = ctrl.next_step();
                assert_eq!((p.proposer, p.depth), (ProposerKind::PromptLookup, 4));
                tokens += 3;
                let action = ctrl.observe_step(AdaptiveStepInput {
                    accepted: 2,
                    drafts: 2,
                    committed: 3,
                    generated_tokens: tokens,
                    cost: step_cost(10_000, 10_500_000, 100_000, 390_000, 1),
                    lookup_peek_len: 0,
                });
                assert_eq!(
                    action,
                    AdaptiveAction::Continue,
                    "partial-fill PromptLookup (2/4) on {plain:?} must not grow depth"
                );
            }
            assert_eq!(ctrl.next_step().depth, 4);
            assert_eq!(ctrl.report().depth_changes, 0);

            // Window 2: proposed = 3, accepted = 3 (100% of proposed, 75% saturation, but only
            // 75% fill < 80% — never even filled depth 4) -> MUST NOT grow depth!
            for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
                let p = ctrl.next_step();
                assert_eq!((p.proposer, p.depth), (ProposerKind::PromptLookup, 4));
                tokens += 4;
                let action = ctrl.observe_step(AdaptiveStepInput {
                    accepted: 3,
                    drafts: 3,
                    committed: 4,
                    generated_tokens: tokens,
                    cost: step_cost(10_000, 11_000_000, 100_000, 390_000, 1),
                    lookup_peek_len: 0,
                });
                assert_eq!(
                    action,
                    AdaptiveAction::Continue,
                    "sub-fill PromptLookup (3/4 = 75% fill < 80%) on {plain:?} must not grow depth"
                );
            }
            assert_eq!(ctrl.next_step().depth, 4);
            assert_eq!(ctrl.report().depth_changes, 0);

            // Window 3: proposed = 4, accepted = 3 (100% fill >= 80%, 75% saturation >= 75%,
            // mean_accepted = 3.0/4) -> DOES grow depth!
            let mut grew_to = None;
            for _ in 0..ACCEPTANCE_PROBE_VERIFIES {
                let p = ctrl.next_step();
                assert_eq!((p.proposer, p.depth), (ProposerKind::PromptLookup, 4));
                tokens += 4;
                let action = ctrl.observe_step(AdaptiveStepInput {
                    accepted: 3,
                    drafts: 4,
                    committed: 4,
                    generated_tokens: tokens,
                    cost: step_cost(10_000, 11_500_000, 100_000, 390_000, 1),
                    lookup_peek_len: 0,
                });
                if let AdaptiveAction::ResizeDepth { from, to } = action {
                    assert_eq!(from, 4);
                    grew_to = Some(to);
                }
            }
            assert_eq!(
                grew_to,
                Some(expected_next_depth),
                "saturated PromptLookup (proposed=4, accepted=3) on {plain:?} must grow depth"
            );
            assert_eq!(ctrl.next_step().depth, expected_next_depth);
        }
    }
}

//! Pure-mock adaptive speculative engine tests for MLX (feature SW-F357, story SW-7237).
//!
//! Zero learned model weights or NN `.forward()` calls — exercises [`generate_speculative`]
//! with a deterministic mock [`SpeculativeTarget`], mock [`Proposer`], and fake [`DecodeClock`]
//! to verify:
//! - AC1: Cost-aware depth adaptation within advertised/provisioned bounds, bounded mid-request
//!   proposer switching (`Mtp <-> PromptLookup -> Plain`), catch-up continuity, pipelined plain
//!   handoff on demotion, and anti-oscillation.
//! - AC2: Exact greedy parity and exact stochastic acceptance through rejection, penalties,
//!   rewindable constraints, stop tokens, caller stop, cancellation, and prefix-cache resume.
//! - AC3: Structured [`AdaptiveSpeculativeReport`] telemetry for adaptive runs, stayed-plain runs,
//!   and authoritative explicit `Off` / `{proposer, depth}` requests.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use core_llm::{
    with_decode_clock, AdaptiveEventKind, AutoCandidate, AutoSpeculativePlan, DecodeClock,
    ProposerKind, Speculative, SpeculativeProposer,
};
use mlx_rs::Array;

use super::*;
use crate::primitives::sampler::SamplingParams;

#[derive(Default)]
struct FakeClock(Cell<Duration>);

impl FakeClock {
    fn advance(&self, delta: Duration) {
        self.0.set(self.0.get() + delta);
    }
}

impl DecodeClock for FakeClock {
    fn now(&self) -> Duration {
        self.0.get()
    }
}

struct MockCache {
    len: i32,
}

struct MockRollback {
    provisioned_width: usize,
    armed: bool,
}

impl CacheRollback<MockCache> for MockRollback {
    fn label(&self) -> &'static str {
        "truncate"
    }

    fn begin(&mut self, _cache: &mut MockCache) -> Result<()> {
        self.armed = true;
        Ok(())
    }

    fn recover(&mut self, cache: &mut MockCache, keep: i32) -> Result<Rollback> {
        assert!(
            keep >= 0 && keep <= cache.len,
            "invalid rollback {keep} > {}",
            cache.len
        );
        assert!(
            (cache.len - keep) as usize <= self.provisioned_width + 1,
            "rollback exceeded provisioned width {}",
            self.provisioned_width
        );
        cache.len = keep;
        self.armed = false;
        Ok(Rollback::Direct)
    }
}

/// Pure-mock [`SpeculativeTarget`] emitting deterministic multi-logit rows (`[1, n, vocab]`) and
/// position-indexed hidden rows (`[1, n, 1]`) with zero learned model layers.
struct ScriptedTarget {
    vocab: usize,
    prompt_len: usize,
    sequence: Vec<i32>,
    clock: Rc<FakeClock>,
    plain_step: Duration,
    verify_extra_per_draft: Duration,
    expensive_after_step: Option<usize>,
    expensive_verify: Duration,
    cancel_on_forward: Option<(u64, CancelFlag)>,
    forwards: Cell<u64>,
    last_provisioned_width: Cell<usize>,
    supports_tree: bool,
    tree_forwards: Cell<u64>,
}

impl ScriptedTarget {
    fn new(vocab: usize, prompt_len: usize, sequence: Vec<i32>, clock: Rc<FakeClock>) -> Self {
        Self {
            vocab,
            prompt_len,
            sequence,
            clock,
            plain_step: Duration::from_millis(10),
            verify_extra_per_draft: Duration::from_micros(600),
            expensive_after_step: None,
            expensive_verify: Duration::from_millis(24),
            cancel_on_forward: None,
            forwards: Cell::new(0),
            last_provisioned_width: Cell::new(0),
            supports_tree: false,
            tree_forwards: Cell::new(0),
        }
    }

    fn target_token(&self, step_idx: usize) -> i32 {
        self.sequence
            .get(step_idx)
            .copied()
            .unwrap_or(((step_idx + 1) % self.vocab) as i32)
    }

    fn logits_for_position(&self, cache_pos: usize) -> Vec<f32> {
        let mut row = vec![-2.0f32; self.vocab];
        let step_idx = cache_pos.saturating_sub(self.prompt_len.saturating_sub(1));
        let top = (self.target_token(step_idx) as usize) % self.vocab;
        row[top] = 4.5;
        row[(top + 1) % self.vocab] = 3.2;
        row[(top + 2) % self.vocab] = 2.6;
        row[(top + 3) % self.vocab] = 1.9;
        row
    }
}

impl SpeculativeTarget for ScriptedTarget {
    type Cache = MockCache;
    type Rollback = MockRollback;

    fn new_cache(&self) -> MockCache {
        MockCache { len: 0 }
    }

    fn cache_len(&self, cache: &MockCache) -> i32 {
        cache.len
    }

    fn rollback(&self, width: usize) -> MockRollback {
        self.last_provisioned_width.set(width);
        MockRollback {
            provisioned_width: width,
            armed: false,
        }
    }

    fn forward(
        &self,
        cache: &mut MockCache,
        ids: &Array,
        _rope_offset: i32,
        scope: LogitsScope,
        want_hidden: bool,
    ) -> Result<TargetOutput> {
        let call = self.forwards.get() + 1;
        self.forwards.set(call);
        let n = ids.shape()[1] as usize;
        let start = cache.len as usize;
        let is_prefill = start == 0 && n == self.prompt_len;
        if !is_prefill && n > 1 {
            let provisioned = self.last_provisioned_width.get();
            if provisioned > 0 {
                assert!(
                    n - 1 <= provisioned,
                    "verify drafts {} exceeded provisioned width {}",
                    n - 1,
                    provisioned
                );
            }
            let step_idx = start.saturating_sub(self.prompt_len);
            let cost = if self
                .expensive_after_step
                .is_some_and(|threshold| step_idx >= threshold)
            {
                self.expensive_verify
            } else {
                self.plain_step + self.verify_extra_per_draft * (n as u32 - 1)
            };
            self.clock.advance(cost);
        } else if !is_prefill {
            self.clock.advance(self.plain_step);
        }
        cache.len += n as i32;
        if let Some((trip_at, flag)) = &self.cancel_on_forward {
            if call == *trip_at {
                flag.cancel();
            }
        }
        let mut rows = Vec::with_capacity(n * self.vocab);
        for i in 0..n {
            rows.extend_from_slice(&self.logits_for_position(start + i));
        }
        let logits = match scope {
            LogitsScope::All => Array::from_slice(&rows, &[1, n as i32, self.vocab as i32]),
            LogitsScope::Last => {
                let last = &rows[(n - 1) * self.vocab..];
                Array::from_slice(last, &[1, self.vocab as i32])
            }
        };
        let hidden = if want_hidden {
            let h: Vec<f32> = (0..n).map(|i| (start + i) as f32).collect();
            Some(Array::from_slice(&h, &[1, n as i32, 1]))
        } else {
            None
        };
        Ok(TargetOutput { logits, hidden })
    }

    fn attention_label(&self) -> &'static str {
        "mock"
    }

    fn tree_unsupported(&self, _cache: &MockCache) -> Option<String> {
        if self.supports_tree {
            None
        } else {
            Some("this target verifies only linear proposals".into())
        }
    }

    fn forward_tree(
        &self,
        _cache: &mut MockCache,
        _inputs: &TreeInputs,
        _want_hidden: bool,
    ) -> Result<TargetOutput> {
        self.tree_forwards.set(self.tree_forwards.get() + 1);
        Err(Error::Msg(
            "unexpected forward_tree on non-tree-building primary".into(),
        ))
    }
}

struct MockPrimaryProposer {
    kind: ProposerKind,
    vocab: usize,
    prompt_len: usize,
    sequence: Vec<i32>,
    clock: Rc<FakeClock>,
    propose_cost: Duration,
    accurate_ranges: Vec<std::ops::Range<usize>>,
    emit_stochastic_dists: bool,
    propose_depths: Vec<usize>,
    catch_up_calls: Vec<(Vec<i32>, Option<usize>, i32)>,
    commits: usize,
}

impl MockPrimaryProposer {
    fn new(
        kind: ProposerKind,
        vocab: usize,
        prompt_len: usize,
        sequence: Vec<i32>,
        clock: Rc<FakeClock>,
        propose_cost: Duration,
        accurate_ranges: Vec<std::ops::Range<usize>>,
    ) -> Self {
        Self {
            kind,
            vocab,
            prompt_len,
            sequence,
            clock,
            propose_cost,
            accurate_ranges,
            emit_stochastic_dists: false,
            propose_depths: Vec::new(),
            catch_up_calls: Vec::new(),
            commits: 0,
        }
    }

    fn is_accurate(&self, step_idx: usize) -> bool {
        self.accurate_ranges.iter().any(|r| r.contains(&step_idx))
    }
}

impl Proposer<ScriptedTarget> for MockPrimaryProposer {
    fn kind(&self) -> ProposerKind {
        self.kind
    }

    fn wants_hidden(&self) -> bool {
        matches!(self.kind, ProposerKind::Mtp | ProposerKind::BlockDraft)
    }

    fn warm(
        &mut self,
        _target: &ScriptedTarget,
        _prompt: &[i32],
        _prompt_hidden: Option<&Array>,
    ) -> Result<Option<Array>> {
        Ok(None)
    }

    fn propose(
        &mut self,
        _target: &ScriptedTarget,
        ctx: &ProposeContext<'_>,
        _sampler: &mut DraftSampler<'_, '_>,
    ) -> Result<Proposal> {
        self.propose_depths.push(ctx.max_drafts);
        self.clock.advance(self.propose_cost);
        let at = ctx.history.len().saturating_sub(self.prompt_len);
        let mut drafts = Vec::with_capacity(ctx.max_drafts);
        let mut dists = Vec::new();
        for i in 0..ctx.max_drafts {
            let step = at + i;
            let expected = self
                .sequence
                .get(step)
                .copied()
                .unwrap_or(((step + 1) % self.vocab) as i32);
            let tok = if self.is_accurate(step) {
                expected
            } else {
                ((expected as usize + 5) % self.vocab) as i32
            };
            drafts.push(tok);
            if self.emit_stochastic_dists {
                let alt = ((tok as usize + 1) % self.vocab) as i32;
                dists.push(vec![(tok, 0.7), (alt, 0.3)]);
            }
        }
        Ok(Proposal { drafts, dists })
    }

    fn commit(
        &mut self,
        _target: &ScriptedTarget,
        _cur: i32,
        _accepted: &[i32],
        _kept_hidden: Option<&Array>,
        _position: i32,
    ) -> Result<()> {
        self.commits += 1;
        Ok(())
    }

    fn catch_up(
        &mut self,
        _target: &ScriptedTarget,
        tokens: &[i32],
        preceding_hidden: Option<&Array>,
        position: i32,
    ) -> Result<()> {
        let rows = preceding_hidden.map(|h| h.shape()[1] as usize);
        if let Some(h) = preceding_hidden {
            eval([h])?;
            let vals: Vec<f32> = h.as_slice::<f32>().to_vec();
            for (i, &val) in vals.iter().enumerate() {
                assert_eq!(
                    val as i32,
                    position - 1 + i as i32,
                    "catch_up hidden row mismatch at offset {i}"
                );
            }
        }
        self.catch_up_calls.push((tokens.to_vec(), rows, position));
        Ok(())
    }
}

struct ForbidMask {
    allow: Vec<bool>,
    accepted: Vec<i32>,
    rewinds: usize,
}

impl ForbidMask {
    fn new(vocab: usize, forbidden: &[usize]) -> Self {
        let mut allow = vec![true; vocab];
        for &f in forbidden {
            allow[f] = false;
        }
        Self {
            allow,
            accepted: Vec::new(),
            rewinds: 0,
        }
    }
}

impl ConstraintMask for ForbidMask {
    fn allowed(&mut self) -> &[bool] {
        &self.allow
    }

    fn accept(&mut self, token: i32) {
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for ForbidMask {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }

    fn rewind(&mut self, checkpoint: usize) {
        self.rewinds += 1;
        self.accepted.truncate(checkpoint);
    }
}

fn greedy_config(max_new_tokens: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens,
        sampling: SamplingParams::default(),
        seed: Some(42),
        stop_tokens: Vec::new(),
    }
}

fn mtp_and_lookup_plan(recommended: u32, max_depth: u32) -> AutoSpeculativePlan {
    AutoSpeculativePlan {
        candidates: vec![
            AutoCandidate {
                proposer: ProposerKind::Mtp,
                recommended_depth: recommended,
                max_depth,
            },
            AutoCandidate {
                proposer: ProposerKind::PromptLookup,
                recommended_depth: 4,
                max_depth: 6,
            },
            AutoCandidate::plain(),
        ],
        fallbacks: Vec::new(),
    }
}

#[test]
fn adaptive_resizes_depth_switches_proposers_and_hands_off_to_pipelined_plain() {
    let vocab = 128;
    let prompt: Vec<i32> = (10..26).collect();
    let mut seq = Vec::new();
    // Phase 1 (0..75): open-ended non-repeating; MTP accurate -> depth grows 2 -> 3+.
    for i in 0..75 {
        seq.push(30 + (i % 90));
    }
    // Phase 2 (75..155): repeated prompt spans; MTP lags on verbatim spans while prompt_lookup
    // wins bounded probe.
    for i in 0..80 {
        seq.push(prompt[i % prompt.len()]);
    }
    // Phase 3 (155..215): open-ended non-repeating; prompt_lookup dries up, MTP catches up & wins back.
    for i in 0..60 {
        seq.push(30 + ((i * 7 + 13) % 90));
    }
    // Phase 4 (215..300): non-repeating tokens where MTP stops matching on expensive verify ->
    // demotes to pipelined plain!
    for i in 0..85 {
        seq.push((i * 17 + i * i) % 120);
    }

    let config = greedy_config(seq.len());
    let plain_clock = Rc::new(FakeClock::default());
    let plain_target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), plain_clock);
    let expected = generate_speculative(
        &plain_target,
        &mut NoProposer,
        SpeculativePrompt::Tokens(&prompt),
        &config,
        0,
        &CancelFlag::new(),
        &mut |_| {},
        EngineOptions {
            pipelining: Pipelining::Off,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(expected.output.tokens, seq);

    let clock = Rc::new(FakeClock::default());
    let mut target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    target.expensive_after_step = Some(215);
    target.expensive_verify = Duration::from_millis(24);
    let mut proposer = MockPrimaryProposer::new(
        ProposerKind::Mtp,
        vocab,
        prompt.len(),
        seq.clone(),
        clock.clone(),
        Duration::from_millis(8),
        vec![0..75, 155..215],
    );

    let run = with_decode_clock(Some(clock), || {
        generate_speculative(
            &target,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                pipelining: Pipelining::Auto,
                speculative_mode: Speculative::Auto,
                adaptive_plan: Some(mtp_and_lookup_plan(2, 5)),
                provisioned_max_depth: Some(6),
                ..Default::default()
            },
        )
    })
    .unwrap();

    assert_eq!(run.output.tokens, expected.output.tokens);
    assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);
    assert!(
        proposer
            .propose_depths
            .iter()
            .all(|&d| (1..=5).contains(&d)),
        "depths stayed in 1..=5: {:?}",
        proposer.propose_depths
    );
    assert!(
        proposer.propose_depths.iter().any(|&d| d > 2),
        "expected depth growth above 2: {:?}",
        proposer.propose_depths
    );
    assert!(
        proposer.catch_up_calls.len() >= 2,
        "expected catch-up after plain probe and PromptLookup switch-back: {:?}",
        proposer.catch_up_calls
    );
    assert!(
        run.stats.pipelined > 0,
        "demotion under Pipelining::Auto must hand off to pipelined plain decode"
    );

    let adaptive = run
        .report
        .adaptive_speculative
        .as_ref()
        .expect("DecodeReport must carry adaptive_speculative telemetry");
    assert!(adaptive.adaptive);
    assert_eq!(adaptive.initial_proposer, ProposerKind::Mtp);
    assert_eq!(adaptive.initial_depth, 2);
    assert!(adaptive.depth_changes >= 1);
    assert!(adaptive.proposer_switches >= 2);
    assert!(adaptive.events.iter().any(|e| matches!(
        e.kind,
        AdaptiveEventKind::DepthIncreased | AdaptiveEventKind::DepthDecreased
    )));
    assert!(adaptive.events.iter().any(|e| {
        e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::PromptLookup
    }));
    assert!(adaptive.events.iter().any(|e| {
        e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::Mtp
    }));
    assert!(adaptive
        .events
        .iter()
        .any(|e| e.kind == AdaptiveEventKind::FallbackToPlain));
    assert_eq!(adaptive.final_proposer, ProposerKind::None);
    assert_eq!(adaptive.final_depth, 0);
    assert!(run.report.speculative_demoted_at.is_some());
    assert!(run.report.speculative_monitor.is_some_and(|m| m.demoted));

    let cost = adaptive.cost.as_ref().expect("timed cost summary present");
    assert!(cost.proposal_ns > 0);
    assert!(cost.verify_ns > 0);
    assert!(cost.total_ns > 0);
    assert!(cost.ms_per_committed_token().is_some_and(|ms| ms > 0.0));
}

#[test]
fn adaptive_preserves_greedy_and_stochastic_semantics_through_penalties_constraints_stops_cancel_and_prefilled(
) {
    let vocab = 32;
    let prompt: Vec<i32> = (0..10).collect();
    let mut seq = Vec::new();
    for i in 0..24 {
        seq.push(((i * 3 + 11) % vocab) as i32);
    }
    for i in 0..24 {
        seq.push(prompt[i % prompt.len()]);
    }
    for i in 0..16 {
        seq.push(((i * 5 + 13) % vocab) as i32);
    }

    let run_pair = |config: &GenerationConfig,
                    forbidden: Option<&[usize]>,
                    stop_after_tokens: Option<usize>,
                    cancel_at_forward: Option<u64>,
                    prefilled: bool| {
        let ref_clock = Rc::new(FakeClock::default());
        let mut ref_target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), ref_clock);
        let ref_cancel = CancelFlag::new();
        if let Some(trip) = cancel_at_forward {
            ref_target.cancel_on_forward = Some((trip, ref_cancel.clone()));
        }
        let mut ref_mask = forbidden.map(|f| ForbidMask::new(vocab, f));
        let ref_emitted = RefCell::new(0usize);
        let ref_stop = || stop_after_tokens.is_some_and(|n| *ref_emitted.borrow() >= n);
        let mut ref_cache = ref_target.new_cache();
        let ref_run = if prefilled {
            let prefill = ref_target
                .forward(
                    &mut ref_cache,
                    &input_ids(&prompt),
                    0,
                    LogitsScope::Last,
                    false,
                )
                .unwrap();
            generate_speculative(
                &ref_target,
                &mut NoProposer,
                SpeculativePrompt::Prefilled {
                    cache: &mut ref_cache,
                    logits: prefill.logits,
                    hidden: None,
                    history: &prompt,
                    position_delta: 0,
                },
                config,
                0,
                &ref_cancel,
                &mut |ev| {
                    if matches!(ev, StreamEvent::Token { .. }) {
                        *ref_emitted.borrow_mut() += 1;
                    }
                },
                EngineOptions {
                    constraint: ref_mask
                        .as_mut()
                        .map(|m| m as &mut dyn RewindableConstraintMask),
                    pipelining: Pipelining::Off,
                    ..Default::default()
                },
            )
            .unwrap()
        } else {
            generate_speculative(
                &ref_target,
                &mut NoProposer,
                SpeculativePrompt::Tokens(&prompt),
                config,
                0,
                &ref_cancel,
                &mut |ev| {
                    if matches!(ev, StreamEvent::Token { .. }) {
                        *ref_emitted.borrow_mut() += 1;
                    }
                },
                EngineOptions {
                    constraint: ref_mask
                        .as_mut()
                        .map(|m| m as &mut dyn RewindableConstraintMask),
                    should_stop: stop_after_tokens.map(|_| &ref_stop as &dyn Fn() -> bool),
                    pipelining: Pipelining::Off,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let clock = Rc::new(FakeClock::default());
        let mut target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
        let cancel = CancelFlag::new();
        if let Some(trip) = cancel_at_forward {
            target.cancel_on_forward = Some((trip, cancel.clone()));
        }
        let mut proposer = MockPrimaryProposer::new(
            ProposerKind::Mtp,
            vocab,
            prompt.len(),
            seq.clone(),
            clock.clone(),
            Duration::from_millis(4),
            vec![0..18, 24..48],
        );
        proposer.emit_stochastic_dists = config.sampling.temperature > 0.0;
        let mut mask = forbidden.map(|f| ForbidMask::new(vocab, f));
        let emitted = RefCell::new(0usize);
        let stop_fn = || stop_after_tokens.is_some_and(|n| *emitted.borrow() >= n);
        let mut cache = target.new_cache();
        let adaptive_run = with_decode_clock(Some(clock), || {
            if prefilled {
                let prefill = target
                    .forward(&mut cache, &input_ids(&prompt), 0, LogitsScope::Last, true)
                    .unwrap();
                generate_speculative(
                    &target,
                    &mut proposer,
                    SpeculativePrompt::Prefilled {
                        cache: &mut cache,
                        logits: prefill.logits,
                        hidden: prefill.hidden,
                        history: &prompt,
                        position_delta: 0,
                    },
                    config,
                    2,
                    &cancel,
                    &mut |ev| {
                        if matches!(ev, StreamEvent::Token { .. }) {
                            *emitted.borrow_mut() += 1;
                        }
                    },
                    EngineOptions {
                        constraint: mask
                            .as_mut()
                            .map(|m| m as &mut dyn RewindableConstraintMask),
                        pipelining: Pipelining::Off,
                        speculative_mode: Speculative::Auto,
                        adaptive_plan: Some(mtp_and_lookup_plan(2, 4)),
                        provisioned_max_depth: Some(6),
                        ..Default::default()
                    },
                )
            } else {
                generate_speculative(
                    &target,
                    &mut proposer,
                    SpeculativePrompt::Tokens(&prompt),
                    config,
                    2,
                    &cancel,
                    &mut |ev| {
                        if matches!(ev, StreamEvent::Token { .. }) {
                            *emitted.borrow_mut() += 1;
                        }
                    },
                    EngineOptions {
                        constraint: mask
                            .as_mut()
                            .map(|m| m as &mut dyn RewindableConstraintMask),
                        should_stop: stop_after_tokens.map(|_| &stop_fn as &dyn Fn() -> bool),
                        pipelining: Pipelining::Auto,
                        speculative_mode: Speculative::Auto,
                        adaptive_plan: Some(mtp_and_lookup_plan(2, 4)),
                        provisioned_max_depth: Some(6),
                        ..Default::default()
                    },
                )
            }
        })
        .unwrap();

        (ref_run, ref_mask, adaptive_run, mask)
    };

    // Case A: Greedy + repetition & presence penalties.
    let mut pen = greedy_config(48);
    pen.sampling.repetition_penalty = 1.4;
    pen.sampling.repetition_context = 12;
    pen.sampling.presence_penalty = 0.6;
    let (r, _, a, _) = run_pair(&pen, None, None, None, false);
    assert_eq!(
        a.output.tokens, r.output.tokens,
        "greedy + penalties parity"
    );
    assert_eq!(a.output.finish_reason, r.output.finish_reason);

    // Case B: Greedy + rewindable constraint mask.
    let forbidden = [0usize, 1, 11, 14, 17];
    let (r, rm, a, am) = run_pair(&greedy_config(48), Some(&forbidden), None, None, false);
    assert_eq!(
        a.output.tokens, r.output.tokens,
        "greedy + constraint parity"
    );
    assert_eq!(am.as_ref().unwrap().accepted, a.output.tokens);
    assert_eq!(rm.as_ref().unwrap().accepted, r.output.tokens);
    assert!(a
        .output
        .tokens
        .iter()
        .all(|&t| !forbidden.contains(&(t as usize))));

    // Case C: Greedy + stop token.
    let mut with_stop = greedy_config(48);
    with_stop.stop_tokens = vec![seq[27]];
    let (r, _, a, _) = run_pair(&with_stop, None, None, None, false);
    assert_eq!(
        a.output.tokens, r.output.tokens,
        "greedy + stop_token parity"
    );
    assert_eq!(a.output.finish_reason, FinishReason::StopToken);
    assert_eq!(r.output.finish_reason, FinishReason::StopToken);

    // Case D: Greedy + caller `should_stop`.
    let (r, _, a, _) = run_pair(&greedy_config(48), None, Some(21), None, false);
    assert_eq!(
        a.output.tokens, r.output.tokens,
        "greedy + should_stop parity"
    );
    assert_eq!(a.output.finish_reason, FinishReason::Stopped);
    assert_eq!(a.output.tokens.len(), 21);

    // Case E: Greedy + mid-run cancellation.
    let (_, _, a, _) = run_pair(&greedy_config(48), None, None, Some(7), false);
    assert_eq!(a.output.finish_reason, FinishReason::Cancelled);
    assert!(a.output.tokens.len() < 48);

    // Case F: Greedy + Prefilled (prefix-cache resume).
    let (r, _, a, _) = run_pair(&greedy_config(48), None, None, None, true);
    assert_eq!(
        a.output.tokens, r.output.tokens,
        "prefilled prefix-resume parity"
    );
    assert_eq!(a.output.finish_reason, FinishReason::MaxTokens);

    // Case G: Stochastic sampling + penalties + rewindable constraint.
    let mut stoch = greedy_config(48);
    stoch.sampling.temperature = 0.8;
    stoch.sampling.top_p = 0.9;
    stoch.sampling.top_k = 6;
    stoch.sampling.repetition_penalty = 1.25;
    stoch.sampling.repetition_context = 10;
    stoch.sampling.presence_penalty = 0.3;
    let (_, _, a_stoch, am_stoch) = run_pair(&stoch, Some(&forbidden), None, None, false);
    assert_eq!(a_stoch.output.finish_reason, FinishReason::MaxTokens);
    assert_eq!(a_stoch.output.tokens.len(), 48);
    assert!(
        a_stoch
            .output
            .tokens
            .iter()
            .all(|&t| (0..vocab as i32).contains(&t) && !forbidden.contains(&(t as usize))),
        "stochastic run emitted forbidden or out-of-range token"
    );
    assert_eq!(
        am_stoch.as_ref().unwrap().accepted,
        a_stoch.output.tokens,
        "constraint accepted trail must match emitted stochastic tokens exactly"
    );
    assert!(a_stoch.report.adaptive_speculative.is_some());
}

#[test]
fn explicit_off_and_explicit_proposer_are_authoritative_and_auto_stayed_plain_reports_reason() {
    let vocab = 32;
    let prompt: Vec<i32> = (0..8).collect();
    let seq: Vec<i32> = (8..48).map(|x| x % vocab as i32).collect();
    let config = greedy_config(36);

    // 1) Explicit Off.
    let clock = Rc::new(FakeClock::default());
    let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    let off = with_decode_clock(Some(clock), || {
        generate_speculative(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                speculative_mode: Speculative::Off,
                plain_reason: Some("speculative: disabled by request".into()),
                ..Default::default()
            },
        )
    })
    .unwrap();
    let off_adaptive = off
        .report
        .adaptive_speculative
        .expect("Off emits AdaptiveSpeculativeReport");
    assert!(!off_adaptive.adaptive);
    assert_eq!(off_adaptive.initial_proposer, ProposerKind::None);
    assert_eq!(off_adaptive.final_proposer, ProposerKind::None);
    assert_eq!(off_adaptive.depth_changes, 0);
    assert_eq!(off_adaptive.proposer_switches, 0);
    assert_eq!(
        off_adaptive.plain_reason.as_deref(),
        Some("speculative: disabled by request")
    );

    // 2) Explicit Proposer { PromptLookup, depth: 3 } with 0% accuracy & expensive verify.
    let clock = Rc::new(FakeClock::default());
    let mut target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    target.expensive_after_step = Some(0);
    target.expensive_verify = Duration::from_millis(25);
    let mut wrong = MockPrimaryProposer::new(
        ProposerKind::PromptLookup,
        vocab,
        prompt.len(),
        seq.clone(),
        clock.clone(),
        Duration::from_millis(5),
        vec![],
    );
    let explicit = with_decode_clock(Some(clock), || {
        generate_speculative(
            &target,
            &mut wrong,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                pipelining: Pipelining::Auto,
                speculative_mode: Speculative::proposer(SpeculativeProposer::PromptLookup, 3),
                ..Default::default()
            },
        )
    })
    .unwrap();
    assert_eq!(explicit.output.tokens, seq[..36]);
    assert_eq!(explicit.report.speculative_demoted_at, None);
    let exp_adaptive = explicit
        .report
        .adaptive_speculative
        .expect("explicit proposer emits AdaptiveSpeculativeReport");
    assert!(!exp_adaptive.adaptive);
    assert_eq!(exp_adaptive.initial_proposer, ProposerKind::PromptLookup);
    assert_eq!(exp_adaptive.final_proposer, ProposerKind::PromptLookup);
    assert_eq!(exp_adaptive.initial_depth, 3);
    assert_eq!(exp_adaptive.final_depth, 3);
    assert_eq!(exp_adaptive.depth_changes, 0);
    assert_eq!(exp_adaptive.proposer_switches, 0);
    assert!(wrong.propose_depths.iter().take(30).all(|&d| d == 3));

    // 3) Auto with no eligible speculative proposer: stays plain with StayedPlain event + reason.
    let clock = Rc::new(FakeClock::default());
    let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    let plain_plan = AutoSpeculativePlan {
        candidates: vec![AutoCandidate::plain()],
        fallbacks: vec!["speculative: no eligible proposer advertised".into()],
    };
    let stayed = with_decode_clock(Some(clock), || {
        generate_speculative(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            EngineOptions {
                speculative_mode: Speculative::Auto,
                adaptive_plan: Some(plain_plan),
                plain_reason: Some("speculative: no eligible proposer advertised".into()),
                ..Default::default()
            },
        )
    })
    .unwrap();
    let stayed_adaptive = stayed
        .report
        .adaptive_speculative
        .expect("Auto stayed-plain emits AdaptiveSpeculativeReport");
    assert!(stayed_adaptive.adaptive);
    assert_eq!(stayed_adaptive.final_proposer, ProposerKind::None);
    assert_eq!(stayed_adaptive.final_depth, 0);
    assert!(stayed_adaptive
        .events
        .iter()
        .any(|e| e.kind == AdaptiveEventKind::StayedPlain));
    assert_eq!(
        stayed_adaptive.plain_reason.as_deref(),
        Some("speculative: no eligible proposer advertised")
    );
}

#[test]
fn adaptive_switch_to_prompt_lookup_runs_linear_when_primary_does_not_build_trees_under_tree_rows()
{
    let vocab = 128;
    // Prompt contains a repeated prefix (`10, 11, 12`) with two distinct continuations
    // (`13, 14, 15` and `20, 21, 22`) so `NgramProposer::propose_tree` would build a branching
    // tree if called, while linear `propose` stays within provisioned `k`.
    let prompt: Vec<i32> = vec![
        10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12, 20, 21, 22, 23, 24, 10, 11, 12, 13, 14, 15,
    ];
    let mut seq = Vec::new();
    for i in 0..32 {
        seq.push(40 + (i % 70));
    }
    let repeated_branch = [
        10, 11, 12, 13, 14, 15, 16, 17, 10, 11, 12, 20, 21, 22, 23, 24,
    ];
    for i in 0..96 {
        seq.push(repeated_branch[i % repeated_branch.len()]);
    }
    let config = greedy_config(seq.len());

    for primary_kind in [ProposerKind::Mtp, ProposerKind::DraftModel] {
        let clock = Rc::new(FakeClock::default());
        let mut target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
        target.supports_tree = true;
        let mut proposer = MockPrimaryProposer::new(
            primary_kind,
            vocab,
            prompt.len(),
            seq.clone(),
            clock.clone(),
            Duration::from_millis(8),
            std::iter::once(0..32).collect(),
        );
        let plan = AutoSpeculativePlan {
            candidates: vec![
                AutoCandidate {
                    proposer: primary_kind,
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

        let run = with_decode_clock(Some(clock), || {
            generate_speculative(
                &target,
                &mut proposer,
                SpeculativePrompt::Tokens(&prompt),
                &config,
                2,
                &CancelFlag::new(),
                &mut |_| {},
                EngineOptions {
                    pipelining: Pipelining::Auto,
                    speculative_mode: Speculative::Auto,
                    tree_rows: 16,
                    adaptive_plan: Some(plan),
                    provisioned_max_depth: Some(6),
                    ..Default::default()
                },
            )
        })
        .unwrap();

        assert_eq!(
            run.output.tokens, seq,
            "exact token parity for primary {:?}",
            primary_kind
        );
        assert_eq!(
            target.tree_forwards.get(),
            0,
            "non-tree primary {:?} must never call forward_tree after switching to PromptLookup",
            primary_kind
        );
        assert_eq!(
            run.stats.tree.as_ref().map(|t| t.tree_passes),
            Some(0),
            "every verify pass must remain linear for {:?}",
            primary_kind
        );
        let adaptive = run
            .report
            .adaptive_speculative
            .expect("adaptive report present");
        assert!(
            adaptive.events.iter().any(|e| {
                e.kind == AdaptiveEventKind::ProposerSwitched
                    && e.to_proposer == ProposerKind::PromptLookup
            }),
            "expected switch from {:?} to PromptLookup: {:?}",
            primary_kind,
            adaptive.events
        );
    }
}

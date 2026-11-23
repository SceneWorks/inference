//! Pure-mock adaptive speculative engine tests (feature SW-F357, story SW-7237).
//!
//! Zero learned model weights or NN `.forward()` calls — exercises [`generate_speculative_adaptive`]
//! with a deterministic mock [`StepModel`], mock [`Proposer`], and [`StepClock`] to verify:
//! - AC1: Cost-aware depth adaptation within advertised/provisioned bounds, bounded mid-request
//!   proposer switching (`Mtp <-> PromptLookup -> Plain`), catch-up continuity, and anti-oscillation.
//! - AC2: Exact greedy parity and exact stochastic acceptance through rejection, penalties,
//!   rewindable constraints, stop tokens, caller stop, cancellation, and prefix-cache resume.
//! - AC3: Structured [`AdaptiveSpeculativeReport`] telemetry for adaptive runs, stayed-plain runs,
//!   and authoritative explicit `Off` / `{proposer, depth}` requests.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::Duration;

use candle_core::{Device, Tensor};
use core_llm::{
    with_decode_clock, AdaptiveEventKind, AutoCandidate, AutoSpeculativePlan, DecodeClock,
    ProposerKind, Speculative, SpeculativeProposer,
};

use super::*;
use crate::decode::StepOutput;
use crate::primitives::decode_cache::CacheMemory;
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
    capacity: usize,
    provisioned_overshoot: usize,
}

impl DecodeCache for MockCache {
    fn len(&self) -> i32 {
        self.len
    }

    fn rollback_to(&mut self, n: i32) -> Result<()> {
        assert!(
            n >= 0 && n <= self.len,
            "invalid rollback {n} > {}",
            self.len
        );
        self.len = n;
        Ok(())
    }

    fn reset(&mut self) {
        self.len = 0;
    }

    fn memory(&self) -> CacheMemory {
        CacheMemory::default()
    }
}

/// Deterministic mock target: for each cache position `p`, emits a full logit vector with a sharp
/// peak at `token_at(p)` plus deterministic secondary logits so penalties, constraints, and
/// stochastic sampling all have rich competing candidates.
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
    supports_tree: bool,
    tree_forwards: Cell<u64>,
    step_shapes: RefCell<Vec<(usize, bool)>>,
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
            expensive_verify: Duration::from_millis(22),
            cancel_on_forward: None,
            forwards: Cell::new(0),
            supports_tree: false,
            tree_forwards: Cell::new(0),
            step_shapes: RefCell::new(Vec::new()),
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

impl StepModel for ScriptedTarget {
    type Cache = MockCache;

    fn new_cache(&self) -> MockCache {
        MockCache {
            len: 0,
            capacity: 0,
            provisioned_overshoot: 0,
        }
    }

    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<MockCache> {
        Ok(MockCache {
            len: 0,
            capacity,
            provisioned_overshoot: overshoot,
        })
    }

    fn tree_unsupported(&self, _cache: &MockCache) -> Option<&'static str> {
        if self.supports_tree {
            None
        } else {
            Some(crate::decode::step::TREE_MODEL_UNSUPPORTED)
        }
    }

    fn forward_tree(
        &self,
        _cache: &mut MockCache,
        _tree: &TreeVerify<'_>,
    ) -> Result<crate::decode::step::TreeOutput> {
        self.tree_forwards.set(self.tree_forwards.get() + 1);
        Err(Error::Msg(
            "unexpected forward_tree on non-tree-building primary".into(),
        ))
    }

    fn device(&self) -> &Device {
        static CPU: Device = Device::Cpu;
        &CPU
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }

    fn forward_step(&self, cache: &mut MockCache, request: StepRequest<'_>) -> Result<StepOutput> {
        let call = self.forwards.get() + 1;
        self.forwards.set(call);
        let n = request.len()?;
        let start = cache.len as usize;
        if cache.capacity > 0 {
            let max_len = cache.capacity + cache.provisioned_overshoot;
            assert!(
                start + n <= max_len,
                "kv capacity exceeded: {} + {} > {} (cap={}, overshoot={})",
                start,
                n,
                max_len,
                cache.capacity,
                cache.provisioned_overshoot
            );
        }
        if !request.prefill {
            self.step_shapes.borrow_mut().push((n, request.want_hidden));
        }
        if !request.prefill && n > 1 {
            if cache.provisioned_overshoot > 0 {
                assert!(
                    n - 1 <= cache.provisioned_overshoot,
                    "verify drafts {} exceeded provisioned overshoot {}",
                    n - 1,
                    cache.provisioned_overshoot
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
        } else if !request.prefill {
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
        let all = Tensor::from_vec(rows, (1, n, self.vocab), &Device::Cpu)?;
        let logits = match request.scope {
            LogitsScope::All => all,
            LogitsScope::Last => all.narrow(1, n - 1, 1)?.reshape((1, self.vocab))?,
        };
        let hidden = if request.want_hidden {
            let h: Vec<f32> = (0..n).map(|i| (start + i) as f32).collect();
            Some(Tensor::from_vec(h, (1, n, 1), &Device::Cpu)?)
        } else {
            None
        };
        Ok(StepOutput { logits, hidden })
    }
}

/// Mock primary GPU proposer (e.g. `Mtp`) with controllable proposal cost, per-step match
/// schedule, optional stochastic proposal distributions, and catch-up / depth telemetry.
struct MockPrimaryProposer {
    kind: ProposerKind,
    vocab: usize,
    prompt_len: usize,
    sequence: Vec<i32>,
    clock: Rc<FakeClock>,
    propose_cost: Duration,
    catch_up_cost: Duration,
    /// Steps where this proposer predicts `sequence` accurately; outside these ranges it emits
    /// wrong tokens so rejection / shrinking / demotion are exercised.
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
            catch_up_cost: Duration::ZERO,
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

impl Proposer for MockPrimaryProposer {
    fn kind(&self) -> ProposerKind {
        self.kind
    }

    fn wants_hidden(&self) -> bool {
        matches!(self.kind, ProposerKind::Mtp | ProposerKind::BlockDraft)
    }

    fn warm(&mut self, _prompt: &[i32], _hidden: Option<&Tensor>) -> Result<()> {
        Ok(())
    }

    fn propose(
        &mut self,
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
        Ok(Proposal {
            drafts: Some(Drafts::Host(drafts)),
            dists,
        })
    }

    fn commit(
        &mut self,
        _cur: i32,
        _accepted: &[i32],
        _hidden: Option<&Tensor>,
        _position: i32,
    ) -> Result<()> {
        self.commits += 1;
        Ok(())
    }

    fn catch_up(
        &mut self,
        tokens: &[i32],
        preceding_hidden: Option<&Tensor>,
        position: i32,
    ) -> Result<()> {
        let rows = preceding_hidden.map(|h| h.dim(1)).transpose()?;
        if let Some(h) = preceding_hidden {
            let vals: Vec<f32> = h.flatten_all()?.to_vec1()?;
            // Each preceding hidden row belongs to the position immediately before each caught-up token.
            for (i, &val) in vals.iter().enumerate() {
                assert_eq!(
                    val as i32,
                    position - 1 + i as i32,
                    "catch_up hidden row mismatch at offset {i}"
                );
            }
        }
        self.clock.advance(self.catch_up_cost);
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
fn adaptive_resizes_depth_and_switches_between_primary_prompt_lookup_and_plain() {
    let vocab = 128;
    // Unique 16-token prompt so repeated copies trigger strong n-gram lookup matches.
    let prompt: Vec<i32> = (10..26).collect();
    let mut seq = Vec::new();
    // Phase 1 (0..75): open-ended non-repeating tokens; MTP accurate -> completes 16-step window
    // and grows depth 2 -> 3+ within max_depth = 5.
    for i in 0..75 {
        seq.push(30 + (i % 90));
    }
    // Phase 2 (75..155): repeated prompt spans; MTP lags on verbatim spans while prompt_lookup
    // matches with 0ms proposal cost -> bounded probe switches MTP -> PromptLookup!
    for i in 0..80 {
        seq.push(prompt[i % prompt.len()]);
    }
    // Phase 3 (155..215): returns to non-repeating open-ended tokens; PromptLookup dries up,
    // MTP is caught up cleanly and switches back PromptLookup -> MTP!
    for i in 0..60 {
        seq.push(30 + ((i * 7 + 13) % 90));
    }
    // Phase 4 (215..300): non-repeating tokens where MTP stops matching on expensive verify ->
    // shrinks depth & demotes to Plain.
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
        None,
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
        generate_speculative_adaptive(
            &target,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            None,
            Speculative::Auto,
            0,
            AdaptiveEngineConfig {
                adaptive_plan: Some(mtp_and_lookup_plan(2, 5)),
                provisioned_max_depth: Some(6),
                plain_reason: None,
            },
        )
    })
    .unwrap();

    // Token-for-token exact match with plain decode.
    assert_eq!(run.output.tokens, expected.output.tokens);
    assert_eq!(run.output.finish_reason, FinishReason::MaxTokens);

    // Every depth used by primary stayed strictly within advertised max_depth = 5.
    assert!(
        proposer
            .propose_depths
            .iter()
            .all(|&d| (1..=5).contains(&d)),
        "depths must stay in 1..=5: {:?}",
        proposer.propose_depths
    );
    assert!(
        proposer.propose_depths.iter().any(|&d| d > 2),
        "high-acceptance open-ended phase should grow depth above 2: {:?}",
        proposer.propose_depths
    );
    // Catch-up ran both after the initial plain probe and when switching back from PromptLookup -> MTP.
    assert!(
        proposer.catch_up_calls.len() >= 2,
        "expected catch-up after plain probe and after PromptLookup switch-back: {:?}",
        proposer.catch_up_calls
    );

    let report = run.record.report(false);
    let adaptive = report
        .adaptive_speculative
        .as_ref()
        .expect("DecodeReport must carry adaptive_speculative telemetry");
    assert!(adaptive.adaptive);
    assert_eq!(adaptive.initial_proposer, ProposerKind::Mtp);
    assert_eq!(adaptive.initial_depth, 2);
    assert!(
        adaptive.depth_changes >= 1,
        "expected depth adaptation: {adaptive:?}"
    );
    assert!(
        adaptive.proposer_switches >= 2,
        "expected MTP -> PromptLookup -> MTP / Plain switches: {adaptive:?}"
    );
    assert!(
        adaptive.events.iter().any(|e| matches!(
            e.kind,
            AdaptiveEventKind::DepthIncreased | AdaptiveEventKind::DepthDecreased
        )),
        "expected depth change event: {:?}",
        adaptive.events
    );
    assert!(
        adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProposerSwitched
                && e.to_proposer == ProposerKind::PromptLookup
        }),
        "expected switch to PromptLookup on repeated context: {:?}",
        adaptive.events
    );
    assert!(
        adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::Mtp
        }),
        "expected switch back to Mtp on open-ended phase: {:?}",
        adaptive.events
    );
    assert!(
        adaptive
            .events
            .iter()
            .any(|e| e.kind == AdaptiveEventKind::FallbackToPlain),
        "expected fallback to plain on losing phase: {:?}",
        adaptive.events
    );
    assert_eq!(adaptive.final_proposer, ProposerKind::None);
    assert_eq!(adaptive.final_depth, 0);
    assert!(report.speculative_demoted_at.is_some());
    assert!(report.speculative_monitor.is_some_and(|m| m.demoted));

    let cost = adaptive.cost.as_ref().expect("timed cost summary present");
    assert!(cost.proposal_ns > 0);
    assert!(cost.verify_ns > 0);
    assert!(cost.total_ns > 0);
    assert!(cost.host_syncs > 0);
    assert!(cost.ms_per_committed_token().is_some_and(|ms| ms > 0.0));
}

#[test]
fn adaptive_preserves_greedy_and_stochastic_semantics_through_penalties_constraints_stops_cancel_and_prefilled(
) {
    let vocab = 32;
    let prompt: Vec<i32> = (0..10).collect();
    // Mix open-ended and repeated prompt tokens so both depth resize and switching are exercised.
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
        // 1) Plain reference run.
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
                .forward_step(&mut ref_cache, StepRequest::last(&prompt).as_prefill())
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
                    warm_proposer: false,
                },
                config,
                0,
                &ref_cancel,
                &mut |ev| {
                    if matches!(ev, StreamEvent::Token { .. }) {
                        *ref_emitted.borrow_mut() += 1;
                    }
                },
                ref_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask),
            )
            .unwrap()
        } else {
            generate_speculative_with(
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
                ref_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask),
                stop_after_tokens.map(|_| &ref_stop as &dyn Fn() -> bool),
                None,
            )
            .unwrap()
        };

        // 2) Adaptive run with mixed accurate/wrong proposals so rejections & rollbacks happen.
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
        let mut cache = target
            .new_cache_for(prompt.len() + config.max_new_tokens + 6, 6)
            .unwrap();
        let adaptive_run = with_decode_clock(Some(clock), || {
            if prefilled {
                let prefill = target
                    .forward_step(
                        &mut cache,
                        StepRequest::last(&prompt).as_prefill().with_hidden(true),
                    )
                    .unwrap();
                generate_speculative_adaptive(
                    &target,
                    &mut proposer,
                    SpeculativePrompt::Prefilled {
                        cache: &mut cache,
                        logits: prefill.logits,
                        hidden: prefill.hidden,
                        history: &prompt,
                        position_delta: 0,
                        warm_proposer: true,
                    },
                    config,
                    2,
                    &cancel,
                    &mut |ev| {
                        if matches!(ev, StreamEvent::Token { .. }) {
                            *emitted.borrow_mut() += 1;
                        }
                    },
                    mask.as_mut()
                        .map(|m| m as &mut dyn RewindableConstraintMask),
                    None,
                    None,
                    Speculative::Auto,
                    0,
                    AdaptiveEngineConfig {
                        adaptive_plan: Some(mtp_and_lookup_plan(2, 4)),
                        provisioned_max_depth: Some(6),
                        plain_reason: None,
                    },
                )
            } else {
                generate_speculative_adaptive(
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
                    mask.as_mut()
                        .map(|m| m as &mut dyn RewindableConstraintMask),
                    stop_after_tokens.map(|_| &stop_fn as &dyn Fn() -> bool),
                    None,
                    Speculative::Auto,
                    0,
                    AdaptiveEngineConfig {
                        adaptive_plan: Some(mtp_and_lookup_plan(2, 4)),
                        provisioned_max_depth: Some(6),
                        plain_reason: None,
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

    // Case G: Stochastic sampling + penalties + rewindable constraint + stop token.
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
    assert!(
        a_stoch.record.report(false).adaptive_speculative.is_some(),
        "stochastic adaptive run must emit adaptive telemetry"
    );
}

#[test]
fn explicit_off_and_explicit_proposer_are_authoritative_and_auto_stayed_plain_reports_reason() {
    let vocab = 32;
    let prompt: Vec<i32> = (0..8).collect();
    let seq: Vec<i32> = (8..48).map(|x| x % vocab as i32).collect();
    let config = greedy_config(36);

    // 1) Explicit Off: non-adaptive, 0 depth, plain reason recorded.
    let clock = Rc::new(FakeClock::default());
    let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    let off = with_decode_clock(Some(clock), || {
        generate_speculative_adaptive(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            None,
            Speculative::Off,
            0,
            AdaptiveEngineConfig {
                adaptive_plan: None,
                provisioned_max_depth: None,
                plain_reason: Some("speculative: disabled by request".into()),
            },
        )
    })
    .unwrap();
    let off_adaptive = off
        .record
        .report(false)
        .adaptive_speculative
        .expect("Off still emits AdaptiveSpeculativeReport");
    assert!(!off_adaptive.adaptive);
    assert_eq!(off_adaptive.initial_proposer, ProposerKind::None);
    assert_eq!(off_adaptive.final_proposer, ProposerKind::None);
    assert_eq!(off_adaptive.depth_changes, 0);
    assert_eq!(off_adaptive.proposer_switches, 0);
    assert_eq!(
        off_adaptive.plain_reason.as_deref(),
        Some("speculative: disabled by request")
    );

    // 2) Explicit Proposer { PromptLookup, depth: 3 } with 0% accuracy & expensive verify:
    //    never resizes, never switches, never demotes.
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
        vec![], // 0% accuracy!
    );
    let explicit = with_decode_clock(Some(clock), || {
        generate_speculative_adaptive(
            &target,
            &mut wrong,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            None,
            Speculative::proposer(SpeculativeProposer::PromptLookup, 3),
            0,
            AdaptiveEngineConfig::default(),
        )
    })
    .unwrap();
    assert_eq!(explicit.output.tokens, seq[..36]);
    assert_eq!(explicit.record.speculative_demoted_at, None);
    let exp_adaptive = explicit
        .record
        .report(false)
        .adaptive_speculative
        .expect("explicit proposer emits AdaptiveSpeculativeReport");
    assert!(!exp_adaptive.adaptive);
    assert_eq!(exp_adaptive.initial_proposer, ProposerKind::PromptLookup);
    assert_eq!(exp_adaptive.final_proposer, ProposerKind::PromptLookup);
    assert_eq!(exp_adaptive.initial_depth, 3);
    assert_eq!(exp_adaptive.final_depth, 3);
    assert_eq!(exp_adaptive.depth_changes, 0);
    assert_eq!(exp_adaptive.proposer_switches, 0);
    // Every step except the budget-clamped tail proposed at exact depth 3.
    assert!(wrong.propose_depths.iter().take(30).all(|&d| d == 3));

    // 3) Auto with no eligible speculative proposer: stays plain with StayedPlain event + reason.
    let clock = Rc::new(FakeClock::default());
    let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    let plain_plan = AutoSpeculativePlan {
        candidates: vec![AutoCandidate::plain()],
        fallbacks: vec!["speculative: no eligible proposer advertised".into()],
    };
    let stayed = with_decode_clock(Some(clock), || {
        generate_speculative_adaptive(
            &target,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            None,
            Speculative::Auto,
            0,
            AdaptiveEngineConfig {
                adaptive_plan: Some(plain_plan),
                provisioned_max_depth: None,
                plain_reason: Some("speculative: no eligible proposer advertised".into()),
            },
        )
    })
    .unwrap();
    let stayed_adaptive = stayed
        .record
        .report(false)
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
    // Phase 1: open-ended non-repeating tokens under primary (`Mtp` / `DraftModel`).
    for i in 0..32 {
        seq.push(40 + (i % 70));
    }
    // Phase 2: repeated branching-prefix prompt spans so adaptive probes & switches to PromptLookup.
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
            generate_speculative_adaptive(
                &target,
                &mut proposer,
                SpeculativePrompt::Tokens(&prompt),
                &config,
                2,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                None,
                Speculative::Auto,
                16,
                AdaptiveEngineConfig {
                    adaptive_plan: Some(plan),
                    provisioned_max_depth: Some(6),
                    plain_reason: None,
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
            .record
            .report(false)
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

#[test]
fn adaptive_keeps_want_hidden_for_multitoken_lookup_after_mtp_retires() {
    let vocab = 256;
    let prompt: Vec<i32> = (10..26).collect();
    let mut seq = Vec::new();
    // Phase 1: open-ended tokens under MTP.
    for i in 0..24 {
        seq.push(30 + (i % 60));
    }
    // Phase 2: long repeated prompt span (> MAX_MTP_CATCHUP_TOKENS) so adaptive switches to
    // PromptLookup and MTP retires while PromptLookup continues running multi-token verifies.
    for i in 0..(MAX_MTP_CATCHUP_TOKENS + 80) {
        seq.push(prompt[i % prompt.len()]);
    }
    // Phase 3: each `10` matches prompt `10 -> 11, 12, 13, 14`, but the next token is a unique
    // mismatch (`100 + i`), so PromptLookup gets 0 accepted drafts on expensive verifies and
    // demotes cleanly to Plain.
    let demote_start = seq.len();
    for i in 0..60 {
        seq.push(10);
        seq.push(100 + i);
    }

    let config = greedy_config(seq.len());
    let clock = Rc::new(FakeClock::default());
    let mut target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
    target.expensive_after_step = Some(demote_start);
    target.expensive_verify = Duration::from_millis(24);
    let mut proposer = MockPrimaryProposer::new(
        ProposerKind::Mtp,
        vocab,
        prompt.len(),
        seq.clone(),
        clock.clone(),
        Duration::from_millis(8),
        std::iter::once(0..24).collect(),
    );

    let run = with_decode_clock(Some(clock), || {
        generate_speculative_adaptive(
            &target,
            &mut proposer,
            SpeculativePrompt::Tokens(&prompt),
            &config,
            2,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
            None,
            Speculative::Auto,
            0,
            AdaptiveEngineConfig {
                adaptive_plan: Some(mtp_and_lookup_plan(2, 4)),
                provisioned_max_depth: Some(6),
                plain_reason: None,
            },
        )
    })
    .unwrap();

    assert_eq!(run.output.tokens, seq);
    let adaptive = run
        .record
        .report(false)
        .adaptive_speculative
        .expect("adaptive report present");
    assert!(adaptive.events.iter().any(|e| {
        e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::PromptLookup
    }));
    assert!(adaptive
        .events
        .iter()
        .any(|e| e.kind == AdaptiveEventKind::FallbackToPlain));

    let shapes = target.step_shapes.borrow();
    assert!(
        shapes.iter().all(|&(n, want_hidden)| n == 1 || want_hidden),
        "no multi-token verify step may drop want_hidden on an MTP primary route: {:?}",
        *shapes
    );
    assert!(
        shapes.iter().any(|&(n, want_hidden)| n == 1 && !want_hidden),
        "demoted plain steps must turn want_hidden off (priced by graph_demoted_step_admission_bytes): {:?}",
        *shapes
    );
}

#[test]
fn adaptive_charges_mtp_catch_up_to_switch_cost_and_summary() {
    let vocab = 128;
    let prompt: Vec<i32> = (10..26).collect();
    let mut seq = Vec::new();
    // Phase 1 (0..45): open-ended; MTP warms depth 2.
    for i in 0..45 {
        seq.push(30 + (i % 90));
    }
    // Phase 2 (45..125): repeated prompt spans; MTP misses while PromptLookup wins probe and
    // buffers tokens for MTP catch-up.
    for i in 0..80 {
        seq.push(prompt[i % prompt.len()]);
    }
    // Phase 3 (125..195): open-ended; PromptLookup dries up and triggers MTP switch-back probe.
    for i in 0..70 {
        seq.push(30 + ((i * 7 + 13) % 90));
    }
    let config = greedy_config(seq.len());

    #[derive(Clone, Copy)]
    enum CatchUpTier {
        Zero,
        Small,
        Expensive,
    }

    let run_tier = |tier: CatchUpTier| {
        let clock = Rc::new(FakeClock::default());
        let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
        let mut proposer = MockPrimaryProposer::new(
            ProposerKind::Mtp,
            vocab,
            prompt.len(),
            seq.clone(),
            clock.clone(),
            Duration::from_millis(6),
            vec![0..45, 125..195],
        );
        proposer.catch_up_cost = match tier {
            CatchUpTier::Zero => Duration::ZERO,
            CatchUpTier::Small => Duration::from_millis(5),
            CatchUpTier::Expensive => Duration::from_millis(250),
        };
        let run = with_decode_clock(Some(clock), || {
            generate_speculative_adaptive(
                &target,
                &mut proposer,
                SpeculativePrompt::Tokens(&prompt),
                &config,
                2,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                None,
                Speculative::Auto,
                0,
                AdaptiveEngineConfig {
                    adaptive_plan: Some(mtp_and_lookup_plan(2, 5)),
                    provisioned_max_depth: Some(6),
                    plain_reason: None,
                },
            )
        })
        .unwrap();
        assert_eq!(run.output.tokens, seq);
        (run, proposer.catch_up_calls.len() as u64)
    };

    let (zero_run, zero_catchups) = run_tier(CatchUpTier::Zero);
    let (small_run, small_catchups) = run_tier(CatchUpTier::Small);
    let (dear_run, dear_catchups) = run_tier(CatchUpTier::Expensive);

    assert!(zero_catchups >= 2, "expected plain + switch-back catch-up");
    assert_eq!(small_catchups, zero_catchups);
    assert!(dear_catchups >= 2, "expected plain + switch-back catch-up");

    let zero_report = zero_run.record.report(false);
    let small_report = small_run.record.report(false);
    let dear_report = dear_run.record.report(false);
    let zero_adaptive = zero_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");
    let small_adaptive = small_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");
    let dear_adaptive = dear_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");

    // Cheap catch-up still switches back to MTP.
    assert!(small_adaptive.events.iter().any(|e| {
        e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::Mtp
    }));
    // Exact nanosecond accounting: every 5 ms catch-up appears in both `proposal_ns` and
    // `total_ns` (which feed `AdaptiveCostSummary::proposal_ms` and `total_ms`).
    let zero_cost = zero_adaptive.cost.expect("zero cost summary");
    let small_cost = small_adaptive.cost.expect("small cost summary");
    assert_eq!(
        small_cost.proposal_ns - zero_cost.proposal_ns,
        small_catchups * 5_000_000
    );
    assert_eq!(
        small_cost.total_ns - zero_cost.total_ns,
        small_catchups * 5_000_000
    );

    // Expensive catch-up is charged into both AdaptiveCostSummary and the switch-back probe,
    // causing the probe to revert rather than switching to MTP at a net loss.
    let dear_cost = dear_adaptive.cost.expect("dear cost summary");
    assert!(
        dear_cost.proposal_ns >= dear_catchups * 250_000_000,
        "catch-up work must be included in proposal_ns: {:?}",
        dear_cost
    );
    assert!(
        dear_cost.total_ns >= dear_cost.proposal_ns,
        "catch-up work must be included in total_ns: {:?}",
        dear_cost
    );
    assert!(
        dear_adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProbeReverted
                && e.from_proposer == ProposerKind::Mtp
                && e.to_proposer == ProposerKind::PromptLookup
                && e.reason.contains("catch-up")
        }),
        "expensive catch-up must revert the MTP switch-back probe: {:?}",
        dear_adaptive.events
    );
    assert!(
        !dear_adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProposerSwitched && e.to_proposer == ProposerKind::Mtp
        }),
        "expensive catch-up must not commit a switch back to MTP: {:?}",
        dear_adaptive.events
    );
}

#[test]
fn adaptive_charges_draft_model_catch_up_to_switch_cost_and_summary() {
    use crate::decode::proposers::DraftModelProposer;

    struct ScriptedDraft {
        vocab: usize,
        prompt_len: usize,
        sequence: Vec<i32>,
        clock: Rc<FakeClock>,
        step_cost: Duration,
        replay_cost: Duration,
        accurate_ranges: Vec<std::ops::Range<usize>>,
        replay_lens: RefCell<Vec<usize>>,
    }

    impl ScriptedDraft {
        fn is_accurate(&self, step_idx: usize) -> bool {
            self.accurate_ranges.iter().any(|r| r.contains(&step_idx))
        }

        fn draft_token(&self, step_idx: usize) -> i32 {
            let expected = self
                .sequence
                .get(step_idx)
                .copied()
                .unwrap_or(((step_idx + 1) % self.vocab) as i32);
            if self.is_accurate(step_idx) {
                expected
            } else {
                ((expected as usize + 5) % self.vocab) as i32
            }
        }
    }

    impl StepModel for ScriptedDraft {
        type Cache = MockCache;

        fn new_cache(&self) -> MockCache {
            MockCache {
                len: 0,
                capacity: 0,
                provisioned_overshoot: 0,
            }
        }

        fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<MockCache> {
            Ok(MockCache {
                len: 0,
                capacity,
                provisioned_overshoot: overshoot,
            })
        }

        fn device(&self) -> &Device {
            static CPU: Device = Device::Cpu;
            &CPU
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn forward_step(
            &self,
            cache: &mut MockCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            let n = request.len()?;
            let start = cache.len as usize;
            if !request.prefill {
                if n > 1 {
                    self.replay_lens.borrow_mut().push(n);
                    self.clock.advance(self.replay_cost);
                } else {
                    self.clock.advance(self.step_cost);
                }
            }
            cache.len += n as i32;
            let last_pos = start + n - 1;
            let step_idx = last_pos.saturating_sub(self.prompt_len.saturating_sub(1));
            let mut row = vec![-2.0f32; self.vocab];
            row[(self.draft_token(step_idx) as usize) % self.vocab] = 4.5;
            let logits = Tensor::from_vec(row, (1, self.vocab), &Device::Cpu)?;
            Ok(StepOutput {
                logits,
                hidden: None,
            })
        }
    }

    let vocab = 128;
    let prompt: Vec<i32> = (10..26).collect();
    let mut seq = Vec::new();
    // Phase 1 (0..45): open-ended; DraftModel matches accurately.
    for i in 0..45 {
        seq.push(30 + (i % 90));
    }
    // Phase 2 (45..125): repeated prompt spans; DraftModel misses while PromptLookup wins probe
    // and accumulates an inactive backlog for DraftModel catch-up.
    for i in 0..80 {
        seq.push(prompt[i % prompt.len()]);
    }
    // Phase 3 (125..195): open-ended; PromptLookup dries up and triggers DraftModel switch-back
    // probe whose catch-up replay runs inside `Proposer::catch_up` (`switch_overhead_ns`).
    for i in 0..70 {
        seq.push(30 + ((i * 7 + 13) % 90));
    }
    let config = greedy_config(seq.len());
    let plan = AutoSpeculativePlan {
        candidates: vec![
            AutoCandidate {
                proposer: ProposerKind::DraftModel,
                recommended_depth: 2,
                max_depth: 5,
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

    #[derive(Clone, Copy)]
    enum CatchUpTier {
        Zero,
        Small,
        Expensive,
    }

    let run_tier = |tier: CatchUpTier| {
        let clock = Rc::new(FakeClock::default());
        let target = ScriptedTarget::new(vocab, prompt.len(), seq.clone(), clock.clone());
        let draft = ScriptedDraft {
            vocab,
            prompt_len: prompt.len(),
            sequence: seq.clone(),
            clock: clock.clone(),
            step_cost: Duration::from_millis(2),
            replay_cost: match tier {
                CatchUpTier::Zero => Duration::ZERO,
                CatchUpTier::Small => Duration::from_millis(5),
                CatchUpTier::Expensive => Duration::from_millis(250),
            },
            accurate_ranges: vec![0..45, 125..195],
            replay_lens: RefCell::new(Vec::new()),
        };
        let mut proposer =
            DraftModelProposer::new(&draft, prompt.len() + config.max_new_tokens + 8, 6);
        let run = with_decode_clock(Some(clock), || {
            generate_speculative_adaptive(
                &target,
                &mut proposer,
                SpeculativePrompt::Tokens(&prompt),
                &config,
                2,
                &CancelFlag::new(),
                &mut |_| {},
                None,
                None,
                None,
                Speculative::Auto,
                0,
                AdaptiveEngineConfig {
                    adaptive_plan: Some(plan.clone()),
                    provisioned_max_depth: Some(6),
                    plain_reason: None,
                },
            )
        })
        .unwrap();
        assert_eq!(run.output.tokens, seq);
        let replays = draft.replay_lens.borrow().clone();
        (run, replays)
    };

    let (zero_run, zero_replays) = run_tier(CatchUpTier::Zero);
    let (small_run, small_replays) = run_tier(CatchUpTier::Small);
    let (dear_run, dear_replays) = run_tier(CatchUpTier::Expensive);

    assert!(
        zero_replays.len() >= 2 && zero_replays.iter().any(|&n| n > 2),
        "expected plain probe and switch-back replay catch-ups: {zero_replays:?}"
    );
    assert_eq!(small_replays, zero_replays);
    assert!(
        dear_replays.len() >= 2 && dear_replays.iter().any(|&n| n > 2),
        "expected plain probe and switch-back replay catch-ups: {dear_replays:?}"
    );

    let zero_report = zero_run.record.report(false);
    let small_report = small_run.record.report(false);
    let dear_report = dear_run.record.report(false);
    let zero_adaptive = zero_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");
    let small_adaptive = small_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");
    let dear_adaptive = dear_report
        .adaptive_speculative
        .as_ref()
        .expect("adaptive report");

    // Cheap catch-up still switches back to DraftModel.
    assert!(
        small_adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProposerSwitched
                && e.to_proposer == ProposerKind::DraftModel
        }),
        "expected cheap catch-up to switch back to DraftModel: {:?}",
        small_adaptive.events
    );
    let zero_cost = zero_adaptive.cost.expect("zero cost summary");
    let small_cost = small_adaptive.cost.expect("small cost summary");
    let small_count = small_replays.len() as u64;
    assert_eq!(
        small_cost.proposal_ns - zero_cost.proposal_ns,
        small_count * 5_000_000
    );
    assert_eq!(
        small_cost.total_ns - zero_cost.total_ns,
        small_count * 5_000_000
    );

    // Expensive catch-up is charged to switch_overhead_ns (not filtered out as a host stall by
    // Stretch::of), causing the switch-back probe to revert rather than switching at a net loss.
    let dear_cost = dear_adaptive.cost.expect("dear cost summary");
    let dear_count = dear_replays.len() as u64;
    assert!(
        dear_cost.proposal_ns >= dear_count * 250_000_000,
        "draft catch-up work must be included in proposal_ns: {:?}",
        dear_cost
    );
    assert!(
        dear_cost.total_ns >= dear_cost.proposal_ns,
        "draft catch-up work must be included in total_ns: {:?}",
        dear_cost
    );
    assert!(
        dear_adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProbeReverted
                && e.from_proposer == ProposerKind::DraftModel
                && e.to_proposer == ProposerKind::PromptLookup
                && e.reason.contains("catch-up")
        }),
        "expensive draft catch-up must revert the DraftModel switch-back probe: {:?}",
        dear_adaptive.events
    );
    assert!(
        !dear_adaptive.events.iter().any(|e| {
            e.kind == AdaptiveEventKind::ProposerSwitched
                && e.to_proposer == ProposerKind::DraftModel
        }),
        "expensive draft catch-up must not commit a switch back to DraftModel: {:?}",
        dear_adaptive.events
    );
}

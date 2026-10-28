//! `speculative.rs`'s unit tests, moved out of the production file as pure code motion (sc-27409).

use super::*;

/// A defaults-table row's recommended depths, spelled out so these tests do not move with the
/// table.
const DEPTHS: crate::RecommendedDepths = crate::RecommendedDepths {
    mtp: 3,
    prompt_lookup: 4,
    draft_model: 4,
};

/// E5: the shared advertisements recommend the depth of the defaults-table row they are handed
/// — a different row advertises a different depth (still clamped to the verify bound) — and
/// every backend's row is what the providers hand them.
#[test]
fn the_advertised_recommended_depth_is_the_rows() {
    let row = crate::RecommendedDepths {
        mtp: 2,
        prompt_lookup: 6,
        draft_model: 5,
    };
    assert_eq!(prompt_lookup_capabilities(8, &row).recommended_depth, 6);
    assert_eq!(prompt_lookup_capabilities(8, &DEPTHS).recommended_depth, 4);
    assert_eq!(draft_model_capabilities(8, &row).recommended_depth, 5);
    assert_eq!(draft_model_capabilities(8, &DEPTHS).recommended_depth, 4);
    assert_eq!(prompt_lookup_capabilities(3, &row).recommended_depth, 3);
    let mut caps = crate::TextLlmCapabilities {
        speculative: vec![prompt_lookup_capabilities(8, &row)],
        ..Default::default()
    };
    let mut fallbacks = Vec::new();
    settle_draft("/d", Ok(()), &mut caps, &mut fallbacks, &row);
    assert_eq!(
        caps.proposer(crate::SpeculativeProposer::DraftModel)
            .map(|c| c.recommended_depth),
        Some(5)
    );
}

// --- n-gram proposer ---

#[test]
fn ngram_proposes_continuation_of_recent_match() {
    // "1 2 3 1 2": last bigram "1 2" recurs at index 0, followed by "3" (then the trailing 1 2).
    let toks = [1, 2, 3, 1, 2];
    assert_eq!(ngram_propose(&toks, 3, 1), vec![3]);
    assert_eq!(ngram_propose(&toks, 3, 3), vec![3, 1, 2]);
}

#[test]
fn ngram_prefers_longer_match() {
    // Longer n-gram "x a b" recurs; its continuation is "y", not the "a b"->"z" of the bigram.
    let toks = [9, 1, 2, 4, 7, 9, 1, 2];
    // last trigram is "9 1 2", earlier at index 0 -> followed by "4".
    assert_eq!(ngram_propose(&toks, 3, 1), vec![4]);
}

#[test]
fn ngram_returns_empty_without_a_match() {
    assert_eq!(ngram_propose(&[1, 2, 3, 4], 3, 4), Vec::<i32>::new());
    assert_eq!(ngram_propose(&[1], 3, 4), Vec::<i32>::new());
    assert_eq!(ngram_propose(&[], 3, 4), Vec::<i32>::new());
}

// --- mode resolution and the greedy commit (sc-24130) ---

/// sc-24433 AC2: `auto` resolves to MTP where a head exists, else prompt lookup, else off
/// with a named reason; an explicit proposer runs as asked; nothing resolves silently.
#[test]
fn speculative_option_resolves_against_the_advertised_proposers() {
    use crate::{
        MtpCapabilities, MtpMode, ProposerCapabilities, Speculative, SpeculativeProposer,
        TextLlmCapabilities,
    };
    let lookup = ProposerCapabilities {
        proposer: SpeculativeProposer::PromptLookup,
        max_depth: 8,
        recommended_depth: 4,
    };
    let head = TextLlmCapabilities {
        mtp: Some(MtpCapabilities {
            max_draft_tokens: 5,
            recommended_draft_tokens: 3,
        }),
        speculative: vec![lookup],
        ..Default::default()
    };
    let no_head = TextLlmCapabilities {
        speculative: vec![lookup],
        ..Default::default()
    };
    let nothing = TextLlmCapabilities::default();
    let plan = |mode, caps: &TextLlmCapabilities| resolve_speculative(mode, caps);
    let run = |proposer, depth| SpeculativePlan::Run { proposer, depth };

    assert_eq!(
        plan(Speculative::Off, &head),
        SpeculativeResolution::default()
    );
    let auto = plan(Speculative::Auto, &head);
    assert_eq!(auto.plan, run(SpeculativeProposer::Mtp, 3));
    assert_eq!(auto.fallback, None);
    assert_eq!(auto.plan.proposer(), ProposerKind::Mtp);
    let auto = plan(Speculative::Auto, &no_head);
    assert_eq!(auto.plan, run(SpeculativeProposer::PromptLookup, 4));
    assert_eq!(auto.plan.proposer().label(), "prompt_lookup");
    assert_eq!(auto.plan.depth(), Some(4));
    let auto = plan(Speculative::Auto, &nothing);
    assert_eq!(auto.plan, SpeculativePlan::Off);
    assert_eq!(auto.plan.proposer(), ProposerKind::None);
    assert_eq!(
        auto.fallback.as_deref(),
        Some(
            "speculative: auto found no proposer this model can run (it advertises none; \
             decoded without a proposer)"
        )
    );

    // The legacy mode is the same option.
    assert_eq!(
        plan(MtpMode::Enabled { draft_tokens: 3 }.into(), &head).plan,
        run(SpeculativeProposer::Mtp, 3)
    );
    assert_eq!(
        plan(MtpMode::Auto.into(), &no_head).plan,
        auto_plan(&no_head)
    );

    let explicit = Speculative::proposer(SpeculativeProposer::PromptLookup, 2);
    assert_eq!(
        plan(explicit, &head).plan,
        run(SpeculativeProposer::PromptLookup, 2)
    );
    let unadvertised = plan(
        Speculative::proposer(SpeculativeProposer::DraftModel, 2),
        &head,
    );
    assert_eq!(unadvertised.plan, SpeculativePlan::Off);
    assert!(unadvertised
        .fallback
        .unwrap()
        .contains("`draft_model` is not available"));
    let too_deep = plan(
        Speculative::proposer(SpeculativeProposer::PromptLookup, 20),
        &head,
    );
    assert_eq!(too_deep.plan, run(SpeculativeProposer::PromptLookup, 8));
    assert!(too_deep.fallback.unwrap().contains("depth 20 clamped to 8"));

    assert_eq!(ProposerKind::default(), ProposerKind::None);
    assert_eq!(ProposerKind::PromptLookup.label(), "prompt_lookup");
    assert_eq!(ProposerKind::DraftModel.label(), "draft_model");
    assert_eq!(ProposerKind::Mtp.label(), "mtp");
    assert_eq!(
        ProposerKind::from(SpeculativeProposer::DraftModel),
        ProposerKind::DraftModel
    );
}

/// A family that advertises no proposer names its own reason for `auto` and for an explicit
/// proposer, with the same leading words the resolver uses; `off` names nothing (E2/E8).
#[test]
fn a_family_without_proposers_names_its_own_reason() {
    use crate::{Speculative, SpeculativeProposer};
    assert_eq!(
        no_proposer_fallback(Speculative::Off, CAPTIONER_NO_PROPOSER),
        None
    );
    assert_eq!(
        no_proposer_fallback(Speculative::Auto, CAPTIONER_NO_PROPOSER).as_deref(),
        Some(
            "speculative: auto found no proposer this model can run (this captioner / SVG \
             decoder advertises no proposer: its continuation decodes token-at-a-time after \
             a multimodal prefill)"
        )
    );
    let explicit = no_proposer_fallback(
        Speculative::proposer(SpeculativeProposer::Mtp, 2),
        CAPTIONER_NO_PROPOSER,
    )
    .unwrap();
    assert!(
        explicit.starts_with("speculative: `mtp` is not available for this model (")
            && explicit.contains(CAPTIONER_NO_PROPOSER),
        "{explicit}"
    );
}

/// E8: the shared prompt-lookup advertisement and verify bound — the recommendation never
/// exceeds the bound, and `draft_model` is advertised at the prompt-lookup bound.
#[test]
fn the_shared_advertisements_respect_the_verify_bound() {
    let depths = &crate::DecodeBackend::Mlx.defaults().recommended_depths;
    let deep = prompt_lookup_capabilities(7, depths);
    assert_eq!(
        (deep.max_depth, deep.recommended_depth),
        (7, depths.prompt_lookup)
    );
    let shallow = prompt_lookup_capabilities(2, depths);
    assert_eq!((shallow.max_depth, shallow.recommended_depth), (2, 2));
    let mut caps = crate::TextLlmCapabilities::default();
    assert_eq!(verify_depth_bound(&caps), 1, "no prompt lookup: one draft");
    caps.speculative.push(deep);
    assert_eq!(verify_depth_bound(&caps), 7);
}

/// E2 (sc-24436): a refused draft is named in both `LoadReport::draft` and
/// `LoadReport::fallbacks` and advertises nothing; a resident one advertises `draft_model` at
/// the verify bound and names no fallback.
#[test]
fn settling_a_draft_names_a_refusal_and_advertises_a_resident_draft() {
    use crate::SpeculativeProposer;
    let mut caps = crate::TextLlmCapabilities {
        speculative: vec![prompt_lookup_capabilities(5, &DEPTHS)],
        ..Default::default()
    };
    let mut fallbacks = Vec::new();
    let why = draft_load_refusal("no such file");
    assert_eq!(why, "draft model: its load failed (no such file)");
    let (kept, report): (Option<()>, _) =
        settle_draft("/d", Err(why.clone()), &mut caps, &mut fallbacks, &DEPTHS);
    assert!(kept.is_none());
    assert_eq!(caps.proposer(SpeculativeProposer::DraftModel), None);
    assert_eq!(report, crate::DraftReport::refused("/d", why.clone()));
    assert_eq!(fallbacks, vec![why.clone()]);
    // The load report's own entry point names an admission refusal the same way.
    let mut load = crate::LoadReport::default();
    load.record_draft(crate::DraftReport::refused("/d", why.clone()));
    assert_eq!(load.fallbacks, vec![why]);

    let mut fallbacks = Vec::new();
    let (kept, report) = settle_draft("/d", Ok(7u8), &mut caps, &mut fallbacks, &DEPTHS);
    assert_eq!(kept, Some(7));
    assert_eq!(
        caps.proposer(SpeculativeProposer::DraftModel),
        Some(draft_model_capabilities(5, &DEPTHS))
    );
    assert_eq!(report, crate::DraftReport::resident("/d"));
    assert!(fallbacks.is_empty());
    assert_eq!(
        draft_unpriced_refusal("x"),
        "draft model: its load cannot be priced (x)"
    );
    assert_eq!(draft_refusal("x"), "draft model: x");
}

fn word_tokenizer(prefix: &str, vocab: usize) -> crate::Tokenizer {
    let entries: Vec<String> = (0..vocab)
        .map(|i| format!("\"{prefix}{i}\": {i}"))
        .collect();
    crate::Tokenizer::from_json(&format!(
        r#"{{"version": "1.0", "added_tokens": [], "normalizer": null,
                "pre_tokenizer": {{ "type": "Whitespace" }}, "post_processor": null,
                "decoder": null,
                "model": {{ "type": "WordLevel", "vocab": {{ {} }}, "unk_token": "{prefix}0" }} }}"#,
        entries.join(", ")
    ))
    .unwrap()
}

#[test]
fn a_draft_must_share_the_vocabulary_and_the_logits_width() {
    use crate::{Speculative, SpeculativeProposer};
    let target = word_tokenizer("t", 8);
    assert_eq!(
        draft_compatibility(&target, 8, &word_tokenizer("t", 8), 8),
        Ok(8)
    );
    // A tokenizer of the same size over different tokens is refused by name.
    let why = draft_compatibility(&target, 8, &word_tokenizer("w", 8), 8).unwrap_err();
    assert!(
        why.starts_with("draft model: its tokenizer vocabulary is not the target's"),
        "{why}"
    );
    // A draft padded less than its target (Qwen2.5-0.5B's 151936 rows beside 7B's 152064) is
    // compatible; one padded past its tokenizer proposes only the tokenizer's ids.
    assert_eq!(
        draft_compatibility(&target, 16, &word_tokenizer("t", 8), 12),
        Ok(8)
    );
    assert_eq!(
        draft_compatibility(&target, 16, &word_tokenizer("t", 8), 8),
        Ok(8)
    );
    // A draft scoring more ids than its target is refused.
    let why = draft_compatibility(&target, 8, &word_tokenizer("t", 8), 16).unwrap_err();
    assert!(
        why.contains("scores 16 token ids, more than the target's 8"),
        "{why}"
    );

    let caps = draft_model_capabilities(8, &DEPTHS);
    assert_eq!(caps.proposer, SpeculativeProposer::DraftModel);
    assert_eq!((caps.max_depth, caps.recommended_depth), (8, 4));
    // The provider's bound is the advertisement's; the recommendation never exceeds it.
    let shallow = draft_model_capabilities(3, &DEPTHS);
    assert_eq!((shallow.max_depth, shallow.recommended_depth), (3, 3));
    // Advertised, `{proposer: draft_model}` runs at the requested depth.
    let resident = crate::TextLlmCapabilities {
        speculative: vec![caps],
        ..Default::default()
    };
    assert_eq!(
        resolve_speculative(
            Speculative::proposer(SpeculativeProposer::DraftModel, 3),
            &resident
        ),
        SpeculativeResolution {
            plan: SpeculativePlan::Run {
                proposer: SpeculativeProposer::DraftModel,
                depth: 3
            },
            fallback: None
        }
    );
}

/// sc-24436 E2: a `draft_model` request past the draft's context window runs what `auto`
/// resolves to, by name; within it (or with an unbounded draft) the plan is unchanged.
#[test]
fn a_request_past_the_draft_context_runs_auto_by_name() {
    use crate::{ProposerCapabilities, Speculative, SpeculativeProposer};
    let lookup = ProposerCapabilities {
        proposer: SpeculativeProposer::PromptLookup,
        max_depth: 8,
        recommended_depth: 4,
    };
    let caps = crate::TextLlmCapabilities {
        speculative: vec![lookup, draft_model_capabilities(8, &DEPTHS)],
        ..Default::default()
    };
    let asked = resolve_speculative(
        Speculative::proposer(SpeculativeProposer::DraftModel, 3),
        &caps,
    );
    // Depth 3: the draft cache holds the prompt, the budget and a step's 3 + 1 positions.
    assert_eq!(fit_draft_context(asked.clone(), &caps, 64, 40, 20), asked);
    assert_eq!(fit_draft_context(asked.clone(), &caps, 0, 4000, 24), asked);
    let fit = fit_draft_context(asked.clone(), &caps, 64, 40, 21);
    assert_eq!(
        fit.plan,
        SpeculativePlan::Run {
            proposer: SpeculativeProposer::PromptLookup,
            depth: 4
        }
    );
    let why = fit.fallback.unwrap();
    assert!(
        why.contains("exceeds the draft model's context window 64")
            && why.contains("`prompt_lookup` at depth 4"),
        "{why}"
    );
    // No proposer for `auto` either: plain decoding, both reasons named.
    let only_draft = crate::TextLlmCapabilities {
        speculative: vec![draft_model_capabilities(8, &DEPTHS)],
        ..Default::default()
    };
    let fit = fit_draft_context(asked, &only_draft, 64, 40, 21);
    assert_eq!(fit.plan, SpeculativePlan::Off);
    let why = fit.fallback.unwrap();
    assert!(
        why.contains("without a proposer") && why.contains("auto found no proposer"),
        "{why}"
    );
    // Any other plan is untouched.
    let lookup_plan = resolve_speculative(
        Speculative::proposer(SpeculativeProposer::PromptLookup, 3),
        &caps,
    );
    assert_eq!(
        fit_draft_context(lookup_plan.clone(), &caps, 64, 40, 25),
        lookup_plan
    );
}

/// sc-24438 AC1: a request above the advertised max depth — new option or legacy `mtp` —
/// runs at the max, and the clamp is named; a request within it is untouched.
#[test]
fn a_too_deep_request_is_clamped_to_the_advertised_max_and_named() {
    use crate::{MtpMode, Speculative, SpeculativeProposer, TextLlmCapabilities};
    let mut caps = TextLlmCapabilities::default();
    caps.advertise_mtp(7, 3);
    let legacy = resolve_speculative(MtpMode::Enabled { draft_tokens: 40 }.into(), &caps);
    assert_eq!(
        legacy.plan,
        SpeculativePlan::Run {
            proposer: SpeculativeProposer::Mtp,
            depth: 7
        }
    );
    assert_eq!(
        legacy.fallback.as_deref(),
        Some("speculative: `mtp` depth 40 clamped to 7 (advertised 1..=7)")
    );
    let within = resolve_speculative(Speculative::proposer(SpeculativeProposer::Mtp, 7), &caps);
    assert_eq!(within.plan.depth(), Some(7));
    assert_eq!(within.fallback, None);
}

fn auto_plan(caps: &crate::TextLlmCapabilities) -> SpeculativePlan {
    resolve_speculative(crate::Speculative::Auto, caps).plan
}

// --- auto's acceptance monitor (sc-24446) ---

/// E5: the thresholds are each plain loop's cheapest-measured break-evens — on MLX prompt
/// lookup 0.4 at any depth but only where the demoted request would continue on the pipelined
/// loop (the measured regime), MTP 0.25 at one draft growing 0.475 per extra draft (1.2 at the
/// recommended depth 3); on Candle (cuda-campaign-a) lookup 0.09 growing 0.04 per draft (0.21
/// at `auto`'s depth 4), MTP 0.25 growing 0.125 (0.5 at depth 3) — and nothing is monitored
/// without a measured cost (no proposer, a zero depth, `draft_model`, lookup falling back to
/// MLX's unpipelined loop).
#[test]
fn the_demotion_thresholds_are_the_measured_break_evens() {
    use PlainDecode::{Candle, MlxPipelined, MlxUnpipelined};
    let close = |a: Option<f64>, b: f64| (a.unwrap() - b).abs() < 1e-12;
    let lookup = |depth, plain| demotion_threshold(ProposerKind::PromptLookup, depth, plain);
    let mtp = |depth, plain| demotion_threshold(ProposerKind::Mtp, depth, plain);
    assert!(close(lookup(4, MlxPipelined), 0.4));
    assert!(close(lookup(1, MlxPipelined), 0.4));
    assert_eq!(lookup(4, MlxUnpipelined), None);
    assert!(close(lookup(1, Candle), 0.09));
    assert!(close(lookup(3, Candle), 0.17));
    assert!(close(lookup(PROMPT_LOOKUP_RECOMMENDED_DEPTH, Candle), 0.21));
    assert!(close(lookup(7, Candle), 0.33));
    for plain in [MlxPipelined, MlxUnpipelined] {
        assert!(close(mtp(1, plain), 0.25));
        assert!(close(mtp(3, plain), 1.2));
    }
    assert!(close(mtp(1, Candle), 0.25));
    assert!(close(mtp(MTP_RECOMMENDED_DEPTH, Candle), 0.5));
    assert!(close(mtp(7, Candle), 1.0));
    for plain in [MlxPipelined, MlxUnpipelined, Candle] {
        assert_eq!(mtp(0, plain), None);
        assert_eq!(lookup(0, plain), None);
        assert_eq!(demotion_threshold(ProposerKind::DraftModel, 4, plain), None);
        assert_eq!(demotion_threshold(ProposerKind::None, 4, plain), None);
    }
    // The measured families land on the documented sides. MLX: a companion head's open-ended
    // mal (0.66–0.85) demotes at depth 3, a native head's (1.38+) does not.
    let mlx3 = mtp(3, MlxPipelined).unwrap();
    assert!(0.85 < mlx3 && mlx3 < 1.38);
    // Candle: Qwen3.8's native head (mal 1.24+ at depth 3, +41..+127 %) is far above;
    // Candle lookup's measured open-ended losers (Bonsai, mal 0.15–0.20 at depth 4) demote,
    // and no measured Candle lookup winner (Qwen3.8 at depth 3, mal 0.40+) does.
    assert!(mtp(3, Candle).unwrap() < 1.24);
    let candle4 = lookup(4, Candle).unwrap();
    assert!(0.20 < candle4 && candle4 < 0.40);
    assert!(lookup(3, Candle).unwrap() < 0.40);
    // Each Candle threshold is at or below its depth's cheapest measured cost minus one.
    for (depth, floor) in [(1, 0.099), (3, 0.197), (7, 0.349)] {
        assert!(
            lookup(depth, Candle).unwrap() <= floor,
            "lookup depth {depth}"
        );
    }
    for (depth, floor) in [(1, 0.27), (3, 0.50), (7, 1.12)] {
        assert!(
            mtp(depth, Candle).unwrap() <= floor + 1e-12,
            "mtp depth {depth}"
        );
    }
}

/// The monitor judges rolling windows of [`ACCEPTANCE_PROBE_VERIFIES`] verify steps: the first
/// window below the threshold demotes (on exactly its last step, once); a window at or above
/// it keeps the proposer for the next window; after the demotion nothing fires again.
#[test]
fn the_monitor_demotes_at_the_first_window_below_the_threshold() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    // Below: 0 accepted every step demotes on the first window's last step, never before or
    // after.
    let mut low = AcceptanceMonitor::with_threshold(0.4).unwrap();
    let fired: Vec<bool> = (0..3 * window).map(|_| low.observe(0)).collect();
    assert_eq!(fired.iter().position(|&f| f), Some(window - 1));
    assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
    // Just below the break-even (6 / 16 = 0.375 < 0.4) still demotes.
    let mut under = AcceptanceMonitor::with_threshold(0.4).unwrap();
    let fired = (0..window).map(|i| under.observe(usize::from(i < 6)));
    assert!(fired.last().unwrap());
    // At it (0.4 × 16 = 6.4 → 7 accepted) a window keeps the proposer.
    let mut at = AcceptanceMonitor::with_threshold(0.4).unwrap();
    assert!(!(0..4 * window).any(|i| at.observe(usize::from(i % window < 7))));
    // Exactly at it (0.25 × 16 = 4 accepted, break-even: not losing) a window keeps it too.
    let mut even = AcceptanceMonitor::with_threshold(0.25).unwrap();
    assert!(!(0..4 * window).any(|i| even.observe(usize::from(i % window < 4))));
    // Consistently above it, never.
    let mut high = AcceptanceMonitor::with_threshold(1.2).unwrap();
    assert!(!(0..8 * window).any(|_| high.observe(3)));
    assert_eq!(AcceptanceMonitor::with_threshold(f64::NAN), None);
    assert_eq!(AcceptanceMonitor::with_threshold(-1.0), None);
}

/// A late loser: two windows well above the threshold, then acceptance collapses. The monitor
/// demotes at the end of the first failing window — the third — not at the first window, and
/// not later as a cumulative mean would (the request's mean is still 2.0 there, far above
/// 1.2). The earlier windows' credit is not carried: a window is judged on its own steps.
#[test]
fn an_early_high_then_low_request_demotes_at_its_first_failing_window() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let mut late = AcceptanceMonitor::with_threshold(1.2).unwrap();
    let accepted = |i: usize| if i < 2 * window { 3 } else { 0 };
    let fired: Vec<bool> = (0..6 * window).map(|i| late.observe(accepted(i))).collect();
    assert_eq!(fired.iter().position(|&f| f), Some(3 * window - 1));
    assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
    // A window straddling the collapse is judged on its own steps: 6 high steps (18 accepted
    // < 1.2 × 16 = 19.2) still demote, 7 (21) do not — and the next all-zero window does.
    for (high_steps, demoted_at) in [(6, 2 * window - 1), (7, 3 * window - 1)] {
        let mut straddle = AcceptanceMonitor::with_threshold(1.2).unwrap();
        let accepted = |i: usize| if i < window + high_steps { 3 } else { 0 };
        let fired: Vec<bool> = (0..4 * window)
            .map(|i| straddle.observe(accepted(i)))
            .collect();
        assert_eq!(
            fired.iter().position(|&f| f),
            Some(demoted_at),
            "{high_steps} high steps"
        );
    }
    // Low early, high later: still demoted at the first window — irreversible.
    let mut early = AcceptanceMonitor::with_threshold(1.2).unwrap();
    let fired: Vec<bool> = (0..4 * window)
        .map(|i| early.observe(if i < window { 0 } else { 3 }))
        .collect();
    assert_eq!(fired.iter().position(|&f| f), Some(window - 1));
    assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
}

/// Only `auto` is monitored: an explicit `{proposer, depth}` is the caller's choice and runs
/// as asked; `off`, no proposer and `draft_model` are never monitored. MLX-unpipelined lookup
/// (no static threshold) is monitored only when the engine times its steps.
#[test]
fn only_auto_is_monitored() {
    use crate::{Speculative, SpeculativeProposer};
    use PlainDecode::{Candle, MlxPipelined, MlxUnpipelined};
    let monitor = |mode, proposer, depth, plain, timed| {
        AcceptanceMonitor::for_request(mode, proposer, depth, plain, timed)
    };
    let auto = monitor(Speculative::Auto, ProposerKind::Mtp, 3, MlxPipelined, false).unwrap();
    assert!((auto.threshold().unwrap() - 1.2).abs() < 1e-12);
    let candle = monitor(
        Speculative::Auto,
        ProposerKind::PromptLookup,
        4,
        Candle,
        false,
    )
    .unwrap();
    assert!((candle.threshold().unwrap() - 0.21).abs() < 1e-12);
    let lookup = |plain, timed| {
        monitor(
            Speculative::Auto,
            ProposerKind::PromptLookup,
            4,
            plain,
            timed,
        )
    };
    assert!(lookup(MlxPipelined, false).is_some());
    assert_eq!(
        lookup(MlxUnpipelined, false),
        None,
        "no static threshold, untimed"
    );
    let timed = lookup(MlxUnpipelined, true).expect("its measured costs decide");
    assert_eq!(timed.threshold(), None);
    assert!(timed.probing());
    for plain in [MlxPipelined, MlxUnpipelined, Candle] {
        for timed in [false, true] {
            for explicit in [
                Speculative::proposer(SpeculativeProposer::PromptLookup, 4),
                Speculative::proposer(SpeculativeProposer::Mtp, 3),
                Speculative::Off,
            ] {
                for (kind, depth) in [(ProposerKind::PromptLookup, 4), (ProposerKind::Mtp, 3)] {
                    assert_eq!(
                        monitor(explicit, kind, depth, plain, timed),
                        None,
                        "{explicit:?} {kind:?} {plain:?}"
                    );
                }
            }
            for (kind, depth) in [
                (ProposerKind::None, 0),
                (ProposerKind::DraftModel, 4),
                (ProposerKind::Mtp, 0),
            ] {
                assert_eq!(
                    monitor(Speculative::Auto, kind, depth, plain, timed),
                    None,
                    "{kind:?} {depth}"
                );
            }
        }
    }
}

// --- the cost-aware (timed) monitor (sc-24446) ---

const MS: u64 = 1_000_000;

/// A timed `auto` monitor fed by a fake step clock: plain steps cost `plain_ns`, a step with
/// drafts `ratio × plain_ns`; `accepted(i)` drafts accepted at speculative step `i` of
/// `drafts` proposed. Returns the speculative step index (0-based, counted after the probe)
/// whose window demoted, and the monitor.
fn drive(
    monitor: AcceptanceMonitor,
    plain_ns: u64,
    ratio: f64,
    drafts: usize,
    steps: usize,
    accepted: impl Fn(usize) -> usize,
) -> (Option<usize>, AcceptanceMonitor) {
    drive_costs(monitor, plain_ns, drafts, steps, accepted, |_| ratio)
}

/// [`drive`] with speculative step `i` costing `ratio(i)` plain steps.
fn drive_costs(
    mut monitor: AcceptanceMonitor,
    plain_ns: u64,
    drafts: usize,
    steps: usize,
    accepted: impl Fn(usize) -> usize,
    ratio: impl Fn(usize) -> f64,
) -> (Option<usize>, AcceptanceMonitor) {
    let mut probes = 0;
    while monitor.probing() {
        assert!(!monitor.observe_step(plain_step(Some(plain_ns))));
        probes += 1;
    }
    assert_eq!(probes, PLAIN_PROBE_MAX_STEPS, "warm-up + timed probe steps");
    let demoted = (0..steps).find(|&i| {
        monitor.observe_step(StepObservation {
            accepted: accepted(i),
            drafts,
            elapsed: Some(Duration::from_nanos(
                (plain_ns as f64 * ratio(i)).round() as u64
            )),
        })
    });
    (demoted, monitor)
}

/// A single-token step of the fake clock's `ns` nanoseconds (`None`: untimed). The tests'
/// durations are a deterministic fake clock's, never the wall clock's.
fn plain_step(ns: Option<u64>) -> StepObservation {
    StepObservation {
        accepted: 0,
        drafts: 0,
        elapsed: ns.map(Duration::from_nanos),
    }
}

fn timed(proposer: ProposerKind, depth: u32, plain: PlainDecode) -> AcceptanceMonitor {
    AcceptanceMonitor::for_request(crate::Speculative::Auto, proposer, depth, plain, true).unwrap()
}

/// The decision the coordinator's evidence asks for: a Bonsai-like request (verify r = 2.4,
/// mal 0.75 — above Candle's static 0.5 at depth 3; gain 0.73) is demoted on its measured
/// gain at the end of its first window — its acceptance (3 drafts every fourth step) is too
/// lumpy for the clear-loss check's optimistic gain to call earlier; a Qwen3.8-like one (r = 1.5, mal 1.25) keeps its proposer for
/// every window — on Candle and on MLX's unpipelined path alike.
#[test]
fn a_dear_verify_demotes_and_a_cheap_one_keeps_at_the_same_scale_of_acceptance() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    for plain in [PlainDecode::Candle, PlainDecode::MlxUnpipelined] {
        // Bonsai-like: 3 accepted every 4th step (mal 0.75), each verify 2.4 plain steps.
        let (at, bonsai) = drive(
            timed(ProposerKind::Mtp, 3, plain),
            10 * MS,
            2.4,
            3,
            8 * window,
            |i| {
                if i % 4 == 3 {
                    3
                } else {
                    0
                }
            },
        );
        // The first three depth-3 steps are the shape's warm-up, then thirteen timed steps.
        assert_eq!(at, Some(window - 1), "{plain:?}");
        let d = bonsai.last_decision().unwrap();
        assert_eq!(
            (d.basis, d.demoted, d.window),
            (DemotionBasis::Measured, true, 1)
        );
        // Every fourth of the sixteen accepted 3.
        assert_eq!((d.verifies, d.accepted), (window as u32, 12));
        assert_eq!(d.plain_step_ns, Some(10 * MS));
        assert_eq!(d.timed_steps, window as u32 - SHAPE_WARMUP_STEPS);
        assert!((d.verify_cost_ratio().unwrap() - 2.4).abs() < 1e-9);
        assert!(d.gain().unwrap() < 1.0 - MEASURED_GAIN_MARGIN);
        // The static threshold alone would have kept it.
        if plain == PlainDecode::Candle {
            assert!(0.75 >= bonsai.threshold().unwrap());
        }
        // Qwen3.8-like: mal 1.25 at r = 1.5 — gain 1.5, kept to the end.
        let (at, qwen) = drive(
            timed(ProposerKind::Mtp, 3, plain),
            10 * MS,
            1.5,
            3,
            8 * window,
            |i| {
                if i % 4 == 0 {
                    2
                } else {
                    1
                }
            },
        );
        assert_eq!(at, None, "{plain:?}");
        let d = qwen.last_decision().unwrap();
        assert_eq!(
            (d.basis, d.demoted, d.window),
            (DemotionBasis::Measured, false, 8)
        );
        assert!((d.gain().unwrap() - 1.5).abs() < 1e-9);
    }
}

/// The margin: a window demotes only below `1 − MEASURED_GAIN_MARGIN` measured gain — at
/// 0.952 it keeps, at 0.943 it demotes (never-accepting verify steps at 1.05 / 1.06 plain
/// steps).
#[test]
fn a_measured_loss_demotes_only_beyond_the_margin() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    for (ratio, demoted) in [(1.05, false), (1.06, true)] {
        let (at, m) = drive(
            timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle),
            100_000,
            ratio,
            4,
            window,
            |_| 0,
        );
        assert_eq!(at.is_some(), demoted, "{ratio}");
        let d = m.last_decision().unwrap();
        assert_eq!((d.basis, d.demoted), (DemotionBasis::Measured, demoted));
    }
    // Measured beats static on Candle: a lookup accepting nothing (static 0.21 would demote)
    // whose verify costs no more than a plain step is kept.
    let (at, m) = drive(
        timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle),
        100_000,
        1.0,
        4,
        4 * window,
        |_| 0,
    );
    assert_eq!(at, None);
    assert_eq!(m.last_decision().unwrap().basis, DemotionBasis::Measured);
}

/// On MLX's pipelined path the probe's plain steps are dearer than the pipelined loop a
/// demoted request continues on: a measured loss demotes (`Measured`), and the static
/// threshold still demotes where the measured gain does not (`Static`).
#[test]
fn mlx_pipelined_demotes_on_a_measured_loss_or_the_static_threshold() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let lookup = || timed(ProposerKind::PromptLookup, 4, PlainDecode::MlxPipelined);
    // mal 0.25 (below the static 0.4) at r = 1.0: measured gain 1.25, static demotes.
    let quarter = |i: usize| usize::from(i.is_multiple_of(4));
    let (at, m) = drive(lookup(), 100_000, 1.0, 4, window, quarter);
    assert_eq!(at, Some(window - 1));
    assert_eq!(m.last_decision().unwrap().basis, DemotionBasis::Static);
    // The same timings and acceptance on Candle: measured decides, kept.
    let (at, _) = drive(
        timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle),
        100_000,
        1.0,
        4,
        window,
        quarter,
    );
    assert_eq!(at, None);
    // mal 1.0 (above the static 0.4) at r = 2.4: a measured loss demotes.
    let (at, m) = drive(lookup(), 100_000, 2.4, 4, window, |_| 1);
    assert_eq!(at, Some(window - 1));
    assert_eq!(m.last_decision().unwrap().basis, DemotionBasis::Measured);
    // mal 1.0 at r = 1.5: neither demotes.
    let (at, m) = drive(lookup(), 100_000, 1.5, 4, 2 * window, |_| 1);
    assert_eq!(at, None);
    assert_eq!(m.last_decision().unwrap().basis, DemotionBasis::Measured);
}

/// The plain probe and its warm-up: a timed monitor probes until it holds
/// [`PLAIN_PROBE_STEPS`] plain samples past the width-1 shape's warm-up; probe steps never
/// count towards a window. Untimed steps (no clock, or a zero-length mark) give no samples —
/// the probe ends at [`PLAIN_PROBE_MAX_STEPS`] and the static threshold decides.
#[test]
fn the_probe_times_plain_steps_past_their_warm_up_and_falls_back_untimed() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    for mark_ns in [None, Some(0)] {
        let mut m = timed(ProposerKind::Mtp, 3, PlainDecode::Candle);
        let mut probes = 0;
        while m.probing() {
            assert!(!m.observe_step(plain_step(mark_ns)));
            probes += 1;
        }
        assert_eq!(probes, PLAIN_PROBE_MAX_STEPS);
        // Static (0.5 at depth 3): mal 0.75 keeps, 0 demotes — at the window's end.
        let fired: Vec<bool> = (0..window)
            .map(|i| {
                m.observe_step(StepObservation {
                    accepted: 0,
                    drafts: 3,
                    elapsed: Some(Duration::from_millis(1)),
                }) && i == window - 1
            })
            .collect();
        assert!(fired[window - 1]);
        let d = m.last_decision().unwrap();
        assert_eq!((d.basis, d.plain_step_ns), (DemotionBasis::Static, None));
    }
    // An untimed monitor never probes.
    let mut untimed = AcceptanceMonitor::for_request(
        crate::Speculative::Auto,
        ProposerKind::Mtp,
        3,
        PlainDecode::Candle,
        false,
    )
    .unwrap();
    assert!(!untimed.probing());
    assert!((0..window).map(|_| untimed.observe(0)).last().unwrap());
    // A window whose draft count keeps changing shape has too few timed steps: static.
    let mut m = timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle);
    while m.probing() {
        m.observe_step(StepObservation {
            accepted: 0,
            drafts: 0,
            elapsed: Some(Duration::from_millis(1)),
        });
    }
    let at = (0..window).find(|&i| {
        m.observe_step(StepObservation {
            accepted: 1,
            drafts: 1 + i % 4,
            elapsed: Some(Duration::from_millis(10)),
        })
    });
    let d = m.last_decision().unwrap();
    assert_eq!(
        d.timed_steps, 4,
        "each of four widths timed once past its warm-up"
    );
    assert_eq!(d.basis, DemotionBasis::Static);
    assert_eq!(at, None, "mal 1.0 is above the static 0.21");
    // Zero-draft speculative steps (a lookup that found nothing) are plain samples too.
    let mut m = timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle);
    for _ in 0..PLAIN_PROBE_MAX_STEPS {
        m.observe_step(StepObservation {
            accepted: 0,
            drafts: 0,
            elapsed: Some(Duration::from_millis(2)),
        });
    }
    for _ in 0..window {
        m.observe_step(StepObservation {
            accepted: 0,
            drafts: 0,
            elapsed: Some(Duration::from_millis(4)),
        });
    }
    // 4 probe samples at 2 ms, 16 window samples at 4 ms: the median is 4 ms.
    assert_eq!(m.last_decision().unwrap().plain_step_ns, Some(4 * MS));
}

/// sc-24446 (cuda-campaign-b4): a clear loser is demoted by the end of the probe and its first
/// window on every plain loop — at the first step its measurement holds
/// [`CLEAR_LOSS_MIN_TIMED_STEPS`] timed steps when even its optimistic gain is a clear loss,
/// else on its window's measured gain; one between [`CLEAR_LOSS_GAIN`] and break-even is
/// judged on its whole window. Bonsai-like lookup under CUDA graphs: verify `r` 2.0–3.1, mal
/// 0.44–1.5.
#[test]
fn a_clear_loser_is_demoted_within_its_first_window() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let first = (SHAPE_WARMUP_STEPS + CLEAR_LOSS_MIN_TIMED_STEPS) as usize;
    assert!(
        first <= window,
        "the clear-loss check can decide inside the first window"
    );
    for plain in [
        PlainDecode::Candle,
        PlainDecode::MlxUnpipelined,
        PlainDecode::MlxPipelined,
    ] {
        for (proposer, depth, ratio, accepted, demoted_at) in [
            // code_edit: mal 0.44 at r 2.25 — gain 0.64 (optimistic 0.71): clear.
            (
                ProposerKind::PromptLookup,
                4,
                2.25,
                [1, 0, 0, 1, 0, 1, 0, 0],
                first - 1,
            ),
            // chat: mal 1.375 at r 3.1 — gain 0.77 (optimistic 0.84): its window decides.
            (
                ProposerKind::PromptLookup,
                4,
                3.1,
                [2, 1, 1, 2, 1, 2, 1, 1],
                window - 1,
            ),
            // A companion MTP head: mal 0.75 at r 2.6 — gain 0.67, but 3 drafts every fourth
            // step is too lumpy to call early (optimistic 0.92): its window decides.
            (
                ProposerKind::Mtp,
                3,
                2.6,
                [3, 0, 0, 0, 3, 0, 0, 0],
                window - 1,
            ),
        ] {
            let label = format!("{plain:?} {proposer:?} r {ratio}");
            let (at, m) = drive(
                timed(proposer, depth, plain),
                50 * MS,
                ratio,
                depth as usize,
                4 * window,
                |i| accepted[i % 8],
            );
            assert_eq!(at, Some(demoted_at), "{label}");
            let d = m.last_decision().unwrap();
            assert_eq!(
                (d.basis, d.demoted, d.window),
                (DemotionBasis::Measured, true, 1),
                "{label}"
            );
            assert!(d.gain().unwrap() < CLEAR_LOSS_GAIN, "{label}: {d:?}");
        }
        // gain 0.83 (mal 1.0 at r 2.4): a loser, but not a clear one — its window decides.
        let (at, m) = drive(
            timed(ProposerKind::PromptLookup, 4, plain),
            50 * MS,
            2.4,
            4,
            4 * window,
            |_| 1,
        );
        assert_eq!(at, Some(window - 1), "{plain:?}");
        assert_eq!(
            m.last_decision().unwrap().verifies,
            ACCEPTANCE_PROBE_VERIFIES
        );
    }
}

/// sc-24446 (cuda-campaign-b4 `rag_answer`): a lookup whose draft count keeps changing shape
/// spends its first window mostly on shape warm-ups — too few timed steps to measure, so the
/// window falls back to its static threshold (mal 1.0 passes Candle's 0.21) — but its timed
/// steps carry forward, and it is demoted as soon as they number
/// [`CLEAR_LOSS_MIN_TIMED_STEPS`] (gain 2 / 3 = 0.67), not at the end of its second window.
/// The decision reports the stretch it judged: every verify since the speculative phase
/// began, and the timed steps among them.
#[test]
fn a_clear_losers_timed_steps_carry_past_an_unmeasured_window() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let mut m = timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle);
    while m.probing() {
        m.observe_step(plain_step(Some(10 * MS)));
    }
    // Widths 2..=5 in turn, each step 3 plain steps, one draft accepted.
    let at = (0..4 * window).find(|&i| {
        m.observe_step(StepObservation {
            accepted: 1,
            drafts: 1 + i % 4,
            elapsed: Some(Duration::from_millis(30)),
        })
    });
    // Four widths in turn, three warm-ups each: steps 12..16 are the first window's only
    // timed steps (static: kept), and the twelfth timed step is step 23 — the second window's
    // eighth, where the window-end decision waits for step 31.
    assert_eq!(at, Some(23));
    let d = m.last_decision().unwrap();
    assert_eq!(
        (d.basis, d.demoted, d.window, d.timed_steps),
        (DemotionBasis::Measured, true, 2, CLEAR_LOSS_MIN_TIMED_STEPS)
    );
    // Steps 0..=23, one draft accepted each.
    assert_eq!((d.verifies, d.accepted), (24, 24));
    assert!((d.gain().unwrap() - 2.0 / 3.0).abs() < 1e-9, "{d:?}");
}

/// sc-24446 review: once a window has decided on its measured gain, the clear-loss check
/// stops and every later window decides at its end — a request that passed its first window
/// (gain 2 / 2.1 = 0.95) and then accepts nothing is demoted by its second window's end, not
/// by a mid-window check over every timed step so far.
#[test]
fn the_clear_loss_check_stops_once_a_window_has_measured() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let (at, m) = drive(
        timed(ProposerKind::Mtp, 3, PlainDecode::Candle),
        10 * MS,
        2.1,
        3,
        4 * window,
        |i| usize::from(i < window),
    );
    assert_eq!(at, Some(2 * window - 1));
    let d = m.last_decision().unwrap();
    assert_eq!(
        (d.basis, d.demoted, d.window, d.verifies, d.accepted),
        (
            DemotionBasis::Measured,
            true,
            2,
            ACCEPTANCE_PROBE_VERIFIES,
            0
        )
    );
}

/// sc-24446: the campaigns' measured winners are never demoted — not by the clear-loss check
/// on any eight-step stretch, nor by a window — over a long request with lumpy acceptance:
/// Qwen3-8B summary lookup on Candle (`r` 1.18, gain ≈ 1.22), Qwen3.8 MTP at depth 3 (`r` 1.71,
/// the dearest measured, gain ≈ 1.17) and Qwen3.6 MTP (`r` 2.0, gain ≈ 1.13).
#[test]
fn measured_winners_are_never_demoted() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    type Pattern = [usize; 16];
    type Case<'a> = (&'a str, ProposerKind, u32, f64, Pattern, &'a [PlainDecode]);
    let cases: [Case<'_>; 3] = [
        (
            "qwen3-8b summary lookup",
            ProposerKind::PromptLookup,
            4,
            1.18,
            // mal 0.44: a 4-token run, then long stretches of nothing.
            [4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            &[PlainDecode::Candle],
        ),
        (
            "qwen3.8 mtp",
            ProposerKind::Mtp,
            3,
            1.71,
            // mal 1.0.
            [3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1, 3, 0, 0, 1],
            &[PlainDecode::Candle, PlainDecode::MlxUnpipelined],
        ),
        (
            "qwen3.6 mtp",
            ProposerKind::Mtp,
            3,
            2.0,
            // mal 1.25.
            [3, 0, 2, 0, 3, 0, 2, 0, 3, 0, 2, 0, 3, 0, 2, 0],
            &[PlainDecode::MlxUnpipelined, PlainDecode::MlxPipelined],
        ),
    ];
    for (label, proposer, depth, ratio, pattern, plains) in cases {
        for &plain in plains {
            let (at, m) = drive(
                timed(proposer, depth, plain),
                10 * MS,
                ratio,
                depth as usize,
                16 * window,
                |i| pattern[i % 16],
            );
            assert_eq!(at, None, "{label} {plain:?}");
            let d = m.last_decision().unwrap();
            assert_eq!(
                (d.basis, d.demoted, d.window),
                (DemotionBasis::Measured, false, 16),
                "{label} {plain:?}"
            );
            assert!(d.gain().unwrap() > 1.05, "{label} {plain:?}: {d:?}");
        }
    }
}

/// sc-24446 (cuda-campaign-a4f1 `f1-qwen38-graphs-off` epic-2 `creative`): a host stall does
/// not demote a winner. Qwen3.8-like MTP requests (`r` 1.5–1.8, gain 1.2–2.5) with stalled
/// steps — one at 5× its cost, one at 15× (the campaign's ≥ 1.7 s in sixteen ~122 ms steps),
/// three at 5× in one window — inside the clear-loss stretch and in a later window keep their
/// proposer: each stall is left out of its stretch ([`STALL_STEP_FACTOR`]). All but the last
/// (one 5× stall a window, which a gain ≥ 1.2 absorbs on a mean cost too) read as a loss on
/// their mean step cost. Bonsai-like losers with the same stalls still demote.
#[test]
fn a_stalled_step_does_not_demote_a_winner() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let plain_ns = 70 * MS;
    // (label, r, accepted per step (mal), stalled steps, stall factor)
    type Case<'a> = (&'a str, f64, [usize; 4], &'a [usize], f64);
    let cases: [Case<'_>; 5] = [
        (
            "r 1.8 mal 1.25, one 15x stall in window 3",
            1.8,
            [2, 1, 1, 1],
            &[37],
            15.0,
        ),
        (
            "r 1.8 mal 1.25, one 15x stall in the clear stretch",
            1.8,
            [2, 1, 1, 1],
            &[6],
            15.0,
        ),
        (
            "r 1.5 mal 1.0, three 5x stalls in window 2",
            1.5,
            [1, 1, 1, 1],
            &[17, 20, 25],
            5.0,
        ),
        (
            "r 1.7 mal 1.25, three 5x stalls in the clear stretch",
            1.7,
            [2, 1, 1, 1],
            &[4, 8, 11],
            5.0,
        ),
        (
            "r 1.5 mal 2.75, one 5x stall in every window",
            1.5,
            [3, 3, 2, 3],
            &[],
            5.0,
        ),
    ];
    for (label, ratio, pattern, stalls, factor) in cases {
        let stalled = |i: usize| {
            if stalls.is_empty() {
                i % window == 9
            } else {
                stalls.contains(&i)
            }
        };
        let (at, m) = drive_costs(
            timed(ProposerKind::Mtp, 3, PlainDecode::Candle),
            plain_ns,
            3,
            8 * window,
            |i| pattern[i % 4],
            |i| if stalled(i) { ratio * factor } else { ratio },
        );
        assert_eq!(at, None, "{label}");
        let d = m.last_decision().unwrap();
        assert_eq!(
            (d.basis, d.demoted, d.window),
            (DemotionBasis::Measured, false, 8),
            "{label}"
        );
        // The stall-free gain, (1 + mal) / r: a winner.
        let mal = pattern.iter().sum::<usize>() as f64 / 4.0;
        assert!((1.0 + mal) / ratio >= 1.2, "{label}");
    }
    // A real loser keeps losing with the same stalls: Bonsai-like lookup (r 2.25, mal 0.44)
    // and MTP (r 3.1, mal 1.0) still go by the end of the probe and two windows.
    for (proposer, depth, ratio, pattern) in [
        (
            ProposerKind::PromptLookup,
            4,
            2.25,
            [1, 0, 0, 1, 0, 1, 0, 0],
        ),
        (ProposerKind::Mtp, 3, 3.1, [1, 1, 1, 1, 1, 1, 1, 1]),
    ] {
        for stalls in [&[][..], &[6][..], &[4, 8, 11][..], &[17, 20, 25][..]] {
            let (at, _) = drive_costs(
                timed(proposer, depth, PlainDecode::Candle),
                plain_ns,
                depth as usize,
                8 * window,
                |i| pattern[i % 8],
                |i| {
                    if stalls.contains(&i) {
                        ratio * 5.0
                    } else {
                        ratio
                    }
                },
            );
            let at = at.unwrap_or_else(|| panic!("{proposer:?} {stalls:?}: never demoted"));
            assert!(at < 2 * window, "{proposer:?} {stalls:?}: demoted at {at}");
        }
    }
}

/// sc-24446 review: a mixed-width loser is not excused as stalls. A lookup whose steps are
/// 70 % single-token (nothing found, 1 plain step) and 30 % full-width verifies at 4.5 plain
/// steps accepting nothing — true gain 10 / 20.5 ≈ 0.49 — still demotes by the end of its
/// second window: its wide steps are ~0.9 plain step per verified token, not stalls, though
/// each is 4.5× the median step.
#[test]
fn a_mixed_width_loser_is_not_excused_as_stalls() {
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let plain_ns = 10 * MS;
    let mut m = timed(ProposerKind::PromptLookup, 4, PlainDecode::Candle);
    while m.probing() {
        m.observe_step(plain_step(Some(plain_ns)));
    }
    let at = (0..8 * window).find(|&i| {
        let wide = matches!(i % 10, 2 | 5 | 8);
        m.observe_step(StepObservation {
            accepted: 0,
            drafts: if wide { 4 } else { 0 },
            elapsed: Some(Duration::from_nanos(if wide {
                plain_ns * 9 / 2
            } else {
                plain_ns
            })),
        })
    });
    let at = at.expect("demoted");
    assert!(at < 2 * window, "demoted at {at}");
    let d = m.last_decision().unwrap();
    assert_eq!((d.basis, d.demoted), (DemotionBasis::Measured, true));
    assert!(d.gain().unwrap() < 1.0 - MEASURED_GAIN_MARGIN, "{d:?}");
}

/// One seeded timed `auto` request for the clear-loss property tests: acceptance from a
/// two-state (good / bad) Markov chain switching with probability 0.3 per step — each draft
/// of a step accepted in turn with the state's probability, up to the step's drafts — and
/// every step's time (probe and verify) jittered uniformly by ±8.7 % (σ 5 %, about the MLX
/// in-process p90). `varying` cycles the draft count 1..=4 (a lookup, whose shapes keep
/// warming up); otherwise every step drafts `depth`. Returns the 0-based speculative step
/// that demoted, and whether a clear-loss decision did it (not a window's end).
fn seeded_request(
    seed: u64,
    proposer: ProposerKind,
    depth: usize,
    ratio: f64,
    (good, bad): (f32, f32),
    varying: bool,
    steps: usize,
) -> Option<(usize, bool)> {
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let plain_ns = 10.0 * MS as f64;
    let jittered = |rng: &mut Rng, cost: f64| {
        let u = f64::from(rng.next()) * 2.0 - 1.0;
        Duration::from_nanos((plain_ns * cost * (1.0 + 0.087 * u)) as u64)
    };
    let mut m = timed(proposer, depth as u32, PlainDecode::Candle);
    while m.probing() {
        let elapsed = jittered(&mut rng, 1.0);
        m.observe_step(StepObservation {
            accepted: 0,
            drafts: 0,
            elapsed: Some(elapsed),
        });
    }
    let mut in_good = rng.next() < 0.5;
    for i in 0..steps {
        if rng.next() < 0.3 {
            in_good = !in_good;
        }
        let p = if in_good { good } else { bad };
        let drafts = if varying { 1 + i % 4 } else { depth };
        let mut accepted = 0;
        while accepted < drafts && rng.next() < p {
            accepted += 1;
        }
        let elapsed = jittered(&mut rng, ratio);
        if m.observe_step(StepObservation {
            accepted,
            drafts,
            elapsed: Some(elapsed),
        }) {
            let window = ACCEPTANCE_PROBE_VERIFIES as usize;
            let at_window_end = m
                .last_decision()
                .is_some_and(|d| d.verifies == ACCEPTANCE_PROBE_VERIFIES && (i + 1) % window == 0);
            return Some((i, !at_window_end));
        }
    }
    None
}

/// sc-24446 review: the clear-loss check does not demote the campaigns' measured winners on
/// a bursty stretch of acceptance under timing jitter — Qwen3.6-like MTP (`r` 2.0, mal 1.25
/// at depth 3), Qwen3.8 MTP (`r` 1.71, mal 1.0) and Qwen3-8B lookup (`r` 1.18, mal 0.44) —
/// over 400 seeded 160-step requests each: at most 1 % (a running point estimate from the
/// eighth timed step demoted about half of the Qwen3.6-like ones). Bonsai-like clear losers
/// whose shapes keep warming up (a lookup at `r` 2.25 / mal 0.44 and `r` 3.0 / mal 1.0) are
/// still all demoted by the end of their second window, and the clear-loss check takes most
/// of them before it.
#[test]
fn seeded_bursty_winners_keep_their_proposer_and_clear_losers_go_early() {
    const REQUESTS: u64 = 400;
    // (label, proposer, depth, r, (good, bad) per-draft acceptance): mal from the chain.
    let winners = [
        ("qwen3.6 mtp", ProposerKind::Mtp, 3, 2.0, (0.772, 0.42)),
        ("qwen3.8 mtp", ProposerKind::Mtp, 3, 1.71, (0.694, 0.338)),
        (
            "qwen3-8b lookup",
            ProposerKind::PromptLookup,
            4,
            1.18,
            (0.457, 0.069),
        ),
    ];
    for (label, proposer, depth, ratio, chain) in winners {
        let clear = (0..REQUESTS)
            .filter(|&seed| {
                seeded_request(seed, proposer, depth, ratio, chain, false, 160)
                    .is_some_and(|(_, clear)| clear)
            })
            .count() as u64;
        assert!(
            clear * 100 <= REQUESTS,
            "{label}: {clear} of {REQUESTS} demoted by clear loss"
        );
    }
    let window = ACCEPTANCE_PROBE_VERIFIES as usize;
    let losers = [
        ("code_edit lookup", 2.25, (0.457, 0.069)),
        ("rag_answer lookup", 3.0, (0.669, 0.279)),
    ];
    for (label, ratio, chain) in losers {
        let mut at: Vec<usize> = (0..REQUESTS)
            .map(|seed| {
                seeded_request(seed, ProposerKind::PromptLookup, 4, ratio, chain, true, 160)
                    .unwrap_or_else(|| panic!("{label} seed {seed}: never demoted"))
                    .0
            })
            .collect();
        at.sort_unstable();
        assert!(
            at[at.len() - 1] < 2 * window,
            "{label}: by the second window's end"
        );
        assert!(
            at[at.len() / 2] < 2 * window - 1,
            "{label}: the median request goes before its second window ends (median {})",
            at[at.len() / 2]
        );
    }
}

#[test]
fn greedy_commit_is_the_accepted_prefix_plus_the_bonus() {
    assert_eq!(greedy_commit(&[1, 2, 9], &[1, 2]), (vec![1, 2, 9], 2));
    assert_eq!(greedy_commit(&[1, 5, 9], &[1, 2]), (vec![1, 5], 1));
    assert_eq!(greedy_commit(&[7, 5, 9], &[1, 2]), (vec![7], 0));
    assert_eq!(greedy_commit(&[9], &[]), (vec![9], 0));
}

// --- greedy acceptance ---

#[test]
fn greedy_accepts_matching_prefix() {
    // drafts d, target argmax (with bonus slot).
    assert_eq!(accept_greedy_run(&[1, 2, 9], &[1, 2]), 2); // both match
    assert_eq!(accept_greedy_run(&[1, 5, 9], &[1, 2]), 1); // 2nd diverges
    assert_eq!(accept_greedy_run(&[7, 5, 9], &[1, 2]), 0); // 1st diverges
    assert_eq!(accept_greedy_run(&[9], &[]), 0); // no drafts -> bonus only
}

// --- stochastic acceptance: basic behaviour ---

#[test]
fn accept_when_target_dominates() {
    // p strongly favours the proposed token -> accept_prob ~1.
    let p = [(0, 0.9f32), (1, 0.1)];
    let q = [(0, 1.0f32)]; // point mass at 0 (prompt-lookup style)
    assert_eq!(accept_token(&p, &q, 0, 0.5, 0.0), Acceptance::Accepted(0));
}

#[test]
fn reject_resamples_from_residual_excluding_proposed_for_point_mass() {
    // Point-mass draft at 0; on rejection the residual is p with 0 removed -> must yield 1 or 2.
    let p = [(0, 0.2f32), (1, 0.5), (2, 0.3)];
    let q = [(0, 1.0f32)];
    // Force rejection with u_accept above accept_prob (=p(0)=0.2).
    match accept_token(&p, &q, 0, 0.99, 0.1) {
        Acceptance::Rejected(t) => assert!(t == 1 || t == 2, "got {t}"),
        other => panic!("expected rejection, got {other:?}"),
    }
}

/// A tiny deterministic PRNG for the Monte-Carlo test (xorshift64*).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

#[test]
fn monte_carlo_output_matches_target_distribution() {
    // The headline guarantee: drawing the proposal from q and running accept_token yields the
    // committed token distributed as p, for arbitrary p, q.
    let p = [(0, 0.10f32), (1, 0.35), (2, 0.05), (3, 0.30), (4, 0.20)];
    let q = [(0, 0.30f32), (1, 0.10), (2, 0.25), (3, 0.20), (4, 0.15)];
    let q_total: f32 = q.iter().map(|&(_, w)| w).sum();

    let mut rng = Rng(0x1234_5678_9abc_def0);
    let n = 400_000;
    let mut counts = [0u64; 5];
    for _ in 0..n {
        let proposed = sample_weighted(&q, rng.next(), 0);
        let committed = accept_token(&p, &q, proposed, rng.next(), rng.next()).token();
        counts[committed as usize] += 1;
    }
    // q is a proper distribution (sums to 1) so the draw is unbiased.
    assert!((q_total - 1.0).abs() < 1e-6);
    for (i, &(_, pi)) in p.iter().enumerate() {
        let emp = counts[i] as f32 / n as f32;
        assert!(
            (emp - pi).abs() < 0.01,
            "token {i}: empirical {emp} vs target {pi}"
        );
    }
}

#[test]
fn greedy_is_the_point_mass_special_case() {
    // Target point mass at argmax=3; proposing 3 accepts, proposing anything else rejects to 3.
    let p = [(3, 1.0f32)];
    let q = [(3, 1.0f32)];
    assert_eq!(accept_token(&p, &q, 3, 0.999, 0.5), Acceptance::Accepted(3));
    let p2 = [(3, 1.0f32)];
    let q2 = [(1, 1.0f32)];
    assert_eq!(
        accept_token(&p2, &q2, 1, 0.999, 0.5),
        Acceptance::Rejected(3)
    );
}

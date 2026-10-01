//! Backend-neutral speculative-decoding policy (epic 7153, stories 7171 + 7172).
//!
//! Speculative decoding generates several tokens per target forward by **proposing** a short
//! continuation cheaply, **verifying** it in one batched target forward, and **accepting** the
//! longest prefix the target agrees with. The proposal source differs by story — an n-gram match of
//! the context ([`ngram_propose`], story 7171) or a small draft model (story 7172) — but the
//! acceptance is the same distribution-preserving rule, so it lives here once, tensor-free, and both
//! backends ([`mlx-llm`], later `candle-llm`) reuse it.
//!
//! ## Distribution preservation
//! [`accept_token`] is the Leviathan et al. / Chen et al. speculative-sampling step: given the target
//! distribution `p` and the draft distribution `q` from which the proposed token was drawn, accept the
//! proposal with probability `min(1, p(t)/q(t))`, and on rejection resample from the normalized
//! residual `max(0, p − q)`. The committed token is then distributed **exactly** as `p` — speculative
//! decoding changes the *speed*, not the *output distribution*. Greedy decoding is the special case
//! where `p` is a point mass at the argmax: [`accept_greedy_run`] is the efficient form (accept the
//! longest draft prefix equal to the target's argmax).
//!
//! The acceptance functions take their random draws as parameters rather than owning an RNG, so this
//! module stays tensor-free *and* RNG-free — deterministic and exhaustively unit-testable, with the
//! backend feeding draws from its own seeded PRNG.
//!
//! ## The unified engine's policy (epic sc-24128, story sc-24130)
//! The Candle engine runs **one** speculative loop over its step-model seam with pluggable
//! proposers; the parts of that loop that are policy rather than tensor work live here so MLX can
//! adopt them unchanged: which proposer a request resolves to ([`resolve_speculative`],
//! [`ProposerKind`]; the proposer-agnostic request option since sc-24433) and the verify decision ([`greedy_commit`] for greedy, [`accept_token`] for
//! stochastic — the same acceptance rule as before, unchanged).
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm

/// Which proposal source a speculative run used (epic sc-24128, story sc-24130). Every decode
/// record names one, so "which proposer ran" is visible per request and a request that resolved
/// to no proposer says so (`none`) rather than silently downgrading. The labels are the request
/// vocabulary of [`SpeculativeProposer`](crate::SpeculativeProposer) plus `none` (sc-24433).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposerKind {
    /// No proposer: the token-at-a-time path (also what a request whose speculation resolved to
    /// nothing runs — its [`DecodeReport::fallbacks`](crate::DecodeReport::fallbacks) says why).
    #[default]
    None,
    /// The checkpoint-native multi-token-prediction head.
    Mtp,
    /// Prompt lookup: [`ngram_propose`] over the context.
    PromptLookup,
    /// A separate draft model.
    DraftModel,
}

impl ProposerKind {
    /// Stable lower-case label for logs and evidence rows (`none`, `mtp`, `prompt_lookup`,
    /// `draft_model`).
    pub fn label(self) -> &'static str {
        match self {
            ProposerKind::None => "none",
            ProposerKind::Mtp => "mtp",
            ProposerKind::PromptLookup => "prompt_lookup",
            ProposerKind::DraftModel => "draft_model",
        }
    }
}

impl From<crate::SpeculativeProposer> for ProposerKind {
    fn from(proposer: crate::SpeculativeProposer) -> Self {
        match proposer {
            crate::SpeculativeProposer::Mtp => ProposerKind::Mtp,
            crate::SpeculativeProposer::PromptLookup => ProposerKind::PromptLookup,
            crate::SpeculativeProposer::DraftModel => ProposerKind::DraftModel,
        }
    }
}

/// What a request's [`Speculative`](crate::Speculative) option resolves to against the loaded
/// model ([`resolve_speculative`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpeculativePlan {
    /// Decode token-at-a-time; the record names `proposer=none`.
    #[default]
    Off,
    /// Run `proposer` with `depth` drafts per verify step.
    Run {
        /// The proposal source.
        proposer: crate::SpeculativeProposer,
        /// Drafts proposed per target verification pass (`>= 1`).
        depth: u32,
    },
}

impl SpeculativePlan {
    /// The proposer this plan runs (`none` when off).
    pub fn proposer(self) -> ProposerKind {
        match self {
            SpeculativePlan::Off => ProposerKind::None,
            SpeculativePlan::Run { proposer, .. } => proposer.into(),
        }
    }

    /// The draft width, or `None` when off.
    pub fn depth(self) -> Option<u32> {
        match self {
            SpeculativePlan::Off => None,
            SpeculativePlan::Run { depth, .. } => Some(depth),
        }
    }
}

/// A resolved speculative option: the plan to run and, when the request asked for more than the
/// plan delivers, the named reason ([`DecodeReport::fallbacks`](crate::DecodeReport::fallbacks),
/// epic sc-24432 E2) — never a silent downgrade.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeResolution {
    /// What to run.
    pub plan: SpeculativePlan,
    /// Why the plan is less than the request asked for, when it is.
    pub fallback: Option<String>,
}

/// Resolve a request's speculative option against what the loaded model advertises — the
/// backend-neutral policy both engines apply (epic sc-24432 E4):
///
/// * `off` never speculates;
/// * `auto` runs MTP at its recommended depth where the model has a head, else prompt lookup at
///   its recommended depth, else decodes plainly with the reason named;
/// * `{proposer, depth}` runs that proposer. Its admissibility (advertised, `depth >= 1`) is
///   checked by [`TextLlmCapabilities::validate_request`](crate::TextLlmCapabilities::validate_request)
///   first, so an un-advertised proposer here is the caller's contract violation: it resolves to
///   `off` with the reason named rather than inventing a proposer. A depth above the advertised
///   `max_depth` — the backend-true bound — is clamped to it and the clamp named in the fallback
///   (sc-24438), never refused and never run past the bound.
pub fn resolve_speculative(
    mode: crate::Speculative,
    capabilities: &crate::TextLlmCapabilities,
) -> SpeculativeResolution {
    use crate::{Speculative, SpeculativeProposer};
    let run = |proposer, depth| SpeculativeResolution {
        plan: SpeculativePlan::Run { proposer, depth },
        fallback: None,
    };
    match mode {
        Speculative::Off => SpeculativeResolution::default(),
        Speculative::Auto => [SpeculativeProposer::Mtp, SpeculativeProposer::PromptLookup]
            .into_iter()
            .find_map(|p| capabilities.proposer(p))
            .map(|cap| {
                run(
                    cap.proposer,
                    cap.recommended_depth.clamp(1, cap.max_depth.max(1)),
                )
            })
            .unwrap_or_else(|| SpeculativeResolution {
                plan: SpeculativePlan::Off,
                fallback: Some(
                    "speculative: auto found no proposer this model can run (no MTP head, no \
                     prompt lookup on this backend)"
                        .into(),
                ),
            }),
        Speculative::Proposer { proposer, depth } => match capabilities.proposer(proposer) {
            None => SpeculativeResolution {
                plan: SpeculativePlan::Off,
                fallback: Some(format!(
                    "speculative: `{proposer}` is not available for this model"
                )),
            },
            Some(cap) => {
                let clamped = depth.clamp(1, cap.max_depth.max(1));
                SpeculativeResolution {
                    plan: SpeculativePlan::Run {
                        proposer,
                        depth: clamped,
                    },
                    fallback: (clamped != depth).then(|| {
                        format!(
                            "speculative: `{proposer}` depth {depth} clamped to {clamped} \
                             (advertised 1..={})",
                            cap.max_depth
                        )
                    }),
                }
            }
        },
    }
}

/// The MTP depth [`Speculative::Auto`](crate::Speculative::Auto) runs on a checkpoint's MTP head (the
/// upstream recommendation). Recorded per backend in [`crate::defaults`].
pub const MTP_RECOMMENDED_DEPTH: u32 = 3;

/// The prompt-lookup depth [`Speculative::Auto`](crate::Speculative::Auto) runs on a model without
/// an MTP head. A lookup that finds no match proposes nothing and the step is an ordinary
/// single-token decode, so the depth only prices a match. Recorded per backend in
/// [`crate::defaults`].
pub const PROMPT_LOOKUP_RECOMMENDED_DEPTH: u32 = 4;

/// The draft-model depth advertised as recommended (sc-24436): four drafts per verify step, or
/// the provider's bound when that is shallower.
pub const DRAFT_MODEL_RECOMMENDED_DEPTH: u32 = 4;

/// The `draft_model` advertisement a provider carries while a compatible draft is resident
/// (sc-24436): `1..=max_depth` drafts per verify step, recommended
/// [`DRAFT_MODEL_RECOMMENDED_DEPTH`] (or `max_depth` when shallower). `max_depth` is the same
/// per-model verify bound the provider advertises for its other proposers — every draft is one
/// more verify row, whichever proposer drew it — so a provider passes that one value here.
pub fn draft_model_capabilities(max_depth: u32) -> crate::ProposerCapabilities {
    crate::ProposerCapabilities {
        proposer: crate::SpeculativeProposer::DraftModel,
        max_depth,
        recommended_depth: DRAFT_MODEL_RECOMMENDED_DEPTH.min(max_depth),
    }
}

/// Whether a draft model can propose for a target (sc-24436), or the named reason it cannot.
/// The target verifies each draft id as its own token and the acceptance rule compares the two
/// models' distributions row for row, so the draft must share the target's tokenizer vocabulary
/// token for token ([`Tokenizer::vocabulary_mismatch`](crate::Tokenizer::vocabulary_mismatch))
/// and score no ids the target does not (`*_logits` — the models' `vocab_size`, which may exceed
/// the tokenizer's by padding rows). A draft padded less than its target is compatible: it
/// scores every real token. `Ok` carries how many leading draft ids a proposer may draw — the
/// tokenizer's tokens the draft scores — so a padding row is never proposed. The reason leads
/// with `draft model:`.
pub fn draft_compatibility(
    target: &crate::Tokenizer,
    target_logits: usize,
    draft: &crate::Tokenizer,
    draft_logits: usize,
) -> Result<usize, String> {
    if let Some(why) = target.vocabulary_mismatch(draft) {
        return Err(format!(
            "draft model: its tokenizer vocabulary is not the target's ({why})"
        ));
    }
    if draft_logits > target_logits {
        return Err(format!(
            "draft model: it scores {draft_logits} token ids, more than the target's {target_logits}"
        ));
    }
    Ok(draft_logits.min(draft.vocab_size()))
}

/// A `draft_model` plan the resident draft cannot cover (sc-24436, E2): when the positions its
/// cache must hold — the request's `prompt_tokens + max_new_tokens` plus the `depth + 1` a
/// draft step writes past the committed tokens — outrun the draft's own context window
/// (`draft_context`; `0` is unbounded), the request runs what
/// [`Speculative::Auto`](crate::Speculative::Auto) resolves to on these `capabilities` instead —
/// MTP or prompt lookup, else plain decoding — with the reason named, rather than driving the
/// draft past the positions it was built for. Any other resolution is returned unchanged.
pub fn fit_draft_context(
    resolution: SpeculativeResolution,
    capabilities: &crate::TextLlmCapabilities,
    draft_context: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> SpeculativeResolution {
    let SpeculativePlan::Run {
        proposer: crate::SpeculativeProposer::DraftModel,
        depth,
    } = resolution.plan
    else {
        return resolution;
    };
    let reach = prompt_tokens
        .saturating_add(max_new_tokens as usize)
        .saturating_add(depth as usize)
        .saturating_add(1);
    if draft_context == 0 || reach <= draft_context {
        return resolution;
    }
    let auto = resolve_speculative(crate::Speculative::Auto, capabilities);
    let runs = match auto.plan {
        SpeculativePlan::Run { proposer, depth } => format!("`{proposer}` at depth {depth}"),
        SpeculativePlan::Off => "without a proposer".into(),
    };
    let why = format!(
        "speculative: `draft_model` cannot cover this request: prompt ({prompt_tokens} tokens) + \
         requested generation ({max_new_tokens}) + a depth-{depth} draft step's {} positions \
         exceeds the draft model's context window {draft_context}; decoded as `auto` would, \
         {runs}",
        depth + 1
    );
    SpeculativeResolution {
        plan: auto.plan,
        fallback: Some(match auto.fallback {
            Some(more) => format!("{why}; {more}"),
            None => why,
        }),
    }
}

/// The greedy verify decision in one call: the committed run (accepted drafts + the bonus token)
/// and how many drafts were accepted, from the target's per-position argmax
/// (`target_argmax.len() == drafts.len() + 1`, see [`accept_greedy_run`]). Every committed token is
/// the target's own greedy choice, so a greedy speculative run commits exactly what token-at-a-time
/// greedy decoding would have chosen at each position.
pub fn greedy_commit(target_argmax: &[i32], drafts: &[i32]) -> (Vec<i32>, usize) {
    let accepted = accept_greedy_run(target_argmax, drafts);
    let mut committed = drafts[..accepted].to_vec();
    committed.push(target_argmax[accepted]);
    (committed, accepted)
}

/// Propose a continuation by **prompt lookup**: find the most recent earlier occurrence of the
/// sequence's trailing n-gram and return the tokens that followed it (story 7171).
///
/// Tries the longest n-gram first (down to length 1): for each size it looks for the rightmost match
/// of the last `n` tokens *strictly before* the trailing copy, and returns up to `max_proposal` tokens
/// that followed that earlier match. Returns empty when nothing matches or inputs are degenerate —
/// the caller then falls back to a single-token step. No draft model, no tensors.
pub fn ngram_propose(tokens: &[i32], max_ngram: usize, max_proposal: usize) -> Vec<i32> {
    let len = tokens.len();
    if len < 2 || max_ngram == 0 || max_proposal == 0 {
        return Vec::new();
    }
    let max_n = max_ngram.min(len - 1);
    for n in (1..=max_n).rev() {
        let suffix = &tokens[len - n..];
        // Search earlier start positions, most-recent first; the match must end before the suffix.
        for start in (0..=len - n - 1).rev() {
            if &tokens[start..start + n] == suffix {
                let from = start + n;
                let to = (from + max_proposal).min(len);
                if from < to {
                    return tokens[from..to].to_vec();
                }
            }
        }
    }
    Vec::new()
}

/// How many leading draft tokens a **greedy** target accepts: the length of the prefix of `drafts`
/// equal, position by position, to the target's argmax (`target_argmax[i]` is the target's greedy
/// token at draft position `i`). The committed run is then those accepted drafts followed by the
/// bonus token `target_argmax[accepted]` — every committed token equals the target's own greedy
/// choice, so greedy speculative output is identical to non-speculative greedy output.
///
/// `target_argmax.len()` must be `drafts.len() + 1` (one extra for the always-present bonus). Returns
/// a value in `0..=drafts.len()`.
pub fn accept_greedy_run(target_argmax: &[i32], drafts: &[i32]) -> usize {
    debug_assert_eq!(
        target_argmax.len(),
        drafts.len() + 1,
        "need a bonus slot past the drafts"
    );
    let mut accepted = 0;
    while accepted < drafts.len() && target_argmax[accepted] == drafts[accepted] {
        accepted += 1;
    }
    accepted
}

/// The outcome of a stochastic [`accept_token`] step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Acceptance {
    /// The proposed token was accepted; the run may continue to the next position.
    Accepted(i32),
    /// The proposal was rejected; `i32` is the bonus token resampled from the residual, and the run
    /// ends here.
    Rejected(i32),
}

impl Acceptance {
    /// The committed token regardless of outcome.
    pub fn token(self) -> i32 {
        match self {
            Acceptance::Accepted(t) | Acceptance::Rejected(t) => t,
        }
    }

    /// Whether the proposal was accepted (the run continues).
    pub fn is_accepted(self) -> bool {
        matches!(self, Acceptance::Accepted(_))
    }
}

/// One distribution-preserving speculative-sampling step.
///
/// `target` and `draft` are `(token, weight)` candidate sets (weights need not be normalized — each
/// is normalized over its own set). `proposed` is the token the draft sampled. `u_accept` and
/// `u_resample` are independent uniform `[0, 1)` draws from the backend's RNG.
///
/// Accepts `proposed` with probability `min(1, p(proposed)/q(proposed))`; on rejection resamples a
/// bonus token from the normalized residual `max(0, p − q)`. The committed token is distributed
/// exactly as the (normalized) `target`.
pub fn accept_token(
    target: &[(i32, f32)],
    draft: &[(i32, f32)],
    proposed: i32,
    u_accept: f32,
    u_resample: f32,
) -> Acceptance {
    let p_total: f32 = target.iter().map(|&(_, w)| w.max(0.0)).sum();
    let q_total: f32 = draft.iter().map(|&(_, w)| w.max(0.0)).sum();
    let p_of = |t: i32| weight_of(target, t).max(0.0) / p_total.max(f32::MIN_POSITIVE);
    let q_of = |t: i32| weight_of(draft, t).max(0.0) / q_total.max(f32::MIN_POSITIVE);

    let p_t = p_of(proposed);
    let q_t = q_of(proposed);
    // q_t should be > 0 (proposed was drawn from q); guard anyway. Accept w.p. min(1, p/q).
    let accept_prob = if q_t > 0.0 { (p_t / q_t).min(1.0) } else { 1.0 };
    if u_accept < accept_prob {
        return Acceptance::Accepted(proposed);
    }

    // Reject: resample from the normalized residual max(0, p - q) over the union of supports.
    let mut residual: Vec<(i32, f32)> = Vec::with_capacity(target.len() + draft.len());
    for &(t, _) in target.iter().chain(draft.iter()) {
        if residual.iter().all(|&(s, _)| s != t) {
            let r = p_of(t) - q_of(t);
            if r > 0.0 {
                residual.push((t, r));
            }
        }
    }
    Acceptance::Rejected(sample_weighted(&residual, u_resample, proposed))
}

/// Draw a token from a `(token, weight)` candidate set by inverse-CDF over the normalized weights.
/// `fallback` is returned only if the set is empty or all-zero (degenerate). Public so a backend can
/// draw the post-all-accepted bonus from the final target distribution with the same policy.
pub fn sample_weighted(candidates: &[(i32, f32)], u: f32, fallback: i32) -> i32 {
    let total: f32 = candidates.iter().map(|&(_, w)| w.max(0.0)).sum();
    if total <= 0.0 {
        return fallback;
    }
    let mut target = u.clamp(0.0, 1.0) * total;
    for &(t, w) in candidates {
        target -= w.max(0.0);
        if target <= 0.0 {
            return t;
        }
    }
    candidates.last().map(|&(t, _)| t).unwrap_or(fallback)
}

/// Weight of `token` in a candidate set (`0` if absent).
fn weight_of(candidates: &[(i32, f32)], token: i32) -> f32 {
    candidates
        .iter()
        .find(|&&(t, _)| t == token)
        .map(|&(_, w)| w)
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(auto.fallback.unwrap().contains("auto found no proposer"));

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

        let caps = draft_model_capabilities(8);
        assert_eq!(caps.proposer, SpeculativeProposer::DraftModel);
        assert_eq!((caps.max_depth, caps.recommended_depth), (8, 4));
        // The provider's bound is the advertisement's; the recommendation never exceeds it.
        let shallow = draft_model_capabilities(3);
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
            speculative: vec![lookup, draft_model_capabilities(8)],
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
            speculative: vec![draft_model_capabilities(8)],
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
}

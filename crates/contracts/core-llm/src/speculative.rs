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
//! adopt them unchanged: which proposer a request resolves to ([`resolve_mtp_plan`],
//! [`ProposerKind`]) and the verify decision ([`greedy_commit`] for greedy, [`accept_token`] for
//! stochastic — the same acceptance rule as before, unchanged).
//!
//! [`mlx-llm`]: https://github.com/SceneWorks/mlx-llm

/// Which proposal source a speculative run used (epic sc-24128, story sc-24130). Every decode
/// record names one, so "which proposer ran" is visible per request and a request that resolved
/// to no proposer says so (`none`) rather than silently downgrading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProposerKind {
    /// No proposer: the token-at-a-time path (also what [`MtpMode::Auto`](crate::MtpMode::Auto)
    /// resolves to on a model without an MTP head).
    #[default]
    None,
    /// The checkpoint-native multi-token-prediction head.
    Mtp,
    /// Prompt lookup: [`ngram_propose`] over the context.
    Ngram,
    /// A separate draft model.
    Draft,
}

impl ProposerKind {
    /// Stable lower-case label for logs and evidence rows (`none`, `mtp`, `ngram`, `draft`).
    pub fn label(self) -> &'static str {
        match self {
            ProposerKind::None => "none",
            ProposerKind::Mtp => "mtp",
            ProposerKind::Ngram => "ngram",
            ProposerKind::Draft => "draft",
        }
    }
}

/// What a request's [`MtpMode`](crate::MtpMode) resolves to against the loaded model
/// ([`resolve_mtp_plan`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MtpPlan {
    /// Decode token-at-a-time; the record names `proposer=none`.
    Off,
    /// Run the MTP proposer with `draft_tokens` drafts per verify step.
    Mtp {
        /// Drafts proposed per target verification pass.
        draft_tokens: u32,
    },
}

impl MtpPlan {
    /// The proposer this plan runs.
    pub fn proposer(self) -> ProposerKind {
        match self {
            MtpPlan::Off => ProposerKind::None,
            MtpPlan::Mtp { .. } => ProposerKind::Mtp,
        }
    }

    /// The draft width, or `None` when off.
    pub fn draft_tokens(self) -> Option<u32> {
        match self {
            MtpPlan::Off => None,
            MtpPlan::Mtp { draft_tokens } => Some(draft_tokens),
        }
    }
}

/// Resolve a request's MTP mode against what the model advertises — the backend-neutral mode
/// policy both engines apply: `Off` never speculates; `Auto` runs the advertised recommended
/// width when the model has a head and decodes normally (`proposer=none`) when it does not;
/// `Enabled` runs exactly the requested width (its admissibility is checked by
/// [`TextLlmCapabilities::validate`](crate::TextLlmCapabilities::validate) before this point, so
/// an un-advertised `Enabled` here is the caller's contract violation and resolves to `Off`
/// rather than inventing a width).
pub fn resolve_mtp_plan(
    mode: crate::MtpMode,
    advertised: Option<crate::MtpCapabilities>,
) -> MtpPlan {
    match (mode, advertised) {
        (crate::MtpMode::Off, _) | (_, None) => MtpPlan::Off,
        (crate::MtpMode::Auto, Some(cap)) => MtpPlan::Mtp {
            draft_tokens: cap.recommended_draft_tokens,
        },
        (crate::MtpMode::Enabled { draft_tokens }, Some(_)) => MtpPlan::Mtp { draft_tokens },
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

    #[test]
    fn mtp_mode_resolves_against_the_advertised_head() {
        use crate::{MtpCapabilities, MtpMode};
        let cap = Some(MtpCapabilities {
            max_draft_tokens: 5,
            recommended_draft_tokens: 3,
        });
        assert_eq!(resolve_mtp_plan(MtpMode::Off, cap), MtpPlan::Off);
        assert_eq!(resolve_mtp_plan(MtpMode::Off, None), MtpPlan::Off);
        // Auto on a model without a head decodes normally and says so.
        let none = resolve_mtp_plan(MtpMode::Auto, None);
        assert_eq!(none, MtpPlan::Off);
        assert_eq!(none.proposer(), ProposerKind::None);
        assert_eq!(none.proposer().label(), "none");
        assert_eq!(none.draft_tokens(), None);
        let auto = resolve_mtp_plan(MtpMode::Auto, cap);
        assert_eq!(auto, MtpPlan::Mtp { draft_tokens: 3 });
        assert_eq!(auto.proposer(), ProposerKind::Mtp);
        assert_eq!(auto.draft_tokens(), Some(3));
        assert_eq!(
            resolve_mtp_plan(MtpMode::Enabled { draft_tokens: 5 }, cap),
            MtpPlan::Mtp { draft_tokens: 5 }
        );
        // An un-advertised Enabled is the caller's contract violation (validate rejects it
        // first); the resolver never invents a width.
        assert_eq!(
            resolve_mtp_plan(MtpMode::Enabled { draft_tokens: 5 }, None),
            MtpPlan::Off
        );
        assert_eq!(ProposerKind::default(), ProposerKind::None);
        assert_eq!(ProposerKind::Ngram.label(), "ngram");
        assert_eq!(ProposerKind::Draft.label(), "draft");
        assert_eq!(ProposerKind::Mtp.label(), "mtp");
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

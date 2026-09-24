//! Speculative-decoding shared types (epics 7253 and sc-24128).
//!
//! Speculation runs through **one** loop: the unified engine
//! ([`generate_speculative`](super::generate_speculative)) over the [`StepModel`](super::StepModel)
//! / [`DecodeCache`](crate::primitives::DecodeCache) seams, with the proposal source behind the
//! [`Proposer`](super::Proposer) trait — prompt lookup ([`NgramProposer`](super::NgramProposer)), a
//! draft model ([`DraftModelProposer`](super::DraftModelProposer)) or the native MTP head
//! ([`MtpProposer`](super::MtpProposer)), all in [`proposers`](super::proposers). The pre-epic
//! `CausalLm`-typed prompt-lookup and draft-model loops that lived here were retired when the
//! llama family moved onto the seams (story sc-24138); what remains is the counter type every
//! speculative run reports.
//!
//! ## The verify-vs-decode kernel caveat
//! Verification packs `K+1` positions into one forward, whereas plain decoding feeds one token at a
//! time. On a GPU the multi-token attention kernel rounds differently from the single-token one —
//! measured at a few bf16 ULP (~0.25 on a logit) for the *same* position — the same shape
//! non-invariance documented for the batched scheduler (story 7255). So speculative output is
//! distribution-preserving **relative to the verify forward**, and a realized greedy run *tracks*
//! (rather than bit-matches) a single-token greedy run, diverging only where that rounding flips a
//! near-tie. With no drafts the verify is itself a single-token forward, so the path is then
//! bit-identical to non-speculative decoding — the exactness gate on the loop / accept / rollback
//! logic.

/// Measured speculation efficiency for a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Target forward passes (the prefill + one per verify step, plus a replay forward per
    /// rejected run on a cache that rolls back only to a step start). Fewer than `generated` ⇒
    /// speedup.
    pub forwards: usize,
    /// The prefill forwards counted in `forwards`: one whenever the prompt was prefilled — by the
    /// engine itself ([`SpeculativePrompt::Tokens`]) or by its caller
    /// ([`SpeculativePrompt::Prefilled`], whose prefill the engine counts as one of the request's
    /// target forwards all the same). `forwards - prefill_forwards` is what the verify steps and
    /// any replays cost.
    ///
    /// [`SpeculativePrompt::Tokens`]: crate::decode::SpeculativePrompt::Tokens
    /// [`SpeculativePrompt::Prefilled`]: crate::decode::SpeculativePrompt::Prefilled
    pub prefill_forwards: usize,
    /// Draft tokens proposed across all steps.
    pub proposed: usize,
    /// Draft tokens accepted across all steps.
    pub accepted: usize,
    /// Verify steps taken: target forwards over `[cur, drafts…]` whose outcome was decided
    /// (sc-24130). The denominator of "host syncs per verify step".
    pub verify_steps: usize,
    /// Verify steps whose partial acceptance was recovered by a **direct** cache rollback to
    /// `start + 1 + accepted` (sc-24131) — no extra target forward.
    pub direct_rollbacks: usize,
    /// Replay forwards (sc-24130, E2): verify steps whose cache answered
    /// [`Error::RollbackUnavailable`] for the direct rollback to `start + 1 + accepted`, so the
    /// engine rolled back to the step start and replayed the kept prefix in one extra forward.
    /// Counted inside `forwards`; `0` on a cache with per-position rollback — the `Qwen35Cache`
    /// since its per-token checkpoint ring (sc-24131).
    ///
    /// [`Error::RollbackUnavailable`]: crate::error::Error::RollbackUnavailable
    pub replays: usize,
}

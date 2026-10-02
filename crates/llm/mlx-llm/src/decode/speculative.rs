//! Prompt-lookup (n-gram) speculative decoding (epic 7153, story 7171).
//!
//! Speculative decoding amortizes the target forward by proposing several tokens at once and
//! verifying them in **one** pass. Prompt-lookup needs no draft model: the proposer copies the
//! continuation that followed the most recent earlier occurrence of the current trailing n-gram
//! ([`core_llm::speculative::ngram_propose`]) — which is exactly right for repetitive / structured
//! output (code, JSON, schema echoes) where the model copies from its context.
//!
//! Each step runs the target once over `[cur, draft₁ … draftₖ]` (all-position logits), then the
//! backend-neutral acceptance sampler ([`core_llm::speculative`]) accepts the longest agreeing prefix
//! plus one bonus token. Rejected drafts are rolled back by truncating the KV cache. Prompt lookup
//! (sc-24434) and draft-model speculation (sc-24436) both run on the model-agnostic
//! [engine](crate::decode::engine) — this module keeps only their entry points. The committed
//! tokens are distributed exactly as the **verify forward's** distribution — the acceptance is exact
//! (proven in `core_llm::speculative`), so this changes throughput, not the sampled distribution.
//!
//! ## The verify-vs-decode kernel caveat
//! Verification packs `K+1` positions into one forward, whereas plain decoding feeds one token at a
//! time. On MLX (and any GPU backend) the multi-token attention kernel rounds differently from the
//! single-token one — measured at a few bf16 ULP (~0.25 on a logit) for the *same* position — the same
//! shape-non-invariance documented for the batched scheduler (story 7167). So speculative output is
//! distribution-preserving **relative to the verify forward**, and a realized greedy run *tracks*
//! (rather than bit-matches) a single-token greedy run, diverging only where that rounding flips a
//! near-tie. With no drafts (`num_draft = 0`) the verify is itself a single-token forward, so the path
//! is then bit-identical to non-speculative decoding — the exactness gate on the loop/accept/rollback
//! logic.

use crate::decode::cancel::CancelFlag;
use crate::decode::engine::{generate_speculative, EngineOptions, SpeculativePrompt};
use crate::decode::proposers::{DraftModelProposer, NgramProposer};
use crate::decode::stream::{GenerationConfig, GenerationOutput, StreamEvent};
use crate::error::{Error, Result};
use crate::models::CausalLm;

/// Knobs for prompt-lookup speculation.
#[derive(Clone, Copy, Debug)]
pub struct SpeculativeConfig {
    /// Longest trailing n-gram to try matching against the context (longest first).
    pub max_ngram: usize,
    /// Maximum draft tokens proposed (and verified) per step.
    pub num_draft: usize,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            max_ngram: 3,
            num_draft: 4,
        }
    }
}

/// Measured speculation efficiency for a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeculativeStats {
    /// Target forward passes (the prefill, one per verify step, and any recovery replays). Fewer
    /// than `generated` ⇒ speedup.
    pub forwards: usize,
    /// Draft tokens proposed across all steps.
    pub proposed: usize,
    /// Draft tokens accepted across all steps.
    pub accepted: usize,
    /// Target verification passes (every decode step after the first token is one).
    pub verify_steps: usize,
    /// Verify steps recovered by restoring the step start and replaying the kept prefix.
    pub replays: usize,
    /// Partially accepted verify steps recovered by a direct cache rollback (no forward).
    pub direct_rollbacks: usize,
    /// Decode steps the pipelined loop enqueued before the previous token was read back (story
    /// sc-24439); `0` for a run that was not pipelined.
    pub pipelined: usize,
    /// Pipelined look-ahead forwards discarded unread because the token before them ended the run
    /// (at most one per run). `forwards` counts them: forwards = prefill + verify steps + replays +
    /// discarded.
    pub discarded: usize,
    /// Tokens generated when `auto`'s acceptance monitor demoted the run to token-at-a-time
    /// decoding (sc-24446); `None` when it was not demoted.
    pub demoted_at: Option<usize>,
}

/// Generate from `prompt_ids` with prompt-lookup speculative decoding, returning the output and
/// [`SpeculativeStats`] — the [engine](crate::decode::engine) with an
/// [`NgramProposer`] (epic sc-24432, story sc-24434). The
/// output is the **same** as [`generate`](crate::decode::generate) for the same prompt+config —
/// token-for-token identical under greedy (modulo the verify-kernel caveat above);
/// distribution-preserving under sampling.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference.
pub fn generate_prompt_lookup(
    model: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    spec: &SpeculativeConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<(GenerationOutput, SpeculativeStats)> {
    let run = generate_speculative(
        model,
        &mut NgramProposer {
            max_ngram: spec.max_ngram,
        },
        SpeculativePrompt::Tokens(prompt_ids),
        config,
        spec.num_draft,
        cancel,
        on_event,
        EngineOptions::default(),
    )?;
    Ok((run.output, run.stats))
}

/// Generate from `prompt_ids` with **draft-model** speculative decoding: the small `draft` model
/// proposes tokens which the big `target` verifies in one forward (epic 7153, story 7172) — the
/// [engine](crate::decode::engine) with a [`DraftModelProposer`] (epic sc-24432, story sc-24436),
/// which replaced this function's own loop. Returns the output and [`SpeculativeStats`].
///
/// `draft` and `target` must be vocab-compatible (same `vocab_size`); otherwise returns an error.
/// Output is distribution-preserving w.r.t. the target's verify forward — token-for-token identical to
/// non-speculative with `num_draft = 0` (the exactness gate); see the module-level kernel caveat for
/// the multi-token verify.
pub fn generate_draft_speculative(
    target: &CausalLm,
    draft: &CausalLm,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    spec: &SpeculativeConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<(GenerationOutput, SpeculativeStats)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if target.config().vocab_size != draft.config().vocab_size {
        return Err(Error::Msg(format!(
            "draft/target vocab mismatch: draft {} vs target {}",
            draft.config().vocab_size,
            target.config().vocab_size
        )));
    }
    let run = generate_speculative(
        target,
        &mut DraftModelProposer::new(draft, spec.num_draft),
        SpeculativePrompt::Tokens(prompt_ids),
        config,
        spec.num_draft,
        cancel,
        on_event,
        EngineOptions::default(),
    )?;
    Ok((run.output, run.stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::engine::tests::causal;
    use crate::decode::generate;

    /// Draft-model speculation draws by the shared rule every MLX decode loop uses: with no
    /// drafts (`num_draft = 0`, the exactness gate) a seeded temperature-0.8 / top-p-0.9 run is the
    /// plain loop's, first token and all. (Top-p is what separates the rules: the heap-order
    /// reference walks the nucleus heaviest-first, the shared sampler in vocabulary order.)
    #[test]
    fn draft_speculation_draws_by_the_shared_sampler() {
        let (target, draft) = (causal(), causal());
        let prompt = [3, 9, 4, 11, 3, 9, 4, 11];
        for seed in [7, 8, 9] {
            let config = GenerationConfig {
                max_new_tokens: 12,
                sampling: crate::primitives::sampler::SamplingParams {
                    temperature: 0.8,
                    top_p: 0.9,
                    ..Default::default()
                },
                seed: Some(seed),
                stop_tokens: Vec::new(),
            };
            let expected =
                generate(&target, &prompt, &config, &CancelFlag::new(), &mut |_| {}).unwrap();
            let (out, _) = generate_draft_speculative(
                &target,
                &draft,
                &prompt,
                &config,
                &SpeculativeConfig {
                    max_ngram: 3,
                    num_draft: 0,
                },
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(
                out.tokens[0], expected.tokens[0],
                "seed {seed}: first token"
            );
            assert_eq!(out.tokens, expected.tokens, "seed {seed}");
        }
    }
}

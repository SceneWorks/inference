//! Prompt-lookup (n-gram) speculative decoding (epic 7253, story 7259).
//!
//! Speculative decoding amortizes the target forward by proposing several tokens at once and
//! verifying them in **one** pass. Prompt-lookup needs no draft model: the proposer copies the
//! continuation that followed the most recent earlier occurrence of the current trailing n-gram
//! ([`core_llm::speculative::ngram_propose`]) — which is exactly right for repetitive / structured
//! output (code, JSON, schema echoes) where the model copies from its context.
//!
//! Each step runs the target once over `[cur, draft₁ … draftₖ]` ([`decode_logits_all`]), then the
//! backend-neutral acceptance sampler ([`core_llm::speculative`]) accepts the longest agreeing prefix
//! plus one bonus token. Rejected drafts are rolled back via [`KvCache::truncate`]. The committed
//! tokens are distributed exactly as the **verify forward's** distribution — the acceptance is exact
//! (proven in `core_llm::speculative`), so this changes throughput, not the sampled distribution.
//!
//! ## The verify-vs-decode kernel caveat
//! Verification packs `K+1` positions into one forward, whereas plain decoding feeds one token at a
//! time. On a GPU the multi-token attention kernel rounds differently from the single-token one —
//! measured at a few bf16 ULP (~0.25 on a logit) for the *same* position — the same shape
//! non-invariance documented for the batched scheduler (story 7255). So speculative output is
//! distribution-preserving **relative to the verify forward**, and a realized greedy run *tracks*
//! (rather than bit-matches) a single-token greedy run, diverging only where that rounding flips a
//! near-tie. With no drafts (`num_draft = 0`) the verify is itself a single-token forward, so the path
//! is then bit-identical to non-speculative decoding — the exactness gate on the loop/accept/rollback
//! logic.
//!
//! [`decode_logits_all`]: crate::models::LlamaModel::decode_logits_all

use candle_core::Tensor;

use core_llm::speculative::{
    accept_greedy_run, accept_token, ngram_propose, sample_weighted, Acceptance,
};

use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{
    default_seed, FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::error::{Error, Result};
use crate::models::LlamaModel;
use crate::primitives::input_ids;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::sampler::{sample, shaped_candidates, SplitMix64, TokenRng};

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
    /// Target forward passes (the prefill + one per verify step). Fewer than `generated` ⇒ speedup.
    pub forwards: usize,
    /// Draft tokens proposed across all steps.
    pub proposed: usize,
    /// Draft tokens accepted across all steps.
    pub accepted: usize,
}

/// Generate from `prompt_ids` with prompt-lookup speculative decoding, returning the output and
/// [`SpeculativeStats`]. The output is the **same** as [`generate`](crate::decode::generate) for the
/// same prompt+config — token-for-token identical under greedy with `num_draft = 0`, and
/// distribution-preserving otherwise (see the module-level kernel caveat).
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference.
pub fn generate_prompt_lookup(
    model: &LlamaModel,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    spec: &SpeculativeConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<(GenerationOutput, SpeculativeStats)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg("generate_prompt_lookup: empty prompt".into()));
    }

    let mut stats = SpeculativeStats::default();
    let mut generated: Vec<i32> = Vec::new();
    let mut finish = FinishReason::MaxTokens;

    if config.max_new_tokens == 0 {
        on_event(StreamEvent::Done {
            reason: finish,
            generated: 0,
        });
        return Ok((
            GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            stats,
        ));
    }

    let mut rng = SplitMix64::new(config.seed.unwrap_or_else(default_seed));
    let device = model.device();
    let mut cache = model.new_cache();
    let greedy = config.sampling.temperature <= 0.0;

    // ---- Prefill: logits for the last prompt position; the first token is sampled as usual. ----
    let logits_last = model.decode_logits(&input_ids(prompt_ids, device)?, &mut cache, 0)?;
    stats.forwards += 1;
    let mut history: Vec<i32> = prompt_ids.to_vec();

    let first = sample(&logits_last, &history, &config.sampling, &mut rng, None)?;
    if config.stop_tokens.contains(&first) {
        finish = FinishReason::StopToken;
        on_event(StreamEvent::Done {
            reason: finish,
            generated: 0,
        });
        return Ok((
            GenerationOutput {
                tokens: generated,
                finish_reason: finish,
            },
            stats,
        ));
    }
    on_event(StreamEvent::Token { id: first, step: 0 });
    generated.push(first);
    history.push(first);
    let mut cur = first; // last committed token, not yet in the cache (cache holds the prompt)

    // ---- Speculative steps: propose after `cur`, verify in one pass, accept a prefix + bonus. ----
    'outer: while generated.len() < config.max_new_tokens {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }

        // Propose drafts; cap so the commit (≤ accepted + 1) cannot overrun the budget.
        let remaining = config.max_new_tokens - generated.len();
        let k_cap = spec.num_draft.min(remaining.saturating_sub(1));
        let drafts = if k_cap == 0 {
            Vec::new()
        } else {
            ngram_propose(&history, spec.max_ngram, k_cap)
        };
        stats.proposed += drafts.len();

        // One target forward over [cur, drafts…]; logits_all[i] predicts the token after verify[i].
        let mut verify = Vec::with_capacity(1 + drafts.len());
        verify.push(cur);
        verify.extend_from_slice(&drafts);
        let base_offset = cache.offset();
        let logits_all =
            model.decode_logits_all(&input_ids(&verify, device)?, &mut cache, base_offset)?;
        stats.forwards += 1;

        let (committed, accepted) = if greedy {
            decide_greedy(&logits_all, &drafts, &history, config, &mut rng)?
        } else {
            decide_stochastic(
                &logits_all,
                &drafts,
                &point_mass_dists(&drafts),
                &history,
                config,
                &mut rng,
            )?
        };
        stats.accepted += accepted;

        // Roll the cache back to keep only `cur` + the accepted drafts; rejected-draft KV is dropped.
        cache.truncate(base_offset + 1 + accepted as i32)?;

        // Commit, honoring stop tokens and the budget; `cur` advances to the last committed token.
        for &t in &committed {
            if config.stop_tokens.contains(&t) {
                finish = FinishReason::StopToken;
                break 'outer;
            }
            on_event(StreamEvent::Token {
                id: t,
                step: generated.len(),
            });
            generated.push(t);
            history.push(t);
            cur = t;
            if generated.len() >= config.max_new_tokens {
                finish = FinishReason::MaxTokens;
                break 'outer;
            }
        }
    }

    on_event(StreamEvent::Done {
        reason: finish,
        generated: generated.len(),
    });
    Ok((
        GenerationOutput {
            tokens: generated,
            finish_reason: finish,
        },
        stats,
    ))
}

/// Greedy acceptance: the target's argmax at each verify position (penalty-aware, via the sampler),
/// accept the longest matching draft prefix, bonus = the argmax at the divergence point. Returns
/// `(committed tokens, accepted draft count)`.
pub(crate) fn decide_greedy(
    logits_all: &Tensor,
    drafts: &[i32],
    history: &[i32],
    config: &GenerationConfig,
    rng: &mut SplitMix64,
) -> Result<(Vec<i32>, usize)> {
    let m = logits_all.dim(1)?; // verify positions = 1 + drafts.len()
    let mut target_argmax = Vec::with_capacity(m);
    let mut hist_i = history.to_vec();
    for i in 0..m {
        let row = logits_row(logits_all, i)?;
        // Greedy ⇒ `sample` returns the (penalty-aware) argmax; rng is untouched.
        target_argmax.push(sample(&row, &hist_i, &config.sampling, rng, None)?);
        if i < drafts.len() {
            hist_i.push(drafts[i]);
        }
    }
    let accepted = accept_greedy_run(&target_argmax, drafts);
    let mut committed = drafts[..accepted].to_vec();
    committed.push(target_argmax[accepted]); // bonus
    Ok((committed, accepted))
}

/// Stochastic acceptance: per-position rejection sampling against the target's shaped distribution,
/// distribution-preserving. `draft_dists[i]` is the proposal distribution `q` the draft sampled
/// `drafts[i]` from — a point mass `[(drafts[i], 1.0)]` for prompt-lookup, the draft model's shaped
/// distribution for draft-model speculation. Returns `(committed tokens, accepted draft count)`.
pub(crate) fn decide_stochastic(
    logits_all: &Tensor,
    drafts: &[i32],
    draft_dists: &[Vec<(i32, f32)>],
    history: &[i32],
    config: &GenerationConfig,
    rng: &mut SplitMix64,
) -> Result<(Vec<i32>, usize)> {
    let mut committed = Vec::new();
    let mut accepted = 0usize;
    let mut hist_i = history.to_vec();
    let mut rejected = false;

    for (i, &d) in drafts.iter().enumerate() {
        let row = logits_row(logits_all, i)?;
        let target = shaped_candidates(&row, &hist_i, &config.sampling, None)?;
        let (u_a, u_r) = (rng.next_f32(), rng.next_f32());
        match accept_token(&target, &draft_dists[i], d, u_a, u_r) {
            Acceptance::Accepted(t) => {
                committed.push(t);
                accepted += 1;
                hist_i.push(t);
            }
            Acceptance::Rejected(bonus) => {
                committed.push(bonus);
                rejected = true;
                break;
            }
        }
    }
    if !rejected {
        // Every draft accepted ⇒ draw the bonus from the position past the last draft.
        let row = logits_row(logits_all, drafts.len())?;
        let target = shaped_candidates(&row, &hist_i, &config.sampling, None)?;
        committed.push(sample_weighted(&target, rng.next_f32(), 0));
    }
    Ok((committed, accepted))
}

/// Point-mass proposal distributions for prompt-lookup drafts (each draft was "sampled" with
/// probability 1).
pub(crate) fn point_mass_dists(drafts: &[i32]) -> Vec<Vec<(i32, f32)>> {
    drafts.iter().map(|&d| vec![(d, 1.0)]).collect()
}

/// Extract position `i`'s logits row `[batch, vocab]` from an all-positions `[batch, seq, vocab]`.
fn logits_row(all: &Tensor, i: usize) -> Result<Tensor> {
    let (b, _s, vocab) = all.dims3()?;
    Ok(all.narrow(1, i, 1)?.reshape((b, vocab))?)
}

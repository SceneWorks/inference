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
//! plus one bonus token. Rejected drafts are rolled back via [`KvCache::truncate`]. The committed
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

use mlx_rs::Array;

use core_llm::speculative::{
    accept_greedy_run, accept_token, ngram_propose, sample_weighted, Acceptance,
};

use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{
    default_seed, FinishReason, GenerationConfig, GenerationOutput, StreamEvent,
};
use crate::decode::BufferRelease;
use crate::error::{Error, Result};
use crate::models::CausalLm;
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
/// same prompt+config — token-for-token identical under greedy; distribution-preserving under
/// sampling.
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
    let mut cache = model.new_cache();
    generate_prompt_lookup_on(
        model, &mut cache, prompt_ids, config, spec, cancel, on_event,
    )
}

/// [`generate_prompt_lookup`] on a caller-chosen empty `cache` — a paged compressed cache
/// included (sc-20681). Each verify forward is preceded by [`KvCache::begin_speculation`], so the
/// rollback of rejected drafts leaves the cache exactly as if only the kept tokens had been
/// appended, on a quantizing cache too.
pub fn generate_prompt_lookup_on(
    model: &CausalLm,
    cache: &mut dyn KvCache,
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
    if cache.offset() != 0 {
        return Err(Error::Msg(
            "generate_prompt_lookup: the cache must start empty".into(),
        ));
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
    let greedy = config.sampling.temperature <= 0.0;

    // ---- Prefill: logits for the last prompt position; the first token is sampled as usual. ----
    let logits_last = model.decode_logits(&input_ids(prompt_ids), cache, 0)?;
    stats.forwards += 1;
    let mut history: Vec<i32> = prompt_ids.to_vec();

    let first = sample(&logits_last, &history, &config.sampling, &mut rng, None)?;
    // That sample evaluated the (lazy) prefill graph. `logits_last` would otherwise live to the
    // end of the function, so retire it explicitly; the release is taken on the loop's first
    // `advance`, once step 0 has also retired its own transients.
    drop(logits_last);
    let mut release = BufferRelease::new();
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
        cache.begin_speculation()?;
        let logits_all = model.decode_logits_all(&input_ids(&verify), cache, base_offset)?;
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
        release.advance(committed.len());

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

/// Generate from `prompt_ids` with **draft-model** speculative decoding: the small `draft` model
/// proposes tokens which the big `target` verifies in one forward (epic 7153, story 7172). Shares the
/// verify / accept / KV-rollback machinery with [`generate_prompt_lookup`]; only the proposer differs
/// (a draft model with its own distribution `q`, rather than n-gram copies). Returns the output and
/// [`SpeculativeStats`].
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
    let mut target_cache = target.new_cache();
    let mut draft_cache = draft.new_cache();
    generate_draft_speculative_on(
        target,
        &mut target_cache,
        draft,
        &mut draft_cache,
        prompt_ids,
        config,
        spec,
        cancel,
        on_event,
    )
}

/// [`generate_draft_speculative`] on caller-chosen empty caches — one per model, never shared
/// (each holds its own model's K/V) — including paged compressed caches (sc-20681). Both caches
/// arm [`KvCache::begin_speculation`] before each speculative step, so their rollbacks are exact.
#[allow(clippy::too_many_arguments)]
pub fn generate_draft_speculative_on(
    target: &CausalLm,
    target_cache: &mut dyn KvCache,
    draft: &CausalLm,
    draft_cache: &mut dyn KvCache,
    prompt_ids: &[i32],
    config: &GenerationConfig,
    spec: &SpeculativeConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(StreamEvent),
) -> Result<(GenerationOutput, SpeculativeStats)> {
    if cancel.is_cancelled() {
        return Err(Error::Canceled);
    }
    if target_cache.offset() != 0 || draft_cache.offset() != 0 {
        return Err(Error::Msg(
            "generate_draft_speculative: both caches must start empty".into(),
        ));
    }
    if prompt_ids.is_empty() {
        return Err(Error::Msg(
            "generate_draft_speculative: empty prompt".into(),
        ));
    }
    if target.config().vocab_size != draft.config().vocab_size {
        return Err(Error::Msg(format!(
            "draft/target vocab mismatch: draft {} vs target {}",
            draft.config().vocab_size,
            target.config().vocab_size
        )));
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
    let greedy = config.sampling.temperature <= 0.0;

    // ---- Prefill both models; first token from the target's last-position logits. ----
    let logits_last = target.decode_logits(&input_ids(prompt_ids), target_cache, 0)?;
    let draft_logits_last = draft.decode_logits(&input_ids(prompt_ids), draft_cache, 0)?;
    stats.forwards += 1;
    let mut history: Vec<i32> = prompt_ids.to_vec();

    let first = sample(&logits_last, &history, &config.sampling, &mut rng, None)?;
    // The sample evaluated the target's prefill graph; the draft's is only pulled by its first
    // draft step, so force it here and retire both prefills' logits together — each would
    // otherwise live to the end of the function. The release itself is taken on the loop's first
    // `advance`, once step 0 has retired its own transients too.
    mlx_rs::transforms::eval([&draft_logits_last])?;
    drop(draft_logits_last);
    drop(logits_last);
    let mut release = BufferRelease::new();
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
    let mut cur = first;

    'outer: while generated.len() < config.max_new_tokens {
        if cancel.is_cancelled() {
            finish = FinishReason::Cancelled;
            break;
        }
        let remaining = config.max_new_tokens - generated.len();
        let k = spec.num_draft.min(remaining.saturating_sub(1));
        let base_target = target_cache.offset();
        let base_draft = draft_cache.offset();
        target_cache.begin_speculation()?;
        draft_cache.begin_speculation()?;

        // ---- Draft proposes K tokens; feed [cur, d₁…dₖ] so the draft cache stays target-synced. ----
        let mut drafts: Vec<i32> = Vec::with_capacity(k);
        let mut draft_dists: Vec<Vec<(i32, f32)>> = Vec::with_capacity(k);
        let mut draft_hist = history.clone();
        let mut feed = cur;
        for step in 0..=k {
            let off = draft_cache.offset();
            let dl = draft.decode_logits(&input_ids(&[feed]), draft_cache, off)?;
            if step < k {
                if !greedy {
                    draft_dists.push(shaped_candidates(&dl, &draft_hist, &config.sampling, None)?);
                }
                let d = sample(&dl, &draft_hist, &config.sampling, &mut rng, None)?;
                drafts.push(d);
                draft_hist.push(d);
                feed = d;
            }
        }
        stats.proposed += drafts.len();

        // ---- Target verifies [cur, drafts…] in one forward. ----
        let mut verify = Vec::with_capacity(1 + drafts.len());
        verify.push(cur);
        verify.extend_from_slice(&drafts);
        let logits_all =
            target.decode_logits_all(&input_ids(&verify), target_cache, base_target)?;
        stats.forwards += 1;

        let (committed, accepted) = if greedy {
            decide_greedy(&logits_all, &drafts, &history, config, &mut rng)?
        } else {
            decide_stochastic(
                &logits_all,
                &drafts,
                &draft_dists,
                &history,
                config,
                &mut rng,
            )?
        };
        stats.accepted += accepted;

        // Roll both caches back to keep `cur` + the accepted drafts.
        target_cache.truncate(base_target + 1 + accepted as i32)?;
        draft_cache.truncate(base_draft + 1 + accepted as i32)?;
        release.advance(committed.len());

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
    logits_all: &Array,
    drafts: &[i32],
    history: &[i32],
    config: &GenerationConfig,
    rng: &mut SplitMix64,
) -> Result<(Vec<i32>, usize)> {
    let m = logits_all.shape()[1];
    let mut target_argmax = Vec::with_capacity(m as usize);
    let mut hist_i = history.to_vec();
    for i in 0..m {
        let row = logits_row(logits_all, i)?;
        // Greedy ⇒ `sample` returns the (penalty-aware) argmax; rng is untouched.
        target_argmax.push(sample(&row, &hist_i, &config.sampling, rng, None)?);
        if (i as usize) < drafts.len() {
            hist_i.push(drafts[i as usize]);
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
    logits_all: &Array,
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
        let row = logits_row(logits_all, i as i32)?;
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
        let row = logits_row(logits_all, drafts.len() as i32)?;
        let target = shaped_candidates(&row, &hist_i, &config.sampling, None)?;
        committed.push(sample_weighted(&target, rng.next_f32(), 0));
    }
    Ok((committed, accepted))
}

/// Point-mass proposal distributions for prompt-lookup drafts (each draft was "sampled" with
/// probability 1).
fn point_mass_dists(drafts: &[i32]) -> Vec<Vec<(i32, f32)>> {
    drafts.iter().map(|&d| vec![(d, 1.0)]).collect()
}

/// Extract position `i`'s logits row `[batch, vocab]` from an all-positions `[batch, seq, vocab]`.
pub(crate) fn logits_row(all: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    let sh = all.shape();
    Ok(all.take_axis(&idx, 1)?.reshape(&[sh[0], sh[2]])?)
}

/// sc-20681: speculation on paged compressed caches, on synthetic models.
#[cfg(all(test, target_os = "macos"))]
mod compressed_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::decode::stream::generate_with_cache;
    use crate::primitives::{
        CompiledKernelHandle, PackedCodeBits, PackedPagePool, PagedPackedKvCache,
        PACKED_METAL_QUANT_GROUP_SIZE,
    };

    const PAGE_TOKENS: usize = 32;

    fn paged(
        model: &CausalLm,
        reader: &CompiledKernelHandle,
    ) -> (PagedPackedKvCache, Rc<RefCell<PackedPagePool>>) {
        let cfg = model.config();
        let pool = PackedPagePool::new(
            cfg.num_layers,
            cfg.num_kv_heads as usize,
            cfg.head_dim as usize,
            PAGE_TOKENS,
            PackedCodeBits::Eight,
        )
        .unwrap();
        (
            PagedPackedKvCache::with_pool(pool.clone(), reader.clone()).unwrap(),
            pool,
        )
    }

    /// The cache holds no page beyond its quantized extent: every rejected draft's page went back.
    fn assert_no_stray_pages(cache: &PagedPackedKvCache, pool: &Rc<RefCell<PackedPagePool>>) {
        let quantized =
            cache.offset() as usize / PACKED_METAL_QUANT_GROUP_SIZE * PACKED_METAL_QUANT_GROUP_SIZE;
        assert_eq!(cache.page_ids().len(), quantized.div_ceil(PAGE_TOKENS));
        assert_eq!(pool.borrow().live_pages(), cache.page_ids().len());
    }

    fn config() -> GenerationConfig {
        GenerationConfig {
            max_new_tokens: 48,
            sampling: Default::default(), // greedy
            seed: Some(0),
            stop_tokens: Vec::new(),
        }
    }

    fn prompt() -> Vec<i32> {
        (0..150).map(|i| [3, 9, 14, 27, 5, 11][i % 6]).collect()
    }

    /// Prompt-lookup speculation on a paged compressed cache: with no drafts it is the plain
    /// compressed decode token for token; with drafts it proposes, accepts and rejects (each
    /// rejection a rollback across group and page boundaries, byte-exact by
    /// `paged_packed_kv`'s `speculative_rollback_is_byte_and_position_exact`), ends holding exactly
    /// the prompt and every committed token but the last (position-exact), and leaves no page
    /// past its quantized extent. (A verify forward rounds differently from single-token decode —
    /// the module's kernel caveat — so drafted tokens track rather than equal the plain decode.)
    #[test]
    fn prompt_lookup_speculation_runs_on_compressed_pages() {
        let model = crate::provider::tests::tiny_causal_model(4, 2, 64);
        let reader = crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap();
        let (mut plain, _) = paged(&model, &reader);
        let reference = generate_with_cache(
            &model,
            &prompt(),
            &mut plain,
            &config(),
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        let speculate = |num_draft: usize| {
            let (mut cache, pool) = paged(&model, &reader);
            let (output, stats) = generate_prompt_lookup_on(
                &model,
                &mut cache,
                &prompt(),
                &config(),
                &SpeculativeConfig {
                    max_ngram: 3,
                    num_draft,
                },
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
            assert_no_stray_pages(&cache, &pool);
            assert_eq!(
                cache.offset() as usize,
                prompt().len() + output.tokens.len() - 1,
                "the cache holds the prompt and every fed token"
            );
            (output, stats)
        };
        let (zero, _) = speculate(0);
        assert_eq!(zero.tokens, reference.tokens, "no drafts: the plain decode");
        let (drafted, stats) = speculate(4);
        eprintln!("prompt lookup on compressed pages: {stats:?}");
        assert!(stats.accepted > 0, "{stats:?}");
        assert!(
            stats.accepted < stats.proposed,
            "some drafts were rolled back: {stats:?}"
        );
        assert_eq!(drafted.tokens.len(), reference.tokens.len());
    }

    /// Prompt-lookup speculation on the contiguous compressed cache (sc-20681): it stays compressed
    /// (each rejection an exact windowed rollback, `packed_group_affine_kv`'s
    /// `speculative_rollback_on_the_contiguous_cache_is_exact_and_stays_compressed`), with no
    /// drafts it is the plain compressed decode token for token, with drafts it accepts and
    /// rejects, and it ends holding exactly the prompt and every committed token but the last.
    #[test]
    fn prompt_lookup_speculation_runs_on_the_contiguous_compressed_cache() {
        let model = crate::provider::tests::tiny_causal_model(4, 2, 64);
        let reader = crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap();
        let compressed = || {
            let (cache, refused) = crate::kv_policy::select_compressed_cache(
                &model,
                reader.clone(),
                prompt().len(),
                crate::provider::tests::admit_any_transition(),
            );
            assert_eq!(refused, None, "the request starts on the compressed cache");
            cache
        };
        let mut plain = compressed();
        let reference = generate_with_cache(
            &model,
            &prompt(),
            plain.as_mut(),
            &config(),
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        let speculate = |num_draft: usize| {
            let mut cache = compressed();
            let (output, stats) = generate_prompt_lookup_on(
                &model,
                cache.as_mut(),
                &prompt(),
                &config(),
                &SpeculativeConfig {
                    max_ngram: 3,
                    num_draft,
                },
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(
                cache.offset() as usize,
                prompt().len() + output.tokens.len() - 1,
                "the cache holds the prompt and every fed token"
            );
            let report = crate::kv_policy::compressed_report(
                core_llm::KvCompressionFormat::GroupAffineK8V8,
                None,
                cache.as_ref(),
            )
            .unwrap();
            assert!(report.ran_compressed(), "{report:?}");
            assert_eq!(report.fallback, None, "{report:?}");
            (output, stats)
        };
        let (zero, _) = speculate(0);
        assert_eq!(zero.tokens, reference.tokens, "no drafts: the plain decode");
        let (drafted, stats) = speculate(4);
        assert!(stats.accepted > 0, "{stats:?}");
        assert!(
            stats.accepted < stats.proposed,
            "some drafts were rolled back: {stats:?}"
        );
        assert_eq!(drafted.tokens.len(), reference.tokens.len());
    }

    /// Records whether every rollback truncation lands inside an armed speculation window.
    struct Spy {
        inner: crate::primitives::kv_cache::ContiguousKvCache,
        armed_at: Option<i32>,
        rollbacks: usize,
        unarmed_rollbacks: usize,
    }

    impl Spy {
        fn new(model: &CausalLm) -> Self {
            Self {
                inner: model.new_cache(),
                armed_at: None,
                rollbacks: 0,
                unarmed_rollbacks: 0,
            }
        }
    }

    impl KvCache for Spy {
        fn update(&mut self, layer: usize, keys: &Array, values: &Array) -> Result<(Array, Array)> {
            self.inner.update(layer, keys, values)
        }
        fn offset(&self) -> i32 {
            self.inner.offset()
        }
        fn batch_size(&self) -> i32 {
            self.inner.batch_size()
        }
        fn num_layers(&self) -> usize {
            self.inner.num_layers()
        }
        fn retain_sequences(&mut self, keep: &[i32]) -> Result<()> {
            self.inner.retain_sequences(keep)
        }
        fn begin_speculation(&mut self) -> Result<()> {
            self.armed_at = Some(self.inner.offset());
            Ok(())
        }
        fn truncate(&mut self, len: i32) -> Result<()> {
            if len < self.inner.offset() {
                self.rollbacks += 1;
                if self.armed_at.is_none_or(|base| len < base) {
                    self.unarmed_rollbacks += 1;
                }
            }
            self.armed_at = None;
            self.inner.truncate(len)
        }
        fn reset(&mut self) -> Result<()> {
            self.inner.reset()
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }

    /// Both speculative loops arm exact rollback before every speculative step: each truncation
    /// that drops positions lands inside a window armed at or below its length.
    #[test]
    fn speculative_loops_arm_exact_rollback_before_every_truncation() {
        let target = crate::provider::tests::tiny_causal_model(4, 2, 64);
        let draft = crate::provider::tests::tiny_causal_model(2, 1, 64);
        let spec = SpeculativeConfig {
            max_ngram: 3,
            num_draft: 4,
        };
        let mut lookup = Spy::new(&target);
        generate_prompt_lookup_on(
            &target,
            &mut lookup,
            &prompt(),
            &config(),
            &spec,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        let (mut target_spy, mut draft_spy) = (Spy::new(&target), Spy::new(&draft));
        generate_draft_speculative_on(
            &target,
            &mut target_spy,
            &draft,
            &mut draft_spy,
            &prompt(),
            &config(),
            &spec,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        for spy in [&lookup, &target_spy, &draft_spy] {
            assert!(spy.rollbacks > 0);
            assert_eq!(spy.unarmed_rollbacks, 0);
        }
    }

    /// Draft-model speculation with the target and the draft each on its own paged compressed
    /// cache and pool (never shared): a draft that disagrees with the target rolls both caches
    /// back every step, both end position-exact (prompt + every fed token), and neither pool keeps
    /// a page past its cache's quantized extent.
    #[test]
    fn draft_speculation_runs_target_and_draft_on_separate_compressed_caches() {
        let target = crate::provider::tests::tiny_causal_model(4, 2, 64);
        let draft = crate::provider::tests::tiny_causal_model(2, 1, 64);
        let reader = crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap();
        let (mut plain, _) = paged(&target, &reader);
        let reference = generate_with_cache(
            &target,
            &prompt(),
            &mut plain,
            &config(),
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        let (mut target_cache, target_pool) = paged(&target, &reader);
        let (mut draft_cache, draft_pool) = paged(&draft, &reader);
        assert!(!Rc::ptr_eq(&target_pool, &draft_pool));
        let (output, stats) = generate_draft_speculative_on(
            &target,
            &mut target_cache,
            &draft,
            &mut draft_cache,
            &prompt(),
            &config(),
            &SpeculativeConfig {
                max_ngram: 3,
                num_draft: 3,
            },
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        eprintln!("draft speculation on compressed pages: {stats:?}");
        assert!(stats.proposed > stats.accepted, "{stats:?}");
        assert_eq!(output.tokens.len(), reference.tokens.len());
        for cache in [&target_cache, &draft_cache] {
            assert_eq!(
                cache.offset() as usize,
                prompt().len() + output.tokens.len() - 1
            );
        }
        assert_no_stray_pages(&target_cache, &target_pool);
        assert_no_stray_pages(&draft_cache, &draft_pool);
    }
}

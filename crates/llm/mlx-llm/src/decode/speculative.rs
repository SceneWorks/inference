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
use crate::decode::engine::{
    generate_speculative, BorrowedCache, DynCacheTarget, EngineOptions, SpeculativePrompt,
};
use crate::decode::proposers::{DraftModelProposer, NgramProposer};
use crate::decode::stream::{GenerationConfig, GenerationOutput, StreamEvent};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::primitives::input_ids;
use crate::primitives::kv_cache::KvCache;

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
    /// The last window `auto`'s acceptance monitor judged (sc-24446); `None` when no monitor ran
    /// or the run ended before its first window.
    pub monitor: Option<core_llm::MonitorDecision>,
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

/// [`generate_prompt_lookup`] on a caller-chosen empty `cache` — a contiguous or paged compressed
/// cache included (sc-20681) — through the same [engine](crate::decode::engine)
/// ([`DynCacheTarget`]). Each verify forward is preceded by [`KvCache::begin_speculation`], so the
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
    let mut cache: Box<dyn KvCache + '_> = Box::new(BorrowedCache(cache));
    let logits = model.decode_logits(&input_ids(prompt_ids), cache.as_mut(), 0)?;
    let run = generate_speculative(
        &DynCacheTarget(model),
        &mut NgramProposer {
            max_ngram: spec.max_ngram,
        },
        SpeculativePrompt::Prefilled {
            cache: &mut cache,
            logits,
            hidden: None,
            history: prompt_ids,
            position_delta: 0,
        },
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
    check_draft_vocab(target, draft)?;
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

/// [`generate_draft_speculative`] on caller-chosen empty caches — one per model, never shared
/// (each holds its own model's K/V) — including contiguous and paged compressed caches
/// (sc-20681), through the same [engine](crate::decode::engine) ([`DynCacheTarget`] for both).
/// Both caches arm [`KvCache::begin_speculation`] at each speculative step's start, so their
/// rollbacks are exact.
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
    check_draft_vocab(target, draft)?;
    let mut target_cache: Box<dyn KvCache + '_> = Box::new(BorrowedCache(target_cache));
    let draft_cache: Box<dyn KvCache + '_> = Box::new(BorrowedCache(draft_cache));
    let draft_target = DynCacheTarget(draft);
    let logits = target.decode_logits(&input_ids(prompt_ids), target_cache.as_mut(), 0)?;
    let run = generate_speculative(
        &DynCacheTarget(target),
        &mut DraftModelProposer::new(&draft_target, spec.num_draft).with_cache(draft_cache),
        SpeculativePrompt::Prefilled {
            cache: &mut target_cache,
            logits,
            hidden: None,
            history: prompt_ids,
            position_delta: 0,
        },
        config,
        spec.num_draft,
        cancel,
        on_event,
        EngineOptions::default(),
    )?;
    Ok((run.output, run.stats))
}

/// `draft` and `target` must score the same token ids.
fn check_draft_vocab(target: &CausalLm, draft: &CausalLm) -> Result<()> {
    if target.config().vocab_size != draft.config().vocab_size {
        return Err(Error::Msg(format!(
            "draft/target vocab mismatch: draft {} vs target {}",
            draft.config().vocab_size,
            target.config().vocab_size
        )));
    }
    Ok(())
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

/// sc-20681: speculation on paged compressed caches, on synthetic models.
#[cfg(all(test, target_os = "macos"))]
mod compressed_tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use mlx_rs::Array;

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
    /// back every step, the target ends position-exact (prompt + every fed token) and the draft
    /// on a prefix of it, and neither pool keeps a page past its cache's quantized extent.
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
        let committed = prompt().len() + output.tokens.len() - 1;
        assert_eq!(target_cache.offset() as usize, committed);
        // The engine's draft proposer feeds the committed tokens its cache has not seen with its
        // next proposal (sc-24436), so the draft cache holds a prefix of the target-confirmed
        // sequence and may trail the target by the run's last unfed tokens.
        let draft_len = draft_cache.offset() as usize;
        assert!(
            (prompt().len()..=committed).contains(&draft_len),
            "draft cache {draft_len}, committed {committed}"
        );
        assert_no_stray_pages(&target_cache, &target_pool);
        assert_no_stray_pages(&draft_cache, &draft_pool);
    }
}

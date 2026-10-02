//! Iteration-level continuous batching (epic 7153, story 7281).
//!
//! [`generate_batch`](crate::decode::generate_batch) (story 7167) is *synchronous*: a batch is
//! assembled, decoded in lockstep to completion, and only then does the next batch start.
//! [`generate_continuous`] is *iteration-level*: it keeps up to `max_batch` sequences decoding at
//! once over **per-sequence** [`PagedKvCache`]s on one shared [`BlockPool`], and the moment a
//! sequence retires it prefills a waiting request into the freed slot — the batch never drains.
//! The host-side admission / retirement policy is the backend-neutral [`core_llm::Scheduler`]; this
//! module owns only the MLX tensors.
//!
//! ## Two modes — an irreducible tradeoff (see [`BatchExactness`])
//! On MLX (and any GPU backend) bit-exactness to a batch-1 run and throughput-scaling-with-occupancy
//! are **mutually exclusive**, because the throughput win of batching *is* the restructured matmul
//! reduction (amortizing the weight reads across the batch), which is exactly what perturbs the
//! floating-point result. Measured on this engine: MLX's bf16 matmul is not row-invariant (a batched
//! projection's row diverges from its batch-1 value by up to ~1 bf16 ULP on the logits — enough to
//! flip a greedy near-tie), and conversely per-sequence forwards do not overlap on the device (decode
//! tok/s is flat across occupancy). So:
//! - [`BatchExactness::Exact`] runs each sequence as its own batch-1 forward: **byte-identical** to
//!   running that request alone, but no throughput scaling.
//! - [`BatchExactness::Throughput`] batches the projections / MLP / lm_head and runs only attention
//!   per-sequence ([`CausalLm::decode_logits_per_seq`]): throughput scales with occupancy, at the
//!   cost of sub-ULP divergence (a row *tracks* its batch-1 run, like `generate_batch`).
//!
//! Both modes get iteration-level admission (admit-on-retire) and per-sequence paged attention (no
//! padding mask, no max-context reservation). The bit-exact equality assertion in `tests/batch.rs`
//! runs against `Exact`.
//!
//! ## Compressed KV (story sc-20681)
//! [`generate_continuous_kv`] runs the same loop under the product's compressed-KV policy
//! ([`ContinuousKv`]). Each request is qualified on its own
//! ([`core_llm::qualify_kv_sequence`]): a qualified request decodes on K8V8 pages of one shared
//! [`PackedPagePool`] read in place by the fused paged reader, every other request on the dense
//! paged cache beside it, and each output carries its own [`KvCacheReport`] saying which it ran
//! and why. In `Throughput` mode the compressed sequences of a step attend through one fused paged
//! dispatch per layer — a page table and per-sequence lengths, no padding mask — so jagged
//! sequences batch compressed. With a [`PagedPrefixCache`] a request starts on the longest stored
//! prefix's pages (reference-counted, copy-on-write) and stores its own sequence back; a store of
//! another cache identity is refused and never consulted.

use std::cell::RefCell;
use std::rc::Rc;

use mlx_rs::transforms::eval;
use mlx_rs::Array;

use core_llm::schedule::{Scheduler, SeqId, SeqSpec};
use core_llm::{
    FinishReason as CoreFinish, KvCacheFallbackReason, KvCacheReport, KvCompressionPolicy,
    KvModelFamily,
};

use crate::decode::batch::BatchRequest;
use crate::decode::cancel::CancelFlag;
use crate::decode::prefix::{PagedPrefixCache, PagedPrefixLookup};
use crate::decode::stream::{default_seed, FinishReason, GenerationOutput, StreamEvent};
use crate::decode::{record_lane_token, BufferRelease, LaneStep};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::nn::input_ids;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{
    BlockPool, CompiledKernelHandle, PackedCodeBits, PackedPagePool, PagedCacheIdentity,
    PagedCacheRequest, PagedCacheSelection, PagedKvCache, PagedModelKey, PagedPackedKvCache,
    PACKED_METAL_QUANT_GROUP_SIZE,
};

/// Numerical mode for [`generate_continuous`]'s decode forward — a throughput/exactness tradeoff
/// that is fundamental, not an implementation choice (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchExactness {
    /// Each sequence is decoded as its own batch-1 forward, so a row's output is **byte-identical**
    /// to running that request alone. Throughput does **not** scale with occupancy (MLX serializes
    /// the independent per-sequence forwards). The default, and the mode the bit-exact equality
    /// assertion runs against.
    #[default]
    Exact,
    /// The projections, MLP, and lm_head are **batched** over the active sequences and only attention
    /// runs per-sequence. Throughput scales with occupancy (weight reads amortized across the batch),
    /// but a row only **tracks** its batch-1 run — MLX's batched matmul is not row-invariant, so the
    /// logits diverge at sub-ULP (the same class as the `generate_batch` caveat).
    Throughput,
}

/// Configuration for [`generate_continuous`].
#[derive(Clone, Debug)]
pub struct ContinuousConfig {
    /// Maximum number of sequences decoding concurrently. Requests beyond this wait in admission
    /// order and fill a slot the instant one retires.
    pub max_batch: usize,
    /// Tokens per block in the shared paged-KV [`BlockPool`].
    pub block_size: usize,
    /// Numerical mode (see [`BatchExactness`]).
    pub exactness: BatchExactness,
}

impl Default for ContinuousConfig {
    fn default() -> Self {
        Self {
            max_batch: 8,
            block_size: 16,
            exactness: BatchExactness::Exact,
        }
    }
}

/// The compressed-KV policy of one [`generate_continuous_kv`] run (sc-20681).
pub struct ContinuousKv<'a> {
    /// The product's opt-in; [`KvCompressionPolicy::Off`] (the default) runs every request on the
    /// dense paged cache exactly as [`generate_continuous`] does.
    pub policy: KvCompressionPolicy,
    /// The loaded decoder's qualification-table family (`None` when it has none).
    pub family: Option<KvModelFamily>,
    /// Tokens per packed page (a positive multiple of the 32-token quantization group) of the
    /// run's own page pool; a prefix store brings its own pool.
    pub page_tokens: usize,
    /// The fused paged reader; `None` builds the K8V8 reader when the policy is on.
    pub reader: Option<CompiledKernelHandle>,
    /// A shared-prefix store over paged compressed KV, consulted and filled by the compressed
    /// requests when its identity matches this run's (its pool is then the run's pool).
    pub prefix: Option<&'a mut PagedPrefixCache>,
    /// The page pool to run on when no matching prefix store brings one (a long-lived caller
    /// keeps one pool across runs); `None` makes one for the run.
    pub pool: Option<Rc<RefCell<PackedPagePool>>>,
    /// The model identity (checkpoint and revision) the run's compressed caches are keyed by,
    /// with the decoder's [`CausalLm::cache_fingerprint`]. Empty means unkeyed: a prefix store
    /// is then refused, never shared.
    pub model_identity: &'a str,
    /// Per-request cancellation (one flag per request, or empty for none): a cancelled request
    /// leaves the batch — finishing [`FinishReason::Cancelled`] with its partial output and its
    /// pages released — while the others keep decoding. The run's own `cancel` stops them all.
    pub cancels: &'a [CancelFlag],
    /// Per-request opt-ins (one per request, or empty to apply [`Self::policy`] to all): a
    /// server batch mixes requests that opted in with requests that did not.
    pub policies: &'a [KvCompressionPolicy],
}

impl Default for ContinuousKv<'_> {
    fn default() -> Self {
        Self {
            policy: KvCompressionPolicy::Off,
            family: None,
            page_tokens: 64,
            reader: None,
            prefix: None,
            pool: None,
            model_identity: "",
            cancels: &[],
            policies: &[],
        }
    }
}

/// One request's result from [`generate_continuous_kv`].
#[derive(Clone, Debug)]
pub struct ContinuousOutput {
    pub output: GenerationOutput,
    /// The KV cache the request ran on: compressed, or dense with its reason.
    pub kv_cache: KvCacheReport,
    /// Prompt positions started from a stored prefix instead of prefilled.
    pub reused_prefix_tokens: usize,
    /// Why the run's prefix store was not consulted for this request (an identity mismatch).
    pub prefix_refused: Option<String>,
}

/// Per-sequence host state for one in-flight slot, alongside its own paged cache.
struct Lane {
    seq: SeqId,
    /// Request index == admission index == `seq.0`; kept explicit for the `on_event` callback.
    req_index: usize,
    selection: PagedCacheSelection,
    rng: SplitMix64,
    params: SamplingParams,
    /// Prompt + generated tokens (the repetition-penalty window the sampler reads).
    history: Vec<i32>,
    /// The token to feed at the next decode step (the most recently sampled token).
    next_token: i32,
}

impl Lane {
    fn cache(&mut self) -> &mut dyn KvCache {
        self.selection.cache_mut()
    }

    fn packed(&mut self) -> Option<&mut PagedPackedKvCache> {
        self.selection
            .cache_mut()
            .as_any_mut()
            .downcast_mut::<PagedPackedKvCache>()
    }
}

/// The caches, pools and prefix store one run draws from.
struct KvRun<'a> {
    policy: KvCompressionPolicy,
    family: Option<KvModelFamily>,
    dense_pool: Rc<RefCell<BlockPool>>,
    /// `None` when the policy is off, or when no pool fits this decoder (`pool_refusal`).
    packed_pool: Option<Rc<RefCell<PackedPagePool>>>,
    pool_refusal: Option<String>,
    reader: Option<CompiledKernelHandle>,
    /// The store and this run's identity, when they match.
    store: Option<(&'a mut PagedPrefixCache, PagedCacheIdentity)>,
    /// A store refused for its identity: the store to record each refusal on, and why.
    refused_store: Option<(&'a mut PagedPrefixCache, String)>,
    /// Per request: the report it finished with and the positions it reused.
    reports: Vec<Option<KvCacheReport>>,
    reused: Vec<usize>,
    /// Per-request cancellation flags (empty for none).
    cancels: Vec<CancelFlag>,
    /// Per-request opt-ins (empty: `policy` for all).
    policies: Vec<KvCompressionPolicy>,
}

impl<'a> KvRun<'a> {
    fn new(model: &CausalLm, config: &ContinuousConfig, kv: ContinuousKv<'a>, n: usize) -> Self {
        let cfg = model.config();
        let mut run = Self {
            policy: kv.policy,
            family: kv.family,
            dense_pool: BlockPool::new(config.block_size),
            packed_pool: None,
            pool_refusal: None,
            reader: None,
            store: None,
            refused_store: None,
            reports: vec![None; n],
            reused: vec![0; n],
            cancels: kv.cancels.to_vec(),
            policies: kv.policies.to_vec(),
        };
        let any_opted_in = kv.policy != KvCompressionPolicy::Off
            || kv.policies.contains(&KvCompressionPolicy::Qualified);
        if !any_opted_in {
            return run;
        }
        let new_pool = |page_tokens: usize| {
            PackedPagePool::new(
                cfg.num_layers,
                usize::try_from(cfg.num_kv_heads).unwrap_or(0),
                usize::try_from(cfg.head_dim).unwrap_or(0),
                page_tokens,
                PackedCodeBits::Eight,
            )
        };
        if let Some(store) = kv.prefix {
            let key = PagedModelKey::new(kv.model_identity, model.cache_fingerprint());
            let verdict = if kv.model_identity.is_empty() {
                Err(
                    "the run names no model identity, so its caches cannot be keyed for sharing"
                        .to_owned(),
                )
            } else {
                match new_pool(store.identity().page_tokens) {
                    Ok(pool) => {
                        let identity = pool.borrow().identity(&key);
                        match store.identity().mismatch(&identity) {
                            None => Ok(identity),
                            Some(mismatch) => Err(mismatch),
                        }
                    }
                    Err(error) => Err(error.to_string()),
                }
            };
            match verdict {
                Ok(identity) => {
                    run.packed_pool = Some(store.pool().clone());
                    run.store = Some((store, identity));
                }
                Err(reason) => run.refused_store = Some((store, reason)),
            }
        }
        if run.packed_pool.is_none() {
            match kv.pool.map_or_else(|| new_pool(kv.page_tokens), Ok) {
                Ok(pool) => run.packed_pool = Some(pool),
                Err(error) => run.pool_refusal = Some(error.to_string()),
            }
        }
        run.reader = kv
            .reader
            .or_else(|| crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).ok());
        run
    }

    /// The cache request `r` runs on, before any K/V mutation.
    fn select(&self, model: &CausalLm, ri: usize, r: &BatchRequest) -> PagedCacheSelection {
        let policy = self.policies.get(ri).copied().unwrap_or(self.policy);
        let prompt_tokens = u64::try_from(r.prompt_ids.len()).unwrap_or(u64::MAX);
        let max_new_tokens = u64::try_from(r.max_new_tokens).unwrap_or(u64::MAX);
        let Some(packed_pool) = self.packed_pool.as_ref() else {
            let reason = match core_llm::qualify_kv_sequence(
                policy,
                self.family,
                prompt_tokens,
                max_new_tokens,
            ) {
                Err(reason) => KvCacheReport::dense(reason, None),
                Ok(_) => KvCacheReport::dense(
                    KvCacheFallbackReason::UnsupportedGeometry,
                    self.pool_refusal.clone(),
                ),
            };
            return PagedCacheSelection::dense(
                PagedKvCache::with_pool(self.dense_pool.clone(), model.config().num_layers),
                reason,
            );
        };
        model.select_paged_cache(PagedCacheRequest {
            policy,
            family: self.family,
            prompt_tokens,
            max_new_tokens,
            dense_pool: &self.dense_pool,
            packed_pool,
            reader: self.reader.as_ref(),
        })
    }

    /// The report of a request that never ran (cancelled in the queue, or a zero budget).
    fn planned_report(
        &self,
        model: &CausalLm,
        ri: usize,
        r: &BatchRequest,
    ) -> Result<KvCacheReport> {
        self.select(model, ri, r).report()
    }

    /// Start a compressed `selection` on the longest stored prefix of `prompt`; the positions it
    /// already holds.
    fn reuse_prefix(
        &mut self,
        selection: &mut PagedCacheSelection,
        prompt: &[i32],
    ) -> Result<usize> {
        if !selection.is_compressed() {
            return Ok(0);
        }
        if let Some((store, _)) = self.refused_store.as_mut() {
            // Counted on the store; nothing is reused.
            store.record_refusal();
            return Ok(0);
        }
        let Some((store, identity)) = self.store.as_mut() else {
            return Ok(0);
        };
        match store.lookup(identity, prompt)? {
            PagedPrefixLookup::Hit { cache, tokens } => {
                selection.replace_compressed(*cache)?;
                Ok(tokens)
            }
            PagedPrefixLookup::Miss | PagedPrefixLookup::Refused(_) => Ok(0),
        }
    }

    /// Offer the sequence `lane` holds (`tokens` truncated to its offset) to the prefix store.
    fn store_sequence(&mut self, lane: &mut Lane, tokens: &[i32]) -> Result<()> {
        let Some((store, identity)) = self.store.as_mut() else {
            return Ok(());
        };
        if let Some(cache) = lane.packed() {
            store.insert(identity, tokens, cache)?;
        }
        Ok(())
    }

    /// Whether request `ri` was cancelled on its own.
    fn cancelled(&self, ri: usize) -> bool {
        self.cancels.get(ri).is_some_and(CancelFlag::is_cancelled)
    }

    fn prefix_refused(&self) -> Option<String> {
        self.refused_store
            .as_ref()
            .map(|(_, reason)| reason.clone())
    }

    /// Before a compressed `selection` prefills `prompt_len` positions: reserve its pages in one
    /// growth (the pool's live pages, the prompt's pages, and one for the first decode group).
    fn reserve_prefill(&self, selection: &PagedCacheSelection, prompt_len: usize) -> Result<()> {
        let Some(pool) = self
            .packed_pool
            .as_ref()
            .filter(|_| selection.is_compressed())
        else {
            return Ok(());
        };
        let mut pool = pool.borrow_mut();
        let pages = pool.live_pages() + prompt_len.div_ceil(pool.page_tokens()) + 1;
        pool.reserve(pages)
    }

    /// Return the pool's capacity above its highest live page (after a sequence retired).
    fn trim(&self) -> Result<()> {
        match self.packed_pool.as_ref() {
            Some(pool) => pool.borrow_mut().trim(),
            None => Ok(()),
        }
    }
}

impl Drop for KvRun<'_> {
    /// A run that ends — normally, cancelled, or on an error — leaves its pool holding only what
    /// is still live (a prefix store's entries). The sequences are dropped first (they are
    /// declared after the run), so their pages are already back.
    fn drop(&mut self) {
        if let Err(error) = self.trim() {
            eprintln!("generate_continuous: trimming the page pool failed: {error}");
        }
    }
}

/// Generate for many requests with **iteration-level continuous batching**: up to
/// `config.max_batch` sequences decode at once over per-sequence paged caches, and a retiring
/// sequence's slot is immediately refilled from the waiting requests. Returns a
/// [`GenerationOutput`] per request, in request order. `on_event` receives `(request_index, event)`
/// as each row streams.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference. A mid-stream cancel
/// stops promptly: every request still decoding **or still waiting in the queue** finishes
/// [`FinishReason::Cancelled`] with whatever partial output it had, and each request emits exactly
/// one terminal [`StreamEvent::Done`].
pub fn generate_continuous(
    model: &CausalLm,
    requests: &[BatchRequest],
    config: &ContinuousConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Vec<GenerationOutput>> {
    Ok(generate_continuous_kv(
        model,
        requests,
        config,
        ContinuousKv::default(),
        cancel,
        on_event,
    )?
    .into_iter()
    .map(|out| out.output)
    .collect())
}

/// [`generate_continuous`] under the compressed-KV policy `kv` (sc-20681; see the module docs):
/// each request runs compressed or dense on its own qualification and reports which, and
/// compressed requests may share prefixes through `kv.prefix`. With the policy off this is
/// exactly [`generate_continuous`]. A step that fails returns the error with every sequence's
/// pages released (a prefix store keeps only the entries it already held).
pub fn generate_continuous_kv(
    model: &CausalLm,
    requests: &[BatchRequest],
    config: &ContinuousConfig,
    kv: ContinuousKv<'_>,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Vec<ContinuousOutput>> {
    if requests.is_empty() {
        return Err(Error::Msg("generate_continuous: no requests".into()));
    }
    if config.max_batch == 0 {
        return Err(Error::Msg(
            "generate_continuous: max_batch must be > 0".into(),
        ));
    }
    if config.block_size == 0 {
        return Err(Error::Msg(
            "generate_continuous: block_size must be > 0".into(),
        ));
    }
    if kv.page_tokens == 0 || !kv.page_tokens.is_multiple_of(PACKED_METAL_QUANT_GROUP_SIZE) {
        return Err(Error::Msg(format!(
            "generate_continuous: page_tokens must be a positive multiple of \
             {PACKED_METAL_QUANT_GROUP_SIZE}"
        )));
    }
    if ![0, requests.len()].contains(&kv.cancels.len())
        || ![0, requests.len()].contains(&kv.policies.len())
    {
        return Err(Error::Msg(
            "generate_continuous: per-request cancels and policies need one entry per request"
                .into(),
        ));
    }
    for (i, r) in requests.iter().enumerate() {
        if r.prompt_ids.is_empty() {
            return Err(Error::Msg(format!(
                "generate_continuous: request {i} has an empty prompt"
            )));
        }
    }
    if cancel.is_cancelled() {
        return Err(Error::Canceled); // typed pre-inference cancel
    }

    let mut run = KvRun::new(model, config, kv, requests.len());

    // Admit every request to the scheduler up front (stable SeqId == request index); only `max_batch`
    // are prefilled into a live lane at a time, the rest wait at `next_req`.
    let mut sched = Scheduler::new();
    let seq_ids: Vec<SeqId> = requests
        .iter()
        .map(|r| {
            sched.admit(SeqSpec::new(
                r.prompt_ids.clone(),
                r.max_new_tokens,
                r.stop_tokens.clone(),
            ))
        })
        .collect();

    let mut lanes: Vec<Lane> = Vec::new();
    let mut next_req = 0usize;

    // Fill the initial slots (prefill is per-sequence: each prompt at its own length, no left-pad).
    while lanes.len() < config.max_batch && next_req < requests.len() {
        if let Some(lane) = admit_lane(
            model, &mut run, requests, &seq_ids, next_req, &mut sched, on_event,
        )? {
            lanes.push(lane);
        }
        next_req += 1;
    }
    // The initial lanes' prefills are done, evaluated (`admit_lane` samples each lane's first token
    // from its prefill logits, which forces the graph), and *dropped* — each lane's logits are a
    // local of `admit_lane` and die on return, so nothing prefill-sized outlives this point. The
    // release is still taken on the loop's first `advance`, keeping one rule across every loop.
    // Later admit-on-retire prefills are covered by the per-step release cadence.
    let mut release = BufferRelease::new();

    // Decode loop: step every live lane, retire finished ones, refill freed slots from the queue.
    // Cancel is checked once at the top of each step (before another forward), so a mid-stream cancel
    // stops promptly without abandoning a forward that has already been computed.
    while !lanes.is_empty() {
        if cancel.is_cancelled() {
            break;
        }
        // A request cancelled on its own leaves the batch before the next step.
        let mut kept = Vec::with_capacity(lanes.len());
        for lane in std::mem::take(&mut lanes) {
            if run.cancelled(lane.req_index) {
                on_event(
                    lane.req_index,
                    StreamEvent::Done {
                        reason: FinishReason::Cancelled,
                        generated: sched.generated(lane.seq).len(),
                    },
                );
                retire_lane(&mut run, lane)?;
            } else {
                kept.push(lane);
            }
        }
        lanes = kept;
        if lanes.is_empty() {
            // Refill below; nothing to step this iteration.
            while lanes.len() < config.max_batch && next_req < requests.len() {
                if let Some(lane) = admit_lane(
                    model, &mut run, requests, &seq_ids, next_req, &mut sched, on_event,
                )? {
                    lanes.push(lane);
                }
                next_req += 1;
            }
            continue;
        }

        let per_lane = step_logits(model, &mut lanes, config.exactness)?;

        let mut survivors: Vec<Lane> = Vec::with_capacity(lanes.len());
        for (mut lane, logits) in std::mem::take(&mut lanes).into_iter().zip(per_lane) {
            let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
            match record_token(&mut sched, &mut lane, tok, on_event) {
                LaneStep::Continue => survivors.push(lane),
                LaneStep::Done => retire_lane(&mut run, lane)?,
            }
        }
        lanes = survivors;
        // After sampling: the step's graphs have been evaluated, so their transients are freeable.
        release.advance(1);

        // Admit-on-retire: refill every freed slot from the waiting requests.
        while lanes.len() < config.max_batch && next_req < requests.len() {
            if let Some(lane) = admit_lane(
                model, &mut run, requests, &seq_ids, next_req, &mut sched, on_event,
            )? {
                lanes.push(lane);
            }
            next_req += 1;
        }
    }

    // A cancel breaks the loop with live lanes still decoding and requests still waiting in the
    // queue; neither has emitted a `Done` yet (a request that retired normally emitted its `Done` as
    // it finished). Emit exactly one `Done` for each so every request signals completion. The final
    // tokens / reason are already recorded in `sched`; these two ranges are disjoint from the
    // already-retired requests and from each other (lane indices are `< next_req`, the queue is
    // `next_req..`), so no request is signalled twice.
    if cancel.is_cancelled() {
        for lane in &lanes {
            on_event(
                lane.req_index,
                StreamEvent::Done {
                    reason: FinishReason::Cancelled,
                    generated: sched.generated(lane.seq).len(),
                },
            );
        }
        for (ri, &seq) in seq_ids.iter().enumerate().skip(next_req) {
            // Never admitted, so never generated: Cancelled — unless it was a zero-budget request the
            // scheduler retired at admission, whose own `MaxTokens` (empty output) we keep.
            let reason = match sched.finish_reason(seq) {
                Some(CoreFinish::Length) => FinishReason::MaxTokens,
                _ => FinishReason::Cancelled,
            };
            on_event(
                ri,
                StreamEvent::Done {
                    reason,
                    generated: 0,
                },
            );
        }
    }
    // Cancelled lanes report what they ran; their pages go back as they drop.
    for lane in lanes {
        run.reports[lane.req_index] = Some(lane.selection.report()?);
    }

    // Assemble per-request outputs in request order from the scheduler's record.
    let prefix_refused = run.prefix_refused();
    seq_ids
        .iter()
        .enumerate()
        .map(|(ri, &seq)| {
            let kv_cache = match run.reports[ri].take() {
                Some(report) => report,
                None => run.planned_report(model, ri, &requests[ri])?,
            };
            Ok(ContinuousOutput {
                output: GenerationOutput {
                    tokens: sched.generated(seq).to_vec(),
                    finish_reason: match sched.finish_reason(seq) {
                        Some(CoreFinish::Stop) => FinishReason::StopToken,
                        Some(CoreFinish::Length) => FinishReason::MaxTokens,
                        // `None` ⇒ still active when a cancel broke the loop (or never admitted).
                        _ => FinishReason::Cancelled,
                    },
                },
                kv_cache,
                reused_prefix_tokens: run.reused[ri],
                prefix_refused: prefix_refused.clone(),
            })
        })
        .collect()
}

/// Prefill request `ri` into a fresh lane and sample its first token. Returns `None` if the request
/// retired immediately (zero budget, or a stop/length on the first token), in which case the
/// scheduler already holds its final state.
#[allow(clippy::too_many_arguments)]
fn admit_lane(
    model: &CausalLm,
    run: &mut KvRun<'_>,
    requests: &[BatchRequest],
    seq_ids: &[SeqId],
    ri: usize,
    sched: &mut Scheduler,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Option<Lane>> {
    let r = &requests[ri];
    let seq = seq_ids[ri];
    if !sched.is_active(seq) {
        // The scheduler retires a zero-budget request (`max_new_tokens == 0`) at admission, so it is
        // already inactive here: emit its terminal event and skip the prefill entirely.
        on_event(
            ri,
            StreamEvent::Done {
                reason: FinishReason::MaxTokens,
                generated: 0,
            },
        );
        return Ok(None);
    }
    if run.cancelled(ri) {
        // Cancelled while it waited: never prefilled, never generated.
        on_event(
            ri,
            StreamEvent::Done {
                reason: FinishReason::Cancelled,
                generated: 0,
            },
        );
        return Ok(None);
    }

    let mut selection = run.select(model, ri, r);
    let reused = run.reuse_prefix(&mut selection, &r.prompt_ids)?;
    run.reused[ri] = reused;
    run.reserve_prefill(&selection, r.prompt_ids.len())?;
    // batch-1 prefill of the positions no stored prefix covers
    let logits = model.decode_logits(
        &input_ids(&r.prompt_ids[reused..]),
        selection.cache_mut(),
        reused as i32,
    )?;
    let mut lane = Lane {
        seq,
        req_index: ri,
        selection,
        rng: SplitMix64::new(r.seed.unwrap_or_else(default_seed)),
        params: r.sampling,
        history: r.prompt_ids.clone(),
        next_token: 0,
    };
    let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
    // Later requests sharing this prompt's prefix start on its pages while it decodes.
    run.store_sequence(&mut lane, &r.prompt_ids)?;
    Ok(match record_token(sched, &mut lane, tok, on_event) {
        LaneStep::Continue => Some(lane),
        LaneStep::Done => {
            retire_lane(run, lane)?;
            None
        }
    })
}

/// A finished lane: offer its sequence to the prefix store, record its report, release its pages.
fn retire_lane(run: &mut KvRun<'_>, mut lane: Lane) -> Result<()> {
    // The cache holds the prompt and every fed token: the history up to its offset.
    let mut tokens = lane.history.clone();
    tokens.truncate(lane.cache().offset().max(0) as usize);
    run.store_sequence(&mut lane, &tokens)?;
    run.reports[lane.req_index] = Some(lane.selection.report()?);
    drop(lane);
    run.trim()
}

/// One decode step's logits, one `[1, vocab]` per live lane (in lane order). `Exact` builds each
/// sequence's own batch-1 forward and evaluates them together (a single device round-trip for the
/// step); `Throughput` runs one batched forward with per-sequence attention and splits the rows.
fn step_logits(
    model: &CausalLm,
    lanes: &mut [Lane],
    exactness: BatchExactness,
) -> Result<Vec<Array>> {
    match exactness {
        BatchExactness::Exact => {
            let mut logits = Vec::with_capacity(lanes.len());
            for lane in lanes.iter_mut() {
                let off = lane.cache().offset();
                logits.push(model.decode_logits(
                    &input_ids(&[lane.next_token]),
                    lane.cache(),
                    off,
                )?);
            }
            eval(logits.iter())?; // force the whole step in one go
            Ok(logits)
        }
        BatchExactness::Throughput => {
            let b = lanes.len();
            let feed: Vec<i32> = lanes.iter().map(|l| l.next_token).collect();
            let ids = Array::from_slice(&feed, &[b as i32, 1]);
            let positions: Vec<i32> = lanes.iter_mut().map(|lane| lane.cache().offset()).collect();
            let mut caches: Vec<&mut dyn KvCache> =
                lanes.iter_mut().map(|lane| lane.cache()).collect();
            let logits = model.decode_logits_per_seq_dyn(&ids, &mut caches, &positions)?; // [b, vocab]
            (0..b as i32)
                .map(|i| {
                    let idx = Array::from_slice(&[i], &[1]);
                    Ok(logits.take_axis(&idx, 0)?) // [1, vocab]
                })
                .collect()
        }
    }
}

/// Record `tok` for `lane` through the scheduler and emit its stream events, identically to the
/// single-sequence loop: a stop token retires the lane (excluded, no `Token` event); otherwise the
/// token is emitted + kept, and a filled budget retires the lane after including it.
fn record_token(
    sched: &mut Scheduler,
    lane: &mut Lane,
    tok: i32,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> LaneStep {
    record_lane_token(
        sched,
        lane.seq,
        lane.req_index,
        tok,
        &mut lane.history,
        &mut lane.next_token,
        on_event,
    )
}

/// sc-20681 on a synthetic two-layer GQA model (hidden 128, 4 query / 2 KV heads, head dim 64).
/// Requests that must qualify carry real prompts at the Qwen3 row's memory-material minimum
/// (10 240 tokens), so the production qualification — not a test override — admits them.
#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::decode::prefix::PagedPrefixStats;
    use crate::primitives::{PagedCacheSnapshot, PagedModelKey};
    use core_llm::KvCompressionFormat;

    const MODEL: &str = "synthetic-tiny@1";

    fn model() -> CausalLm {
        crate::provider::tests::tiny_causal_model(4, 2, 64)
    }

    fn reader() -> CompiledKernelHandle {
        crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).unwrap()
    }

    fn tokens(len: usize, salt: usize) -> Vec<i32> {
        (0..len)
            .map(|i| ((i * 7 + salt * 13 + i / 97) % 31 + 1) as i32)
            .collect()
    }

    fn request(prompt: Vec<i32>, max_new_tokens: usize) -> BatchRequest {
        BatchRequest {
            prompt_ids: prompt,
            sampling: SamplingParams::default(), // greedy
            seed: Some(0),
            max_new_tokens,
            stop_tokens: Vec::new(),
        }
    }

    fn config(max_batch: usize, exactness: BatchExactness) -> ContinuousConfig {
        ContinuousConfig {
            max_batch,
            block_size: 16,
            exactness,
        }
    }

    fn qualified<'a>(prefix: Option<&'a mut PagedPrefixCache>) -> ContinuousKv<'a> {
        ContinuousKv {
            policy: KvCompressionPolicy::Qualified,
            family: Some(KvModelFamily::Qwen3),
            page_tokens: 64,
            reader: Some(reader()),
            prefix,
            model_identity: MODEL,
            ..ContinuousKv::default()
        }
    }

    fn run(
        model: &CausalLm,
        requests: &[BatchRequest],
        config: &ContinuousConfig,
        kv: ContinuousKv<'_>,
    ) -> Vec<ContinuousOutput> {
        generate_continuous_kv(
            model,
            requests,
            config,
            kv,
            &CancelFlag::new(),
            &mut |_, _| {},
        )
        .unwrap()
    }

    fn store(model: &CausalLm, identity: &str) -> PagedPrefixCache {
        let cfg = model.config();
        let pool = PackedPagePool::new(
            cfg.num_layers,
            cfg.num_kv_heads as usize,
            cfg.head_dim as usize,
            64,
            PackedCodeBits::Eight,
        )
        .unwrap();
        PagedPrefixCache::new(
            pool,
            PagedModelKey::new(identity, model.cache_fingerprint()),
            16,
        )
    }

    /// A report without the shared pool's held bytes (which depend on what else the pool holds).
    fn own(report: &KvCacheReport) -> KvCacheReport {
        let mut report = report.clone();
        report.counters.pool_held_bytes = 0;
        report
    }

    fn assert_compressed(out: &ContinuousOutput) {
        let report = &out.kv_cache;
        assert!(report.ran_compressed(), "{report:?}");
        assert_eq!(report.format, Some(KvCompressionFormat::GroupAffineK8V8));
        // A cold prefill attends its fresh K/V densely; a prefill after a reused prefix and every
        // decode step read the pages fused, in both layers.
        let fed = out.output.tokens.len() as u64 - 1 + u64::from(out.reused_prefix_tokens > 0);
        assert_eq!(report.counters.fused_attention_calls, 2 * fed, "{report:?}");
        assert_eq!(report.counters.dense_gather_fallbacks, 0);
        assert!(report.counters.compressed_cache_bytes > 0);
    }

    /// AC1: a continuous batch of jagged qualified requests (and more requests than slots) runs
    /// each qualified request on compressed pages and the short one dense with its reason. In
    /// `Exact` mode every request's tokens equal it run alone; in `Throughput` mode the
    /// compressed requests stay wholly compressed (no padding mask, no dense gather). With the
    /// policy off the run is exactly `generate_continuous`.
    #[test]
    fn continuous_batching_runs_jagged_qualified_requests_compressed_and_the_rest_dense() {
        let model = model();
        let requests = vec![
            request(tokens(10_240, 0), 9),
            request(tokens(10_301, 1), 14),
            request(tokens(300, 2), 6),
            request(tokens(10_433, 3), 11),
        ];
        let exact = run(
            &model,
            &requests,
            &config(3, BatchExactness::Exact),
            qualified(None),
        );
        for (i, out) in exact.iter().enumerate() {
            assert_eq!(out.output.tokens.len(), requests[i].max_new_tokens);
            let alone = run(
                &model,
                std::slice::from_ref(&requests[i]),
                &config(1, BatchExactness::Exact),
                qualified(None),
            );
            assert_eq!(out.output.tokens, alone[0].output.tokens, "request {i}");
            assert_eq!(own(&out.kv_cache), own(&alone[0].kv_cache), "request {i}");
            if i == 2 {
                assert_eq!(
                    out.kv_cache,
                    KvCacheReport::dense(KvCacheFallbackReason::BelowMinimumContext, None)
                );
            } else {
                assert_compressed(out);
            }
        }

        let throughput = run(
            &model,
            &requests,
            &config(3, BatchExactness::Throughput),
            qualified(None),
        );
        for (i, out) in throughput.iter().enumerate() {
            assert_eq!(out.output.tokens.len(), requests[i].max_new_tokens);
            if i == 2 {
                assert_eq!(
                    out.kv_cache.fallback,
                    Some(KvCacheFallbackReason::BelowMinimumContext)
                );
            } else {
                assert_compressed(out);
            }
        }

        let off = run(
            &model,
            &requests,
            &config(3, BatchExactness::Throughput),
            ContinuousKv::default(),
        );
        let established = generate_continuous(
            &model,
            &requests,
            &config(3, BatchExactness::Throughput),
            &CancelFlag::new(),
            &mut |_, _| {},
        )
        .unwrap();
        for (out, established) in off.iter().zip(&established) {
            assert_eq!(out.output.tokens, established.tokens);
            assert_eq!(
                out.kv_cache,
                KvCacheReport::dense(KvCacheFallbackReason::PolicyDisabled, None)
            );
        }
    }

    enum Layout {
        /// Compressed lanes on one pool: one batched dispatch per layer.
        Shared,
        /// Compressed lanes on their own pools: each attends on its own.
        Separate,
        /// Every lane dense paged.
        Dense,
    }

    /// AC1 at the decode-forward boundary: in a `Throughput` step mixing three paged compressed
    /// sequences of jagged lengths with a dense one, the compressed sequences attend through one
    /// batched paged dispatch per layer and produce exactly the logits they produce attending one
    /// by one; the dense sequence's logits are exactly those of an all-dense batch; the
    /// compressed rows stay within 8-bit rounding of the dense run.
    #[test]
    fn throughput_batches_compressed_sequences_in_one_dispatch_beside_a_dense_one() {
        let model = model();
        let reader = reader();
        let lengths = [40, 75, 130, 57];
        let run = |layout: Layout| {
            let cfg = model.config();
            let new_pool =
                || PackedPagePool::new(cfg.num_layers, 2, 64, 32, PackedCodeBits::Eight).unwrap();
            let shared = new_pool();
            let mut caches: Vec<Box<dyn KvCache>> = (0..4)
                .map(|b| -> Box<dyn KvCache> {
                    match (&layout, b) {
                        (Layout::Dense, _) | (_, 1) => Box::new(PagedKvCache::new(2, 16)),
                        (Layout::Shared, _) => Box::new(
                            PagedPackedKvCache::with_pool(shared.clone(), reader.clone()).unwrap(),
                        ),
                        (Layout::Separate, _) => Box::new(
                            PagedPackedKvCache::with_pool(new_pool(), reader.clone()).unwrap(),
                        ),
                    }
                })
                .collect();
            for (b, cache) in caches.iter_mut().enumerate() {
                let logits = model
                    .decode_logits(&input_ids(&tokens(lengths[b], b)), cache.as_mut(), 0)
                    .unwrap();
                logits.eval().unwrap();
            }
            let mut rows = Vec::new();
            for step in 0..24 {
                let feed = (0..4)
                    .map(|b| (step * 5 + b) % 31 + 1)
                    .collect::<Vec<i32>>();
                let ids = Array::from_slice(&feed, &[4, 1]);
                let positions = caches
                    .iter()
                    .map(|cache| cache.offset())
                    .collect::<Vec<_>>();
                let mut refs: Vec<&mut dyn KvCache> = caches
                    .iter_mut()
                    .map(|cache| cache.as_mut() as &mut dyn KvCache)
                    .collect();
                let logits = model
                    .decode_logits_per_seq_dyn(&ids, &mut refs, &positions)
                    .unwrap()
                    .as_dtype(mlx_rs::Dtype::Float32)
                    .unwrap();
                logits.eval().unwrap();
                let flat = logits.as_slice::<f32>().to_vec();
                rows.push(flat.chunks(32).map(<[f32]>::to_vec).collect::<Vec<_>>());
            }
            let batched = caches
                .iter_mut()
                .map(|cache| {
                    cache
                        .as_any_mut()
                        .downcast_mut::<PagedPackedKvCache>()
                        .map_or(0, |cache| cache.batched_calls())
                })
                .collect::<Vec<_>>();
            (rows, batched)
        };
        let (shared, shared_batched) = run(Layout::Shared);
        let (separate, separate_batched) = run(Layout::Separate);
        let (dense, _) = run(Layout::Dense);
        assert_eq!(
            shared, separate,
            "batched vs one-by-one compressed attention"
        );
        // The first step's layer 0 warms the paged reader one sequence at a time.
        assert_eq!(shared_batched, vec![2 * 24 - 1, 0, 2 * 24 - 1, 2 * 24 - 1]);
        assert_eq!(separate_batched, vec![0; 4]);
        let range = dense
            .iter()
            .flatten()
            .flatten()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        let mut worst = 0.0f32;
        for (shared_step, dense_step) in shared.iter().zip(&dense) {
            assert_eq!(shared_step[1], dense_step[1], "the dense lane is untouched");
            for b in [0, 2, 3] {
                for (a, d) in shared_step[b].iter().zip(&dense_step[b]) {
                    worst = worst.max((a - d).abs() / range);
                }
            }
        }
        assert!(worst < 0.02, "compressed vs dense paged: {worst}");
    }

    /// AC2: requests sharing a 10 272-token prefix (not page-aligned, so the shared last page is
    /// partially filled and every writer copies it) decode in one batch, the later ones started on
    /// the first one's pages while it still decodes. Each request's tokens and report equal the
    /// same requests run one after another on a fresh store — so no request saw another's writes
    /// — and when the run ends the pool holds exactly the store's pages; clearing the store frees
    /// every page.
    #[test]
    fn in_batch_prefix_sharing_equals_sequential_sharing_and_returns_every_page() {
        let model = model();
        let prefix = tokens(10_272, 9);
        // Suffixes start with distinct tokens, so the longest shared prefix is exactly `prefix`.
        let with_suffix = |salt: usize, len: usize| {
            let mut prompt = prefix.clone();
            prompt.push(salt as i32 + 1);
            prompt.extend(tokens(len - 1, 20 + salt));
            prompt
        };
        let requests = vec![
            request(with_suffix(0, 37), 8),
            request(with_suffix(1, 101), 12),
            request(with_suffix(2, 5), 6),
        ];
        let mut concurrent = store(&model, MODEL);
        let batched = run(
            &model,
            &requests,
            &config(3, BatchExactness::Exact),
            qualified(Some(&mut concurrent)),
        );
        assert_eq!(
            batched
                .iter()
                .map(|out| out.reused_prefix_tokens)
                .collect::<Vec<_>>(),
            vec![0, 10_272, 10_272]
        );
        let stats = concurrent.stats();
        assert_eq!((stats.hits, stats.reused_tokens), (2, 2 * 10_272));
        assert_eq!(stats.refused, 0);

        let mut sequential = store(&model, MODEL);
        for (i, r) in requests.iter().enumerate() {
            let alone = run(
                &model,
                std::slice::from_ref(r),
                &config(1, BatchExactness::Exact),
                qualified(Some(&mut sequential)),
            );
            assert_eq!(
                batched[i].output.tokens, alone[0].output.tokens,
                "request {i}"
            );
            assert_eq!(
                batched[i].reused_prefix_tokens,
                alone[0].reused_prefix_tokens
            );
            assert_compressed(&batched[i]);
        }

        for store in [&mut concurrent, &mut sequential] {
            let pool = store.pool().clone();
            assert!(pool.borrow().live_pages() > 0);
            assert_eq!(
                pool.borrow().live_pages(),
                store.held_pages(),
                "only the store's pages"
            );
            store.clear().unwrap();
            assert_eq!(pool.borrow().live_pages(), 0);
        }
    }

    /// AC2 refcounts: a run cancelled mid-decode, and a run whose fused reader faults mid-decode
    /// (an error, not a fallback), both leave the pool holding exactly the prefix store's pages.
    #[test]
    fn cancellation_and_failure_return_the_pool_to_the_store_baseline() {
        let model = model();
        let prefix = tokens(10_240, 4);
        let requests = (0..4)
            .map(|i| {
                let mut prompt = prefix.clone();
                prompt.extend(tokens(20 + 7 * i, 30 + i));
                request(prompt, 10)
            })
            .collect::<Vec<_>>();

        let mut cancelled_store = store(&model, MODEL);
        let cancel = CancelFlag::new();
        let mut emitted = 0;
        let outputs = generate_continuous_kv(
            &model,
            &requests,
            &config(2, BatchExactness::Throughput),
            qualified(Some(&mut cancelled_store)),
            &cancel,
            &mut |_, event| {
                emitted += usize::from(matches!(event, StreamEvent::Token { .. }));
                if emitted == 5 {
                    cancel.cancel();
                }
            },
        )
        .unwrap();
        assert!(outputs
            .iter()
            .any(|out| out.output.finish_reason == FinishReason::Cancelled));
        let pool = cancelled_store.pool().clone();
        assert_eq!(pool.borrow().live_pages(), cancelled_store.held_pages());
        cancelled_store.clear().unwrap();
        assert_eq!(pool.borrow().live_pages(), 0);

        let mut failed_store = store(&model, MODEL);
        let failing = ContinuousKv {
            reader: Some(crate::primitives::paged_packed_kv::tests::interrupting(
                9, false,
            )),
            ..qualified(Some(&mut failed_store))
        };
        let error = generate_continuous_kv(
            &model,
            &requests,
            &config(2, BatchExactness::Exact),
            failing,
            &CancelFlag::new(),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(error.to_string().contains("injected"), "{error}");
        let pool = failed_store.pool().clone();
        assert!(
            !failed_store.is_empty(),
            "the admitted prompts were stored before the fault"
        );
        assert_eq!(pool.borrow().live_pages(), failed_store.held_pages());
        failed_store.clear().unwrap();
        assert_eq!(pool.borrow().live_pages(), 0);
    }

    /// AC2 identity: a store keyed by another model identity, or built for another page geometry,
    /// is refused for every compressed request — nothing is reused from it or stored into it, and
    /// its pool is untouched — while the requests still run (compressed, on the run's own pool)
    /// exactly as they do with no store.
    #[test]
    fn a_store_of_another_identity_is_refused_and_never_reused() {
        let model = model();
        let prefix = tokens(10_240, 6);
        let requests = (0..2)
            .map(|i| {
                let mut prompt = prefix.clone();
                prompt.extend(tokens(11 + i, 40 + i));
                request(prompt, 5)
            })
            .collect::<Vec<_>>();
        let plain = run(
            &model,
            &requests,
            &config(2, BatchExactness::Exact),
            qualified(None),
        );
        let mut primed = store(&model, "another-model@2");
        run(
            &model,
            &requests[..1],
            &config(1, BatchExactness::Exact),
            ContinuousKv {
                model_identity: "another-model@2",
                ..qualified(Some(&mut primed))
            },
        );
        let held = (primed.len(), primed.held_pages(), primed.stats());
        let other_geometry = PagedPrefixCache::new(
            PackedPagePool::new(2, 1, 64, 64, PackedCodeBits::Eight).unwrap(),
            PagedModelKey::new(MODEL, model.cache_fingerprint()),
            4,
        );
        for (mut store, field) in [(primed, "model"), (other_geometry, "KV heads")] {
            let before = store.pool().borrow().live_pages();
            let refused_before = store.stats().refused;
            let outputs = run(
                &model,
                &requests,
                &config(2, BatchExactness::Exact),
                qualified(Some(&mut store)),
            );
            for (out, plain) in outputs.iter().zip(&plain) {
                assert_eq!(out.reused_prefix_tokens, 0);
                let refused = out.prefix_refused.as_deref().unwrap_or_default();
                assert!(refused.contains(field), "{field}: {refused}");
                assert_eq!(out.output.tokens, plain.output.tokens);
                assert_compressed(out);
            }
            assert_eq!(store.stats().refused, refused_before + 2);
            assert_eq!(store.pool().borrow().live_pages(), before);
            if field == "model" {
                let after = store.stats();
                assert_eq!((store.len(), store.held_pages()), (held.0, held.1));
                assert_eq!(
                    (after.hits, after.stored),
                    (held.2.hits, held.2.stored),
                    "nothing reused or stored"
                );
            }
        }
    }

    /// AC2 restore: a store saved as bytes and restored into a new pool (a process restart) serves
    /// a new request exactly as the original store does; a snapshot whose KV format version was
    /// changed is refused and counted, and nothing is stored.
    #[test]
    fn a_saved_prefix_store_restores_and_a_version_mismatch_is_refused() {
        let model = model();
        let prefix = tokens(10_240, 8);
        let with_suffix = |salt: usize| {
            let mut prompt = prefix.clone();
            prompt.extend(tokens(17 + salt, 50 + salt));
            request(prompt, 6)
        };
        let mut original = store(&model, MODEL);
        run(
            &model,
            &[with_suffix(0)],
            &config(1, BatchExactness::Exact),
            qualified(Some(&mut original)),
        );
        let saved = original
            .snapshots()
            .unwrap()
            .into_iter()
            .map(|(tokens, snapshot)| (tokens, snapshot.to_bytes().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(saved.len(), 2, "the prompt and the finished sequence");

        let mut restored = store(&model, MODEL);
        for (tokens, bytes) in &saved {
            let snapshot = PagedCacheSnapshot::from_bytes(bytes).unwrap();
            restored
                .restore(tokens.clone(), &snapshot, reader())
                .unwrap();
        }
        assert_eq!(restored.len(), 2);
        let next = [with_suffix(1)];
        let from_original = run(
            &model,
            &next,
            &config(1, BatchExactness::Exact),
            qualified(Some(&mut original)),
        );
        let from_restored = run(
            &model,
            &next,
            &config(1, BatchExactness::Exact),
            qualified(Some(&mut restored)),
        );
        assert_eq!(from_restored[0].reused_prefix_tokens, 10_240);
        assert_eq!(
            from_original[0].reused_prefix_tokens,
            from_restored[0].reused_prefix_tokens
        );
        assert_eq!(
            from_original[0].output.tokens,
            from_restored[0].output.tokens
        );

        // An edited header (here its format version) fails the digest when read back.
        let mut tampered = saved[0].1.clone();
        let needle = br#""format_version":1"#;
        let at = tampered
            .windows(needle.len())
            .position(|window| window == needle)
            .expect("the header names the format version");
        tampered[at + needle.len() - 1] = b'2';
        let error = PagedCacheSnapshot::from_bytes(&tampered)
            .unwrap_err()
            .to_string();
        assert!(error.contains("digest"), "{error}");

        // A snapshot of another format version, and one bound to other (same-length) tokens, are
        // refused and counted; nothing is stored or allocated.
        let mut fresh = store(&model, MODEL);
        let original_snapshot = PagedCacheSnapshot::from_bytes(&saved[0].1).unwrap();
        let mut other_version = original_snapshot.clone();
        other_version.identity_mut().format_version += 1;
        let error = fresh
            .restore(saved[0].0.clone(), &other_version, reader())
            .unwrap_err()
            .to_string();
        assert!(error.contains("KV format version"), "{error}");
        let mut other_tokens = saved[0].0.clone();
        other_tokens[0] = other_tokens[0] % 31 + 1;
        assert_eq!(other_tokens.len(), saved[0].0.len());
        let error = fresh
            .restore(other_tokens, &original_snapshot, reader())
            .unwrap_err()
            .to_string();
        assert!(error.contains("token ids"), "{error}");
        assert_eq!(
            fresh.stats(),
            PagedPrefixStats {
                refused: 2,
                ..PagedPrefixStats::default()
            }
        );
        assert!(fresh.is_empty());
        assert_eq!(fresh.pool().borrow().live_pages(), 0);

        // A sequence on another pool is never stored: its pages are not the store's to share.
        let foreign_pool = PackedPagePool::new(2, 2, 64, 64, PackedCodeBits::Eight).unwrap();
        let mut foreign = PagedPackedKvCache::with_pool(foreign_pool, reader()).unwrap();
        let prompt = tokens(40, 1);
        model
            .decode_logits(&input_ids(&prompt), &mut foreign, 0)
            .unwrap()
            .eval()
            .unwrap();
        let identity = fresh.identity().clone();
        assert!(!fresh.insert(&identity, &prompt, &foreign).unwrap());
        assert_eq!(fresh.stats().refused, 3);
        assert!(fresh.is_empty());
    }

    /// AC1 pool sizing: four jagged qualified requests (page_tokens 32) admitted together hold no
    /// more page capacity than their live pages plus one growth chunk while they prefill and
    /// decode — each prefill reserves exactly its pages (at most two spare while they are
    /// admitted), growth is bounded — and every report names the pool's held bytes. The last
    /// request admitted (holding the top pages) retires first and the pool is trimmed right away;
    /// once every request has retired it is trimmed back to (at most) one growth chunk.
    ///
    /// The resident-memory claim, at every decode event: the bytes the pool holds (its whole
    /// capacity, not only the live pages) stay at or under 0.60 of the dense bf16 K/V of the
    /// positions the live sequences hold — before and after the first (top-page) retirement.
    /// Doubling growth, unreserved prefills or a pool that is never trimmed hold more (0.85 of
    /// dense measured before sc-20681's fix).
    #[test]
    fn continuous_runs_reserve_their_prefill_and_trim_the_pool_on_retirement() {
        let model = model();
        let pool = PackedPagePool::new(2, 2, 64, 32, PackedCodeBits::Eight).unwrap();
        let (page_codes, page_metadata) = pool.borrow().page_bytes();
        let requests = [10_240, 10_331, 10_478, 10_603]
            .into_iter()
            .enumerate()
            .map(|(i, len)| request(tokens(len, i), if i == 3 { 2 } else { 70 }))
            .collect::<Vec<_>>();
        let mut checked = 0;
        let (mut first_tokens, mut retired) = (0, 0);
        let watched = pool.clone();
        let prompt_lens = requests
            .iter()
            .map(|request| request.prompt_ids.len() as u64)
            .collect::<Vec<_>>();
        // Positions each sequence's cache holds at its latest event (0 once retired).
        let mut cached = vec![0_u64; requests.len()];
        let (mut ratio_checks, mut after_retirement, mut worst) = (0, 0, 0.0_f64);
        // Dense bf16 K/V per position: 2 layers × K and V × 2 KV heads × 64 dims × 2 bytes.
        let dense_per_position = (2 * 2 * 2 * 64 * 2) as u64;
        let outputs = generate_continuous_kv(
            &model,
            &requests,
            &config(4, BatchExactness::Throughput),
            ContinuousKv {
                page_tokens: 32,
                pool: Some(pool.clone()),
                ..qualified(None)
            },
            &CancelFlag::new(),
            &mut |i, event| {
                match event {
                    StreamEvent::Done { .. } => {
                        retired += 1;
                        cached[i] = 0;
                    }
                    StreamEvent::Token { step, .. } => {
                        if step == 0 {
                            first_tokens += 1;
                        }
                        cached[i] = prompt_lens[i] + step as u64;
                        if first_tokens == 4 && retired <= 1 {
                            // Every request is prefilled (and at most the top-page one retired):
                            // compare the pool's held bytes with the dense K/V the live sequences
                            // hold. A retirement below the top leaves its pages as free capacity
                            // the pool reuses (lowest id first) rather than compacts; the final
                            // step's retirements are past this check.
                            let storage = watched.borrow().storage();
                            let held = (storage.code_bytes + storage.metadata_bytes) as f64;
                            let dense = (cached.iter().sum::<u64>() * dense_per_position) as f64;
                            let ratio = held / dense;
                            worst = worst.max(ratio);
                            assert!(
                                ratio <= 0.60,
                                "pool holds {ratio:.3} of dense bf16 ({retired} retired, \
                                 {storage:?})"
                            );
                            ratio_checks += 1;
                            after_retirement += usize::from(retired == 1);
                        }
                    }
                }
                if retired > 1 {
                    return;
                }
                let pool = watched.borrow();
                let storage = pool.storage();
                let live = pool.live_pages() as u64;
                // Retirement events precede the retirement's trim: check the next event.
                if retired == 0 || !matches!(event, StreamEvent::Done { .. }) {
                    // After the first retirement's trim: rounded to the (pre-trim) growth chunk.
                    let chunk = (4_u64).max(live / 32) * if retired == 0 { 1 } else { 2 };
                    let bound = (live + chunk) * (page_codes + page_metadata);
                    assert!(
                        storage.code_bytes + storage.metadata_bytes <= bound,
                        "{} pages held for {live} live",
                        storage.capacity_pages
                    );
                    checked += 1;
                }
                if retired == 0 && first_tokens < 4 {
                    // While admitting: exactly the reserved prefill pages, plus one.
                    assert!(storage.capacity_pages <= live + 2, "{storage:?}");
                }
            },
        )
        .unwrap();
        assert!(
            checked >= 4 + 3,
            "every prefill and decode step was checked"
        );
        eprintln!("pool held / dense bf16: worst {worst:.3} over {ratio_checks} events");
        // Decode steps after the first retirement (request 3 leaves after 2 tokens) were checked.
        assert!(
            ratio_checks >= 4 && after_retirement >= 3,
            "{ratio_checks} ratio checks, {after_retirement} after the first retirement"
        );
        for out in &outputs {
            assert_compressed(out);
            assert!(
                out.kv_cache.counters.pool_held_bytes
                    >= out.kv_cache.counters.compressed_cache_bytes
            );
        }
        let pool = pool.borrow();
        assert_eq!(pool.live_pages(), 0);
        assert!(
            pool.capacity_pages() <= pool.growth_chunk(),
            "{} pages kept after every request retired",
            pool.capacity_pages()
        );
    }

    /// Per-request cancellation in a batch: a request cancelled while it decodes leaves with its
    /// partial output (`Cancelled`), one cancelled while it waits is never prefilled, and the
    /// others decode to their budgets, each with exactly one terminal event.
    #[test]
    fn a_request_cancelled_on_its_own_leaves_the_batch_and_the_rest_finish() {
        let model = model();
        let requests = (0..3)
            .map(|i| request(tokens(40 + 9 * i, i), 8))
            .collect::<Vec<_>>();
        let cancels = [CancelFlag::new(), CancelFlag::new(), CancelFlag::new()];
        cancels[2].cancel();
        let mut dones = vec![Vec::new(); 3];
        let mut tokens_seen = [0usize; 3];
        let outputs = generate_continuous_kv(
            &model,
            &requests,
            &config(2, BatchExactness::Throughput),
            ContinuousKv {
                cancels: &cancels,
                ..ContinuousKv::default()
            },
            &CancelFlag::new(),
            &mut |i, event| match event {
                StreamEvent::Token { .. } => {
                    tokens_seen[i] += 1;
                    if i == 1 && tokens_seen[i] == 3 {
                        cancels[1].cancel();
                    }
                }
                StreamEvent::Done { reason, generated } => dones[i].push((reason, generated)),
            },
        )
        .unwrap();
        assert_eq!(outputs[0].output.finish_reason, FinishReason::MaxTokens);
        assert_eq!(outputs[0].output.tokens.len(), 8);
        assert_eq!(outputs[1].output.finish_reason, FinishReason::Cancelled);
        assert_eq!(outputs[1].output.tokens.len(), 3);
        assert_eq!(outputs[2].output.finish_reason, FinishReason::Cancelled);
        assert!(outputs[2].output.tokens.is_empty());
        assert_eq!(dones[0], vec![(FinishReason::MaxTokens, 8)]);
        assert_eq!(dones[1], vec![(FinishReason::Cancelled, 3)]);
        assert_eq!(dones[2], vec![(FinishReason::Cancelled, 0)]);
    }

    /// AC2 identity: a store keyed by one decoder is refused by another decoder of the same
    /// geometry and name but another configuration (here its RoPE base), naming the fingerprint;
    /// and a run with no model identity never consults a store.
    #[test]
    fn a_store_of_another_decoder_or_an_unnamed_run_is_refused() {
        let model = model();
        let other = crate::provider::tests::synthetic_causal_model_with_rope(128, 2, 4, 2, 64, 5e5);
        assert_ne!(model.cache_fingerprint(), other.cache_fingerprint());
        let prompt = tokens(10_240, 12);
        let with_suffix = |salt: usize| {
            let mut ids = prompt.clone();
            ids.extend(tokens(9 + salt, 60 + salt));
            request(ids, 4)
        };
        let mut store = store(&model, MODEL);
        run(
            &model,
            &[with_suffix(0)],
            &config(1, BatchExactness::Exact),
            qualified(Some(&mut store)),
        );
        let held = store.held_pages();
        for (decoder, identity, refusal) in [
            (&other, MODEL, "model fingerprint"),
            (&model, "", "no model identity"),
        ] {
            let refused_before = store.stats().refused;
            let outputs = run(
                decoder,
                &[with_suffix(1)],
                &config(1, BatchExactness::Exact),
                ContinuousKv {
                    model_identity: identity,
                    ..qualified(Some(&mut store))
                },
            );
            let reason = outputs[0].prefix_refused.as_deref().unwrap_or_default();
            assert!(reason.contains(refusal), "{refusal}: {reason}");
            assert_eq!(outputs[0].reused_prefix_tokens, 0);
            assert_compressed(&outputs[0]);
            assert_eq!(store.stats().refused, refused_before + 1);
            assert_eq!(store.held_pages(), held, "nothing stored");
        }
        // A store keyed by an empty identity serves nobody, not even its own identity.
        let mut unnamed = super::tests::store(&model, "");
        let identity = unnamed.identity().clone();
        match unnamed.lookup(&identity, &prompt).unwrap() {
            PagedPrefixLookup::Refused(reason) => {
                assert!(reason.contains("no model identity"), "{reason}")
            }
            other => panic!("an unnamed store served a lookup: {other:?}"),
        }
        assert_eq!(unnamed.stats().refused, 1);
    }

    /// AC3 measurement (not a gate): `Throughput` continuous batching of `B` qualified sequences
    /// with jagged ~10–11k-token contexts on a synthetic 4-layer GQA model (8 query / 2 KV heads,
    /// head dim 128), paged compressed (K8V8, one batched paged dispatch per layer) against the
    /// dense paged cache. Reports the KV memory the caches hold after prefill (MLX active memory
    /// over the weights-only baseline), the decode step's peak, and steady decode throughput over
    /// three rounds of 64 teacher-forced batched steps (median).
    ///
    /// ```text
    /// cargo test -p mlx-llm --lib -- --ignored --nocapture continuous_batching_memory_and_decode
    /// ```
    #[test]
    #[ignore = "GPU measurement; run explicitly with --ignored --nocapture"]
    fn continuous_batching_memory_and_decode_throughput_vs_dense_paged() {
        use mlx_rs::memory;
        let model = crate::provider::tests::synthetic_causal_model(256, 4, 8, 2, 128);
        let reader = reader();
        for batch in [4usize, 8] {
            let lengths = (0..batch)
                .map(|b| 10_240 + (b * 97) % 700)
                .collect::<Vec<_>>();
            let tokens_total = lengths.iter().sum::<usize>();
            let mut results = Vec::new();
            for policy in [KvCompressionPolicy::Qualified, KvCompressionPolicy::Off] {
                memory::clear_cache();
                let baseline = memory::get_active_memory() as u64;
                let cfg = model.config();
                let dense_pool = BlockPool::new(16);
                let packed_pool = PackedPagePool::new(
                    cfg.num_layers,
                    cfg.num_kv_heads as usize,
                    cfg.head_dim as usize,
                    64,
                    PackedCodeBits::Eight,
                )
                .unwrap();
                let mut lanes = lengths
                    .iter()
                    .map(|&len| {
                        model.select_paged_cache(PagedCacheRequest {
                            policy,
                            family: Some(KvModelFamily::Qwen3),
                            prompt_tokens: len as u64,
                            max_new_tokens: 1_024,
                            dense_pool: &dense_pool,
                            packed_pool: &packed_pool,
                            reader: Some(&reader),
                        })
                    })
                    .collect::<Vec<_>>();
                for (b, lane) in lanes.iter_mut().enumerate() {
                    assert_eq!(
                        lane.is_compressed(),
                        policy == KvCompressionPolicy::Qualified
                    );
                    if lane.is_compressed() {
                        // What `generate_continuous_kv` reserves before each prefill.
                        let mut pool = packed_pool.borrow_mut();
                        let pages = pool.live_pages() + lengths[b].div_ceil(64) + 1;
                        pool.reserve(pages).unwrap();
                    }
                    let logits = model
                        .decode_logits(&input_ids(&tokens(lengths[b], b)), lane.cache_mut(), 0)
                        .unwrap();
                    logits.eval().unwrap();
                }
                memory::clear_cache();
                let held = memory::get_active_memory() as u64 - baseline;
                // The compressed pages' own bytes (live pages, not the pool's doubled capacity)
                // plus each sequence's table and residuals; the dense caches' K/V.
                let attributed = lanes
                    .iter()
                    .map(|lane| {
                        lane.cache()
                            .compressed_storage()
                            .unwrap()
                            .map_or(0, |storage| storage.device_bytes())
                    })
                    .sum::<u64>();
                let capacity = packed_pool.borrow().storage();
                let step = |lanes: &mut [PagedCacheSelection], i: usize| {
                    let feed = (0..lanes.len())
                        .map(|b| ((i * 5 + b) % 31 + 1) as i32)
                        .collect::<Vec<_>>();
                    let ids = Array::from_slice(&feed, &[lanes.len() as i32, 1]);
                    let positions = lanes
                        .iter()
                        .map(|lane| lane.cache().offset())
                        .collect::<Vec<_>>();
                    let mut refs: Vec<&mut dyn KvCache> =
                        lanes.iter_mut().map(|lane| lane.cache_mut()).collect();
                    let logits = model
                        .decode_logits_per_seq_dyn(&ids, &mut refs, &positions)
                        .unwrap();
                    logits.eval().unwrap();
                };
                for i in 0..4 {
                    step(&mut lanes, i);
                }
                memory::reset_peak_memory();
                // Median of three 64-step rounds (the host GPU is shared).
                let steps = 64;
                let mut rates = (0..3)
                    .map(|round| {
                        let started = std::time::Instant::now();
                        for i in 0..steps {
                            step(&mut lanes, 4 + round * steps + i);
                        }
                        (batch * steps) as f64 / started.elapsed().as_secs_f64()
                    })
                    .collect::<Vec<_>>();
                rates.sort_by(f64::total_cmp);
                let peak = memory::get_peak_memory() as u64 - baseline;
                let reports = lanes
                    .iter()
                    .map(|lane| lane.report().unwrap())
                    .collect::<Vec<_>>();
                if policy == KvCompressionPolicy::Qualified {
                    assert!(reports.iter().all(KvCacheReport::ran_compressed));
                }
                if policy == KvCompressionPolicy::Qualified {
                    eprintln!(
                        "  (B={batch} compressed: live pages + tables + residuals {:.1} MiB; pool \
                         capacity {} pages for {} live = {:.1} MiB)",
                        attributed as f64 / (1 << 20) as f64,
                        capacity.capacity_pages,
                        capacity.live_pages,
                        (capacity.code_bytes + capacity.metadata_bytes) as f64 / (1 << 20) as f64
                    );
                }
                results.push((policy, held, peak, rates[1]));
                drop(lanes);
            }
            let dense_kv = tokens_total as u64
                * model.config().num_layers as u64
                * 2
                * model.config().num_kv_heads as u64
                * model.config().head_dim as u64
                * 2;
            eprintln!(
                "B={batch}, {tokens_total} context tokens (dense bf16 K/V {:.1} MiB):",
                dense_kv as f64 / (1 << 20) as f64
            );
            for (policy, held, peak, rate) in &results {
                eprintln!(
                    "  {policy:?}: KV held after prefill {:.1} MiB, decode peak {:.1} MiB, \
                     decode {rate:.1} tok/s",
                    *held as f64 / (1 << 20) as f64,
                    *peak as f64 / (1 << 20) as f64
                );
            }
            let (compressed, dense) = (&results[0], &results[1]);
            let held = compressed.1 as f64 / dense.1 as f64;
            eprintln!(
                "  compressed / dense: memory {held:.3}, decode-step peak {:.3}, throughput {:.3}",
                compressed.2 as f64 / dense.2 as f64,
                compressed.3 / dense.3
            );
            // The resident (held) KV, pool capacity included, not only the live pages.
            assert!(
                held <= 0.60,
                "B={batch}: held compressed / held dense {held:.3}"
            );
        }
    }
}

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
    PagedCacheRequest, PagedCacheSelection, PagedKvCache, PagedPackedKvCache,
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
    /// requests when its identity matches this run's.
    pub prefix: Option<&'a mut PagedPrefixCache>,
    /// The model identity (checkpoint and revision) the run's compressed caches are keyed by.
    pub model_identity: &'a str,
}

impl Default for ContinuousKv<'_> {
    fn default() -> Self {
        Self {
            policy: KvCompressionPolicy::Off,
            family: None,
            page_tokens: 64,
            reader: None,
            prefix: None,
            model_identity: "",
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
    /// A store refused for its identity: the mismatch, and the store to record each refusal on.
    refused_store: Option<(&'a mut PagedPrefixCache, PagedCacheIdentity, String)>,
    /// Per request: the report it finished with and the positions it reused.
    reports: Vec<Option<KvCacheReport>>,
    reused: Vec<usize>,
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
        };
        if kv.policy == KvCompressionPolicy::Off {
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
            let expected = new_pool(store.identity().page_tokens)
                .map(|pool| pool.borrow().identity(kv.model_identity));
            match expected {
                Ok(identity) => match store.identity().mismatch(&identity) {
                    None => {
                        run.packed_pool = Some(store.pool().clone());
                        run.store = Some((store, identity));
                    }
                    Some(mismatch) => run.refused_store = Some((store, identity, mismatch)),
                },
                Err(error) => {
                    let identity = store.identity().clone();
                    run.refused_store = Some((store, identity, error.to_string()));
                }
            }
        }
        if run.packed_pool.is_none() {
            match new_pool(kv.page_tokens) {
                Ok(pool) => run.packed_pool = Some(pool),
                Err(error) => run.pool_refusal = Some(error.to_string()),
            }
        }
        run.reader = kv.reader.or_else(|| {
            crate::kv_policy::group_affine_reader(PackedCodeBits::Eight).ok()
        });
        run
    }

    /// The cache request `r` runs on, before any K/V mutation.
    fn select(&self, model: &CausalLm, r: &BatchRequest) -> PagedCacheSelection {
        let prompt_tokens = u64::try_from(r.prompt_ids.len()).unwrap_or(u64::MAX);
        let max_new_tokens = u64::try_from(r.max_new_tokens).unwrap_or(u64::MAX);
        let Some(packed_pool) = self.packed_pool.as_ref() else {
            let reason = match core_llm::qualify_kv_sequence(
                self.policy,
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
            policy: self.policy,
            family: self.family,
            prompt_tokens,
            max_new_tokens,
            dense_pool: &self.dense_pool,
            packed_pool,
            reader: self.reader.as_ref(),
        })
    }

    /// The report of a request that never ran (cancelled in the queue, or a zero budget).
    fn planned_report(&self, model: &CausalLm, r: &BatchRequest) -> Result<KvCacheReport> {
        self.select(model, r).report()
    }

    /// Start a compressed `selection` on the longest stored prefix of `prompt`; the positions it
    /// already holds.
    fn reuse_prefix(&mut self, selection: &mut PagedCacheSelection, prompt: &[i32]) -> Result<usize> {
        if !selection.is_compressed() {
            return Ok(0);
        }
        if let Some((store, identity, _)) = self.refused_store.as_mut() {
            // Counted on the store; nothing is reused.
            store.lookup(identity, prompt)?;
            return Ok(0);
        }
        let Some((store, identity)) = self.store.as_mut() else {
            return Ok(0);
        };
        match store.lookup(identity, prompt)? {
            PagedPrefixLookup::Hit { cache, tokens } => {
                selection.replace_compressed(cache)?;
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

    fn prefix_refused(&self) -> Option<String> {
        self.refused_store
            .as_ref()
            .map(|(_, _, mismatch)| mismatch.clone())
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
                None => run.planned_report(model, &requests[ri])?,
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

    let mut selection = run.select(model, r);
    let reused = run.reuse_prefix(&mut selection, &r.prompt_ids)?;
    run.reused[ri] = reused;
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
    Ok(())
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
            let positions: Vec<i32> = lanes
                .iter_mut()
                .map(|lane| lane.cache().offset())
                .collect();
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

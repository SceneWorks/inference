//! Iteration-level continuous batching (story 7347; the Candle port of mlx-llm sc-7281).
//!
//! [`generate_batch`](crate::decode::generate_batch) (story 7255) is *synchronous*: a batch is
//! assembled, left-padded, decoded in lockstep to completion, and only then does the next batch start.
//! [`generate_continuous`] is *iteration-level*: it keeps up to `max_batch` sequences decoding at once
//! over **per-sequence** [`PagedKvCache`]s on one shared [`BlockPool`], and the moment a sequence
//! retires it prefills a waiting request into the freed slot — the batch never drains, and there is no
//! left-padding (each sequence attends only its own real KV). The host-side admission / retirement
//! policy is the backend-neutral [`core_llm::schedule::Scheduler`]; this module owns only the tensors.
//!
//! ## Two modes — an irreducible tradeoff (see [`BatchExactness`])
//! On a GPU, bit-exactness to a batch-1 run and throughput-scaling-with-occupancy are **mutually
//! exclusive**: the throughput win of batching *is* the restructured matmul reduction (amortizing the
//! weight reads across the batch), which is exactly what perturbs the floating-point result (Candle's
//! bf16 matmul is not M-invariant — the same caveat the batched and prefix-reuse paths already carry).
//! So:
//! - [`BatchExactness::Exact`] runs each sequence as its own batch-1 forward
//!   ([`CausalLm::decode_logits`] on that sequence's cache): **byte-identical** to running the
//!   request alone. The bit-exact equality assertions run against this mode.
//! - [`BatchExactness::Throughput`] batches the projections / MLP / lm_head and runs only attention
//!   per-sequence ([`CausalLm::decode_logits_per_seq`]): throughput scales with occupancy, at the
//!   cost of a row *tracking* (not matching) its batch-1 run — like `generate_batch`.
//!
//! Both modes get iteration-level admission (admit-on-retire) and per-sequence paged attention (no
//! padding mask, no max-context reservation).

use candle_core::{Device, Tensor};

use core_llm::schedule::{Scheduler, SeqId, SeqSpec};
use core_llm::FinishReason as CoreFinish;

use crate::decode::batch::BatchRequest;
use crate::decode::cancel::CancelFlag;
use crate::decode::stream::{default_seed, FinishReason, GenerationOutput, StreamEvent};
use crate::decode::{record_lane_token, LaneStep};
use crate::error::{Error, Result};
use crate::models::CausalLm;
use crate::primitives::input_ids;
use crate::primitives::kv_cache::KvCache;
use crate::primitives::sampler::{sample, SamplingParams, SplitMix64};
use crate::primitives::{BlockPool, PagedKvCache};

/// Numerical mode for [`generate_continuous`]'s decode forward — a throughput/exactness tradeoff that
/// is fundamental, not an implementation choice (see the module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BatchExactness {
    /// Each sequence is decoded as its own batch-1 forward, so a row's output is **byte-identical** to
    /// running that request alone. Throughput does not scale with occupancy. The default, and the mode
    /// the bit-exact equality assertions run against.
    #[default]
    Exact,
    /// The projections, MLP, and lm_head are **batched** over the active sequences and only attention
    /// runs per-sequence. Throughput scales with occupancy (weight reads amortized across the batch),
    /// but a row only **tracks** its batch-1 run — the batched matmul is not M-invariant on a GPU, so
    /// the logits diverge at sub-ULP (the same class as the `generate_batch` caveat).
    Throughput,
}

/// Configuration for [`generate_continuous`].
#[derive(Clone, Debug)]
pub struct ContinuousConfig {
    /// Maximum number of sequences decoding concurrently. Requests beyond this wait in admission order
    /// and fill a slot the instant one retires.
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

/// Per-sequence host state for one in-flight slot, alongside its own paged cache.
struct Lane {
    seq: SeqId,
    /// Request index == admission index == `seq.0`; kept explicit for the `on_event` callback.
    req_index: usize,
    cache: PagedKvCache,
    rng: SplitMix64,
    params: SamplingParams,
    /// Prompt + generated tokens (the repetition-penalty window the sampler reads).
    history: Vec<i32>,
    /// The token to feed at the next decode step (the most recently sampled token).
    next_token: i32,
}

/// Generate for many requests with **iteration-level continuous batching**: up to `config.max_batch`
/// sequences decode at once over per-sequence paged caches, and a retiring sequence's slot is
/// immediately refilled from the waiting requests. Returns a [`GenerationOutput`] per request, in
/// request order. `on_event` receives `(request_index, event)` as each row streams.
///
/// Returns [`Error::Canceled`] if `cancel` is already set before any inference. A mid-stream cancel
/// stops promptly: every request still decoding **or still waiting in the queue** finishes
/// [`FinishReason::Cancelled`] with whatever partial output it had, and each request emits exactly one
/// terminal [`StreamEvent::Done`].
pub fn generate_continuous(
    model: &CausalLm,
    requests: &[BatchRequest],
    config: &ContinuousConfig,
    cancel: &CancelFlag,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<Vec<GenerationOutput>> {
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

    let device = model.device().clone();
    let pool = BlockPool::new(config.block_size);
    let num_layers = model.config().num_layers;

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

    // Fill the initial slots. In `Throughput` mode a wave of newly-admitted requests sharing a prompt
    // length is prefilled **together** in one batched forward (story 7485); `Exact` mode and odd
    // lengths prefill batch-1 (no left-pad — each prompt at its own length).
    admit_to_capacity(
        model,
        &pool,
        num_layers,
        requests,
        &seq_ids,
        &mut next_req,
        &mut lanes,
        config.max_batch,
        config.exactness,
        &mut sched,
        &device,
        on_event,
    )?;

    // Decode loop: step every live lane, retire finished ones, refill freed slots from the queue.
    // Cancel is checked once at the top of each step (before another forward) so a mid-stream cancel
    // stops promptly without abandoning a forward that has already been computed.
    while !lanes.is_empty() {
        if cancel.is_cancelled() {
            break;
        }

        let per_lane = step_logits(model, &mut lanes, config.exactness, &device)?;

        let mut survivors: Vec<Lane> = Vec::with_capacity(lanes.len());
        for (mut lane, logits) in std::mem::take(&mut lanes).into_iter().zip(per_lane) {
            let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
            if let LaneStep::Continue = record_token(&mut sched, &mut lane, tok, on_event) {
                survivors.push(lane);
            }
        }
        lanes = survivors;

        // Admit-on-retire: refill every freed slot, batching a same-length wave's prefill (7485).
        admit_to_capacity(
            model,
            &pool,
            num_layers,
            requests,
            &seq_ids,
            &mut next_req,
            &mut lanes,
            config.max_batch,
            config.exactness,
            &mut sched,
            &device,
            on_event,
        )?;
    }

    // A cancel breaks the loop with live lanes still decoding and requests still waiting in the queue;
    // neither has emitted a `Done` yet (a request that retired normally emitted its `Done` as it
    // finished). Emit exactly one `Done` for each so every request signals completion. Lane indices
    // are `< next_req` and the queue is `next_req..`, so the two ranges are disjoint from each other
    // and from the already-retired requests — no request is signalled twice.
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

    // Assemble per-request outputs in request order from the scheduler's record.
    Ok(seq_ids
        .iter()
        .map(|&seq| GenerationOutput {
            tokens: sched.generated(seq).to_vec(),
            finish_reason: match sched.finish_reason(seq) {
                Some(CoreFinish::Stop) => FinishReason::StopToken,
                Some(CoreFinish::Length) => FinishReason::MaxTokens,
                // `None` ⇒ still active when a cancel broke the loop (or never admitted).
                _ => FinishReason::Cancelled,
            },
        })
        .collect())
}

/// Prefill request `ri` into a fresh lane and sample its first token. Returns `None` if the request
/// retired immediately (zero budget, or a stop/length on the first token), in which case the scheduler
/// already holds its final state.
#[allow(clippy::too_many_arguments)]
fn admit_lane(
    model: &CausalLm,
    pool: &std::rc::Rc<std::cell::RefCell<BlockPool>>,
    num_layers: usize,
    requests: &[BatchRequest],
    seq_ids: &[SeqId],
    ri: usize,
    sched: &mut Scheduler,
    device: &Device,
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

    let mut cache = PagedKvCache::with_pool(pool.clone(), num_layers);
    let logits = model.decode_logits(&input_ids(&r.prompt_ids, device)?, &mut cache, 0)?; // batch-1 prefill
    let mut lane = Lane {
        seq,
        req_index: ri,
        cache,
        rng: SplitMix64::new(r.seed.unwrap_or_else(default_seed)),
        params: r.sampling,
        history: r.prompt_ids.clone(),
        next_token: 0,
    };
    let tok = sample(&logits, &lane.history, &lane.params, &mut lane.rng, None)?;
    Ok(match record_token(sched, &mut lane, tok, on_event) {
        LaneStep::Continue => Some(lane),
        LaneStep::Done => None,
    })
}

/// Fill every free slot from the waiting requests. `Exact` admits one batch-1 [`admit_lane`] at a time
/// (bit-exact). `Throughput` (story 7485) prefills a wave of newly-admitted requests sharing a prompt
/// length **together** in one batched [`CausalLm::decode_logits_per_seq`] instead of N batch-1 forwards
/// — the lever that kills the inter-wave batch-1 prefill stalls (worst under bursty admit-on-retire,
/// e.g. uniform budgets where a whole wave retires at once). Differing lengths split into per-length
/// groups; singletons fall back to batch-1. Loops until the batch is full or the queue drains, so a
/// just-admitted request that retires on its own first token frees its slot again and is refilled in
/// the same call — matching the one-at-a-time loop it replaces.
#[allow(clippy::too_many_arguments)]
fn admit_to_capacity(
    model: &CausalLm,
    pool: &std::rc::Rc<std::cell::RefCell<BlockPool>>,
    num_layers: usize,
    requests: &[BatchRequest],
    seq_ids: &[SeqId],
    next_req: &mut usize,
    lanes: &mut Vec<Lane>,
    max_batch: usize,
    exactness: BatchExactness,
    sched: &mut Scheduler,
    device: &Device,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<()> {
    while lanes.len() < max_batch && *next_req < requests.len() {
        // The request indices that fill this pass's free slots, in admission order.
        let avail = max_batch - lanes.len();
        let mut wave: Vec<usize> = Vec::with_capacity(avail);
        while wave.len() < avail && *next_req < requests.len() {
            wave.push(*next_req);
            *next_req += 1;
        }
        match exactness {
            BatchExactness::Exact => {
                for ri in wave {
                    if let Some(lane) = admit_lane(
                        model, pool, num_layers, requests, seq_ids, ri, sched, device, on_event,
                    )? {
                        lanes.push(lane);
                    }
                }
            }
            BatchExactness::Throughput => admit_wave_throughput(
                model, pool, num_layers, requests, seq_ids, &wave, lanes, sched, device, on_event,
            )?,
        }
    }
    Ok(())
}

/// Prefill an admission `wave` for the `Throughput` mode (story 7485): group the requests by prompt
/// length and prefill each same-length group of ≥2 in **one** batched [`CausalLm::decode_logits_per_seq`]
/// (its projections / MLP / lm_head batched, attention per-sequence over fresh paged caches), sampling
/// each lane's first token from the shared `[group, vocab]` logits. Singletons (and any group of one)
/// prefill batch-1 via [`admit_lane`]. Zero-budget / already-retired requests are dropped first with
/// their terminal `Done` — exactly as [`admit_lane`] does — so each request still signals completion
/// exactly once. Like the `Throughput` decode step, a batched-prefilled row only *tracks* its batch-1
/// run on a GPU (the batched matmul is not M-invariant); on CPU it is bit-exact.
#[allow(clippy::too_many_arguments)]
fn admit_wave_throughput(
    model: &CausalLm,
    pool: &std::rc::Rc<std::cell::RefCell<BlockPool>>,
    num_layers: usize,
    requests: &[BatchRequest],
    seq_ids: &[SeqId],
    wave: &[usize],
    lanes: &mut Vec<Lane>,
    sched: &mut Scheduler,
    device: &Device,
    on_event: &mut dyn FnMut(usize, StreamEvent),
) -> Result<()> {
    use std::collections::BTreeMap;

    // Drop zero-budget / already-retired requests (the scheduler retires them at admission), emitting
    // their terminal `Done` — keep only those that still need a prefill.
    let mut active: Vec<usize> = Vec::with_capacity(wave.len());
    for &ri in wave {
        if sched.is_active(seq_ids[ri]) {
            active.push(ri);
        } else {
            on_event(
                ri,
                StreamEvent::Done {
                    reason: FinishReason::MaxTokens,
                    generated: 0,
                },
            );
        }
    }
    if active.is_empty() {
        return Ok(());
    }

    // Group by prompt length so each batched prefill is a clean uniform-width forward (no left-pad).
    // `BTreeMap` keeps groups in a deterministic order; within a group, admission order is preserved.
    let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &ri in &active {
        by_len
            .entry(requests[ri].prompt_ids.len())
            .or_default()
            .push(ri);
    }

    for (len, group) in by_len {
        if group.len() < 2 {
            // Singleton: a batched-of-1 forward buys nothing, so prefill batch-1 (the request is
            // already active, so `admit_lane` will not re-emit a zero-budget `Done`).
            if let Some(lane) = admit_lane(
                model, pool, num_layers, requests, seq_ids, group[0], sched, device, on_event,
            )? {
                lanes.push(lane);
            }
            continue;
        }

        // Batched prefill: `group.len()` fresh caches, one `decode_logits_per_seq` over `[g, len]`.
        let g = group.len();
        let mut caches: Vec<PagedKvCache> = (0..g)
            .map(|_| PagedKvCache::with_pool(pool.clone(), num_layers))
            .collect();
        let ids: Vec<u32> = group
            .iter()
            .flat_map(|&ri| requests[ri].prompt_ids.iter().map(|&t| t as u32))
            .collect();
        let ids = Tensor::from_vec(ids, (g, len), device)?;
        // Each row's RoPE positions are `0..len` — a fresh prefill at offset 0.
        let mut positions: Vec<i32> = Vec::with_capacity(g * len);
        for _ in 0..g {
            positions.extend(0..len as i32);
        }
        let logits = {
            let mut crefs: Vec<&mut PagedKvCache> = caches.iter_mut().collect();
            model.decode_logits_per_seq(&ids, &mut crefs, &positions)? // [g, vocab]
        };

        // Sample each lane's first token from its row and create the lane (mirrors `admit_lane`).
        for (gi, (cache, &ri)) in caches.into_iter().zip(group.iter()).enumerate() {
            let r = &requests[ri];
            let mut lane = Lane {
                seq: seq_ids[ri],
                req_index: ri,
                cache,
                rng: SplitMix64::new(r.seed.unwrap_or_else(default_seed)),
                params: r.sampling,
                history: r.prompt_ids.clone(),
                next_token: 0,
            };
            let lg = logits.narrow(0, gi, 1)?; // [1, vocab]
            let tok = sample(&lg, &lane.history, &lane.params, &mut lane.rng, None)?;
            if let LaneStep::Continue = record_token(sched, &mut lane, tok, on_event) {
                lanes.push(lane);
            }
        }
    }
    Ok(())
}

/// One decode step's logits, one `[1, vocab]` per live lane (in lane order). `Exact` runs each
/// sequence's own batch-1 forward; `Throughput` runs one batched forward with per-sequence attention
/// and splits the rows.
fn step_logits(
    model: &CausalLm,
    lanes: &mut [Lane],
    exactness: BatchExactness,
    device: &Device,
) -> Result<Vec<Tensor>> {
    match exactness {
        BatchExactness::Exact => {
            let mut logits = Vec::with_capacity(lanes.len());
            for lane in lanes.iter_mut() {
                let off = lane.cache.offset();
                let ids = input_ids(&[lane.next_token], device)?;
                logits.push(model.decode_logits(&ids, &mut lane.cache, off)?); // [1, vocab]
            }
            Ok(logits)
        }
        BatchExactness::Throughput => {
            let b = lanes.len();
            let feed: Vec<u32> = lanes.iter().map(|l| l.next_token as u32).collect();
            let ids = Tensor::from_vec(feed, (b, 1), device)?;
            let positions: Vec<i32> = lanes.iter().map(|l| l.cache.offset()).collect();
            let mut caches: Vec<&mut PagedKvCache> =
                lanes.iter_mut().map(|l| &mut l.cache).collect();
            let logits = model.decode_logits_per_seq(&ids, &mut caches, &positions)?; // [b, vocab]
            (0..b).map(|i| Ok(logits.narrow(0, i, 1)?)).collect() // [1, vocab] each
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

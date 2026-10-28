//! Typed, admission-aware **execution domains** for the request-scoped execution planner (sc-18317).
//!
//! [`crate::memory_strategy`] owns the five-rung *memory ladder* — which mechanism runs, and with
//! what parameter. This module owns the levers that are **not** rungs: knobs that change the
//! execution *schedule* of one forward pass without changing which components are resident. Epic
//! 18304's P2 planner selects them per request beside the ladder, so they ride the same
//! [`crate::GenerationMemory`] block and are gated by the same shared request floor.
//!
//! # Why these four, and why they are not rungs
//!
//! The first three existed already, as per-provider ad hoc parameters with no shared vocabulary, no
//! declaration, and therefore no way for a planner to discover them or for a caller to be told one
//! was ignored; the fourth is a new mechanism declared the same way from the start:
//!
//! | domain | prior form | equivalence class |
//! |---|---|---|
//! | [graph-evaluation cadence](GraphEvalCadence) | `eval_per_block` — a bool (every block, or never) — on `mlx_gen_flux2::MemoryConfig` (sc-6266) and `mlx_gen_wan::DitMemoryConfig` (sc-5681) | **bit-identical** |
//! | [CFG batching](CfgBatching) | `mlx_gen_kolors` `cfg_conditioning`/`cfg_batch_*`, `candle_gen_kolors::common::cfg_batch_context` (an implicit `cfg > 1.0` doubling convention) | **bit-identical** up to a row-count-dependent GEMM (see below) |
//! | [FFN chunking](FfnChunk) | `ffn_seq_chunk` on the same two config structs | *numerically* equivalent (see below) |
//! | [prefix KV cache](PrefixKvCacheMode) | none — no prefix cache before epic 25899; its port (sc-25903, sc-25904) is selected only through this domain (sc-25905), and an unset request runs the prefill path | *numerically* equivalent (see below) |
//!
//! The fourth row joined in sc-25905, for the reason [`CfgBatching`] is here: it trades a
//! request's activation high-water against its wall time, and a planner can only make that trade
//! if the knob is declared, selectable and refused where it does not exist.
//!
//! What deliberately did **not** become a domain: `DitMemoryConfig::attn_query_chunk`. Bounding
//! attention scratch is already the memory ladder's rung 3
//! ([`GenerationMemory::chunk_attention`](crate::GenerationMemory::chunk_attention) +
//! `attention_chunk_size`), and a second request-level control over the same mechanism would let a
//! selector and the ladder disagree about one request's attention budget.
//!
//! None of them sheds or bounds a *component's residency*, which is what the ladder's rungs are
//! defined over, and none of them has a place in the ladder's cost order (a cadence is not "more
//! expensive" than bounded decode — it is a different axis entirely). Adding a sixth rung would have
//! cost new conformance cells on every catalog entry for a knob no rung prerequisite mentions. They
//! are therefore a **parallel typed surface**: declared per provider on
//! [`Capabilities::execution`](crate::Capabilities::execution), selected per request on
//! [`GenerationMemory`], and refused — never silently ignored — by
//! [`Capabilities::validate_request`](crate::Capabilities::validate_request).
//!
//! # Fail closed, and defaults that are byte-for-byte the old behaviour
//!
//! Every field is an `Option` whose `None` (the `Default`) means *the provider's own historical
//! constant*, exactly like the SC-15510 ladder parameters. A request that sets none of them is
//! bit-identical to the same request before this module existed, on every provider.
//!
//! The other half is the tier-contract principle: a provider that cannot honour a *set* value must
//! **fail closed with an actionable refusal**, because silently executing a different schedule than
//! the planner selected produces a peak (or a wall-time) nobody predicted. The default
//! [`ExecutionSurface`] declares every domain [`ExecutionValueDomain::Unsupported`], so a provider
//! that never opts in refuses every request that names one of these knobs — declaration is opt-in,
//! refusal is the default.
//!
//! # Declaration is not reachability
//!
//! A provider may only declare a domain it actually *consumes*. The pairing is proven per family the
//! way the ladder families prove theirs: a weights-free production-path test asserts that a declared
//! value is admitted (and fails later, on the absent snapshot) while an out-of-domain or
//! undeclared-mechanism value is refused **by name** before any weight is opened. `Unsupported` is
//! the truthful state for a provider whose pipeline has no such mechanism, and it is deliberately
//! cheaper to declare than to fake.
//!
//! # Equivalence classes are part of the contract, not a footnote
//!
//! [`GraphEvalCadence`] only reorders *when* work is materialized, so it is exactly bit-identical
//! at every value. [`CfgBatching`] runs the same per-row arithmetic, but a sequential half hands
//! each GEMM half the rows the batched pair does, so it is bit-identical except where the backend
//! picks its matmul kernel by row count (measured ~1e-6 of peak on the MLX CPU device, sc-26268;
//! see [`CfgBatching`]). [`FfnChunk`] splits a GEMM along its row dimension, and
//! MLX's Metal kernels are tile-specialized by that dimension, so a chunked FFN is *numerically*
//! equivalent rather than bit-identical (cosine ≈ 1, max|Δ| ~1e-3 — the same class as the models'
//! own torch parity; see `mlx_gen_flux2::chunk`). A selector that must not perturb pixels at all
//! therefore leaves [`GenerationMemory::ffn_chunk`](crate::GenerationMemory::ffn_chunk) unset, which
//! is why the class is declared here rather than left for a caller to rediscover.
//!
//! [`PrefixKvCacheMode`] is in the same *numerically* equivalent class: a cached evaluation attends
//! over the stored prefix keys/values instead of recomputing them, which is the same mathematics
//! through a GEMM of a different row count (max|Δ| ~1e-6 in the latents on the Qwen-Image 2.1
//! fixtures — below one 8-bit level, but not bit-identical).

use std::num::NonZeroU32;

use crate::{Error, GenerationMemory, MemoryBudget, Result};

/// How often a provider forces its lazy graph to materialize **inside one denoise forward**, in
/// transformer blocks per forced evaluation.
///
/// A deferred-evaluation backend (MLX) builds the whole depth of a transformer forward as one lazy
/// graph, so its activation high-water is the *sum* of every block's transients rather than one
/// block's. Forcing an evaluation every `blocks` blocks caps it at roughly that many blocks' worth.
/// [`EVERY_BLOCK`](Self::EVERY_BLOCK) is the tightest bound; a larger cadence trades peak for fewer
/// synchronization points, and the field being **unset** is the whole-forward graph (no forced
/// evaluation at all — the historical shipped behaviour of every provider).
///
/// **Bit-identical at every cadence.** Forcing evaluation changes only *when* a value is computed,
/// never what it is, so a cadence sweep moves the memory schedule and the step time and leaves the
/// pixels alone. That is what makes this safe for a planner to choose freely — and it is why the
/// cadence's own effect differs by lane: for image families the request peak is usually bound by the
/// decode rather than the denoise, so cadence reads as a **speed** knob (fewer synchronization
/// points), while for weight-bound requests and long-sequence video it is a genuine memory lever.
///
/// # What this is *not*
///
/// It is **not** the per-*step* latent evaluation some providers run at the end of a denoise step
/// (`mlx_gen_sdxl::pipeline::denoise_core`'s `latents.eval()`). That one is load-bearing for
/// output identity, not just for scheduling: it materializes the global-RNG split between steps, and
/// deferring it perturbs an ancestral sampler's draws (sc-2400 S5). A provider whose per-step
/// evaluation carries that meaning must **not** declare this domain over it. The unit here is
/// deliberately named in *blocks* so the two cannot be conflated at a call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphEvalCadence(NonZeroU32);

impl GraphEvalCadence {
    /// Force an evaluation after every block — the tightest activation bound, and the cadence the
    /// pre-sc-18317 `eval_per_block: true` configs expressed.
    pub const EVERY_BLOCK: Self = Self(NonZeroU32::MIN);

    /// A cadence of `blocks` blocks per forced evaluation. Zero is not a cadence (it would mean
    /// "evaluate never", which is the field being unset) and is rejected rather than clamped.
    pub fn new(blocks: u32) -> Result<Self> {
        NonZeroU32::new(blocks).map(Self).ok_or_else(|| {
            Error::Msg(
                "graph_eval_cadence must be >= 1 block; leave the field unset for the \
                 whole-forward lazy graph"
                    .to_owned(),
            )
        })
    }

    /// Infallible constructor for a statically non-zero cadence.
    pub const fn from_nonzero(blocks: NonZeroU32) -> Self {
        Self(blocks)
    }

    /// Blocks per forced evaluation.
    pub const fn blocks(self) -> u32 {
        self.0.get()
    }

    /// Whether the block at zero-based `index` within one stack is an evaluation boundary.
    ///
    /// The single decision point for every consumer, so a provider cannot get the off-by-one wrong
    /// in one loop and right in another: with [`EVERY_BLOCK`](Self::EVERY_BLOCK) this is `true` for
    /// every index, reproducing the pre-sc-18317 `if mem.eval_per_block` branch exactly. The index
    /// is per *stack* — a provider with separate double/single block stacks restarts the count at
    /// each, because the two run over different tensors and the boundary between them is already an
    /// evaluation point.
    pub const fn evaluates_after_block(self, index: usize) -> bool {
        (index as u64 + 1).is_multiple_of(self.blocks() as u64)
    }
}

/// Sequence rows per chunk of a chunked feed-forward (FFN) intermediate.
///
/// A transformer FFN materializes a `[rows, ffn_dim]` intermediate that is frequently the largest
/// single denoise transient; running it over sequence row-blocks bounds that transient to
/// `[rows, ffn_dim]` for this many rows at a time. The FFN is per-token, so the mathematics is
/// unchanged — but see the module docs: this is the one domain here that is *numerically* equivalent
/// rather than bit-identical.
///
/// A chunk at least as large as the sequence is a no-op by construction (one whole-sequence call), so
/// a provider never has to special-case an over-large value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FfnChunk(NonZeroU32);

impl FfnChunk {
    /// A chunk of `rows` sequence rows. Zero is rejected rather than clamped: a zero-row chunk is not
    /// "no chunking" (that is the field being unset), it is a malformed request.
    pub fn new(rows: u32) -> Result<Self> {
        NonZeroU32::new(rows).map(Self).ok_or_else(|| {
            Error::Msg(
                "ffn_chunk must be >= 1 sequence row; leave the field unset for the \
                 whole-sequence FFN"
                    .to_owned(),
            )
        })
    }

    /// Infallible constructor for a statically non-zero chunk.
    pub const fn from_nonzero(rows: NonZeroU32) -> Self {
        Self(rows)
    }

    /// Sequence rows per chunk.
    pub const fn rows(self) -> u32 {
        self.0.get()
    }

    /// Sequence rows per chunk, in the `usize` the tensor-side chunk helpers take.
    pub const fn rows_usize(self) -> usize {
        self.0.get() as usize
    }
}

/// How a provider evaluates the conditional and unconditional halves of classifier-free guidance.
///
/// The two are mathematically identical — the same two forwards over the same inputs, combined by the
/// same rule — and differ only in whether the accelerator sees them as one batched call or two:
///
/// * [`Batched`](Self::Batched) doubles the batch (`[cond, uncond]`) and runs **one** forward. Fewer
///   kernel launches, but every activation is held for two rows at once.
/// * [`Sequential`](Self::Sequential) runs **two** batch-1 forwards. Roughly halves the activation
///   high-water of the guided denoise at the cost of a second pass over the weights.
///
/// This is the axis a planner wants on a constrained device: batched CFG doubles exactly the
/// transients the memory ladder's rungs 2-4 exist to bound. It is deliberately a closed enum rather
/// than a bool — `cfg_batch: false` reads as "no CFG" at a call site, which is a third, unrelated
/// state (guidance off) that neither variant here describes.
///
/// **Bit-identical where a provider implements both, up to the backend's GEMM.** The one admitted
/// deviation is a matrix multiply whose reduction order depends on its row count: a sequential
/// half runs `M = tokens` rows where the batched pair runs `2·tokens`, and a BLAS is free to pick a
/// different kernel (GEMV for one row, a different blocking for a few) for each. That is float reassociation of the same
/// arithmetic — sc-26268 measured ~1e-6 of peak on the MLX CPU device, where a lone row and row 0
/// of a two-row product differ in the last ulps — never a different computation. A provider test
/// bounds it at reassociation scale (the SDXL real-weight check asserts the 8-bit image unchanged);
/// anything above that is a provider bug (a row mix-up, a coupled batch statistic).
///
/// A provider that implements only one mode declares only that one: [`CfgBatchingDomain::Modes`]
/// is a closed candidate list precisely so the missing mode is a typed refusal instead of a silent
/// fallback to the mode that exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CfgBatching {
    /// One forward over a `[cond, uncond]` batch — the convention every CFG provider in this
    /// workspace shipped before sc-18317, and therefore the effective default everywhere.
    Batched,
    /// Two batch-1 forwards, conditional then unconditional.
    Sequential,
}

impl CfgBatching {
    /// Every mode, for exhaustive declaration/conformance walks.
    pub const ALL: [Self; 2] = [Self::Batched, Self::Sequential];

    /// `true` for the doubled-batch single forward.
    pub const fn is_batched(self) -> bool {
        matches!(self, Self::Batched)
    }

    /// The batch dimension a guided forward runs at under this mode.
    pub const fn guided_batch(self) -> u32 {
        match self {
            Self::Batched => 2,
            Self::Sequential => 1,
        }
    }

    /// Stable lower-case label for messages and structured plan events.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Batched => "batched",
            Self::Sequential => "sequential",
        }
    }
}

/// Whether a block-causal denoiser keeps the **prefix key/value cache** across one denoise loop
/// (sc-25905).
///
/// A block-causal joint transformer (Qwen-Image 2.1) attends from the target image's tokens to a
/// *prefix* — the prompt and every reference image — that does not change between denoise steps.
/// The two modes are the same mathematics and differ in what is held and what is recomputed:
///
/// * [`Cached`](Self::Cached) evaluates the prefix once per prompt branch, keeps every block's
///   prefix K and V for the length of the denoise loop, and runs every later model evaluation over
///   the target tokens only. Fewer tokens per step, at the cost of holding
///   `2 · blocks · prefix_tokens · inner_dim` activations per prompt branch.
/// * [`Prefill`](Self::Prefill) holds nothing and evaluates the whole joint sequence at every
///   model evaluation — the path such a provider ran before it had a cache.
///
/// This is the axis a planner wants when a request with many reference images does not fit: the
/// cache is activations at the compute dtype, so no numeric tier shrinks it, and shedding it costs
/// time, never an image, a reference or a tier. Like [`CfgBatching`] it is a closed enum rather
/// than a bool, and a provider that implements only one mode declares only that one.
///
/// **Numerically equivalent, not bit-identical** — see the module docs.
///
/// # The unset field is the prefill path
///
/// `None` on [`GenerationMemory::prefix_kv_cache`] means the provider's own default like every
/// other domain here, and for a cache that default is deliberately the mode that **holds
/// nothing**: a request admitted against a budget that never priced the cache must not run above
/// it. The cache is therefore opt-in per request — a caller selects [`Cached`](Self::Cached) after
/// pricing it ([`Self::select`]).
///
/// # What a caller must get right, because nothing checks it for them
///
/// * **Write the selection after the request scope has configured the request.**
///   [`MemoryRequestScope::configure_request`](crate::MemoryRequestScope::configure_request)
///   *replaces* [`GenerationRequest::memory`](crate::GenerationRequest::memory) with the block
///   for the admitted ladder selection, and for a `Resident` selection on a provider that
///   preserves its load defaults that block is `None`. A selection written before that call is
///   lost and the render runs prefill, in silence — the report says so, the request does not. So:
///   configure, then set the field, creating a default block if there is none.
/// * **Admit the cached demand before selecting `Cached`.** A provider honours the field; it
///   does not verify that the request was admitted against a budget that included the cache.
///
/// # A provider prices an upper bound, and holds a cached render to it
///
/// The real prefix is known only once the prompt is tokenized and the references are fitted, so
/// a provider prices the cache over published **bounds** (text rows per branch, tokens per
/// reference). A `Cached` request whose real caches would exceed that priced figure is run
/// **prefill** — never cached above its price, never failed — and the render's
/// [`PrefixKvCacheReport::overflow`] says so.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrefixKvCacheMode {
    /// Prefill once per prompt branch, then decode the target tokens against the cached prefix.
    Cached,
    /// Evaluate the full joint sequence at every model evaluation; hold no cache.
    Prefill,
}

impl PrefixKvCacheMode {
    /// Every mode, for exhaustive declaration/conformance walks.
    pub const ALL: [Self; 2] = [Self::Cached, Self::Prefill];

    /// `true` for the mode that holds the cache.
    pub const fn is_cached(self) -> bool {
        matches!(self, Self::Cached)
    }

    /// Stable lower-case label for messages, structured plan events and job records.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Prefill => "prefill",
        }
    }

    /// The mode a request runs under `budget`: the cheapest in **time** that fits.
    ///
    /// `shed_demand_bytes` is the request's incremental live demand with the cache shed — the
    /// number [`MemoryBudget::fits`] has always been asked about — and `cache_bytes` is what the
    /// provider prices the request's caches at on top of it.
    ///
    /// * [`Cached`](Self::Cached) when `shed_demand_bytes + cache_bytes` fits;
    /// * [`Prefill`](Self::Prefill) when only `shed_demand_bytes` fits — the request is still
    ///   admitted, it runs without the cache;
    /// * `None` when even the shed demand does not fit: the caller refuses exactly as it did
    ///   before the cache existed. Nothing here shrinks an image, drops a reference or changes a
    ///   tier.
    ///
    /// The caller stays the owner of the budget and of the ladder; this is only the two-way
    /// comparison, written once so a provider's tests and its consumer cannot disagree on a
    /// boundary (exact fits are accepted, as [`MemoryBudget::fits`] accepts them, and an
    /// overflowing sum does not fit).
    pub fn select(budget: MemoryBudget, shed_demand_bytes: u64, cache_bytes: u64) -> Option<Self> {
        if !budget.fits(shed_demand_bytes) {
            return None;
        }
        match shed_demand_bytes.checked_add(cache_bytes) {
            Some(cached_demand_bytes) if budget.fits(cached_demand_bytes) => Some(Self::Cached),
            _ => Some(Self::Prefill),
        }
    }
}

/// Which prefix-KV-cache path one render ran, and what its caches held
/// ([`GenerationReport::prefix_kv_cache`](crate::GenerationReport::prefix_kv_cache), sc-25905).
///
/// Written from what the denoise loop did, not from what the request asked for, so a consumer can
/// record the path on the job and compare it with the mode it selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixKvCacheReport {
    /// The path the denoise loop took.
    pub path: PrefixKvCacheMode,
    /// Bytes of prefix K/V held at once (every prompt branch of one image);
    /// `0` on [`PrefixKvCacheMode::Prefill`]. A request for several images renders them one after
    /// another, each with its own caches, so this does not grow with the image count.
    pub cache_bytes: u64,
    /// `Some` when the request selected [`PrefixKvCacheMode::Cached`] but the provider ran
    /// [`PrefixKvCacheMode::Prefill`] because the request's real caches would have exceeded what
    /// the provider prices for it. `None` on every other render — in particular a `Prefill` path
    /// with `None` here is prefill **by selection** (or by an unset field).
    pub overflow: Option<PrefixKvCacheOverflow>,
}

/// Why a `Cached` selection ran prefill ([`PrefixKvCacheReport::overflow`]): the caches this
/// request would really have held against the figure the provider prices for it. Both are in the
/// loaded model's own terms (its block count, width and compute dtype), so they compare directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixKvCacheOverflow {
    /// Bytes the request's caches would have held, from its real prefix (tokenized prompt rows
    /// plus every fitted reference grid, per prompt branch) at the model's compute dtype.
    pub actual_bytes: u64,
    /// Bytes the provider prices for this request: the same arithmetic over its published
    /// per-branch text bound and per-reference bound.
    pub priced_bytes: u64,
}

/// The production domain of
/// [`GenerationMemory::prefix_kv_cache`](crate::GenerationMemory::prefix_kv_cache).
///
/// A closed mode list, for the reason [`CfgBatchingDomain`] is one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PrefixKvCacheDomain {
    /// The provider has no selectable prefix KV cache. Every set value is refused.
    #[default]
    Unsupported,
    /// Exactly these modes are implemented. An empty list is a declaration error.
    Modes(Vec<PrefixKvCacheMode>),
}

impl PrefixKvCacheDomain {
    /// Whether the provider declares a selectable prefix KV cache.
    pub const fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Whether `mode` is implemented.
    pub fn accepts(&self, mode: PrefixKvCacheMode) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Modes(modes) => modes.contains(&mode),
        }
    }

    /// Human-readable domain for a refusal message.
    pub fn describe(&self) -> String {
        match self {
            Self::Unsupported => "unsupported".to_owned(),
            Self::Modes(modes) => {
                let labels: Vec<&str> = modes.iter().map(|mode| mode.label()).collect();
                format!("[{}]", labels.join(", "))
            }
        }
    }
}

/// The production domain of one **numeric** execution knob.
///
/// Two admitting shapes, because the two knobs here differ in what evidence exists rather than in
/// how they are checked:
///
/// * [`Candidates`](Self::Candidates) is the ladder's shape — exactly these measured values, nothing
///   else. Use it as soon as a provider has evidence for specific operating points.
/// * [`AtLeast`](Self::AtLeast) says the mechanism is exact by construction over the whole range from
///   that floor up, and claims **no** per-value measurement. It is the honest declaration for the
///   pre-existing sc-6266 knobs, whose helpers accept any positive value (an over-large one degrades
///   to the un-chunked fast path) and for which the epic measured no candidate set. Narrowing to
///   `Candidates` later is a tightening, never a behaviour change for an in-domain value.
///
/// [`Unsupported`](Self::Unsupported) is the `Default`, so a provider gains a fail-closed refusal for
/// a mechanism it does not have without writing anything.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ExecutionValueDomain {
    /// The provider has no such mechanism. Every set value is refused.
    #[default]
    Unsupported,
    /// Any value `>= floor` is accepted; no candidate set is claimed as measured.
    AtLeast(NonZeroU32),
    /// Exactly these values are accepted. An empty list is a declaration error, not a silent
    /// `Unsupported` — see [`ExecutionSurface::declaration_errors`].
    Candidates(Vec<u32>),
}

impl ExecutionValueDomain {
    /// `AtLeast(1)`: the mechanism accepts every positive value.
    pub const ANY_POSITIVE: Self = Self::AtLeast(NonZeroU32::MIN);

    /// Whether the provider declares the mechanism at all.
    pub const fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Whether `value` is inside the declared domain. Always `false` for
    /// [`Unsupported`](Self::Unsupported).
    pub fn accepts(&self, value: u32) -> bool {
        match self {
            Self::Unsupported => false,
            Self::AtLeast(floor) => value >= floor.get(),
            Self::Candidates(candidates) => candidates.contains(&value),
        }
    }

    /// Human-readable domain for a refusal message.
    pub fn describe(&self) -> String {
        match self {
            Self::Unsupported => "unsupported".to_owned(),
            Self::AtLeast(floor) => format!(">= {}", floor.get()),
            Self::Candidates(candidates) => format!("{candidates:?}"),
        }
    }
}

/// The production domain of [`GenerationMemory::cfg_batching`](crate::GenerationMemory::cfg_batching).
///
/// A closed mode list rather than an [`ExecutionValueDomain`]: the values are not numeric, and the
/// interesting refusal is *within* a supporting provider (one that batches CFG but has no sequential
/// path) rather than only at a provider with no CFG at all.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum CfgBatchingDomain {
    /// The provider has no selectable CFG batching (it may have no guidance at all, or exactly one
    /// hard-wired convention it does not expose). Every set value is refused.
    #[default]
    Unsupported,
    /// Exactly these modes are implemented. An empty list is a declaration error.
    Modes(Vec<CfgBatching>),
}

impl CfgBatchingDomain {
    /// Whether the provider declares selectable CFG batching.
    pub const fn is_supported(&self) -> bool {
        !matches!(self, Self::Unsupported)
    }

    /// Whether `mode` is implemented.
    pub fn accepts(&self, mode: CfgBatching) -> bool {
        match self {
            Self::Unsupported => false,
            Self::Modes(modes) => modes.contains(&mode),
        }
    }

    /// Human-readable domain for a refusal message.
    pub fn describe(&self) -> String {
        match self {
            Self::Unsupported => "unsupported".to_owned(),
            Self::Modes(modes) => {
                let labels: Vec<&str> = modes.iter().map(|mode| mode.label()).collect();
                format!("[{}]", labels.join(", "))
            }
        }
    }
}

/// One provider's declared execution-domain surface, carried on
/// [`Capabilities::execution`](crate::Capabilities::execution).
///
/// `Default` declares nothing, which is the fail-closed state: a provider that has not opted a
/// mechanism in refuses every request that selects it. Opting in is one field, and the family's own
/// reachability test is what keeps the declaration honest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionSurface {
    /// Blocks-per-forced-evaluation cadences this provider honours inside its denoise forward.
    pub graph_eval_cadence_blocks: ExecutionValueDomain,
    /// Sequence-row FFN chunks this provider honours.
    pub ffn_chunk_rows: ExecutionValueDomain,
    /// CFG batching modes this provider implements.
    pub cfg_batching: CfgBatchingDomain,
    /// Prefix-KV-cache modes this provider implements (sc-25905).
    pub prefix_kv_cache: PrefixKvCacheDomain,
}

impl ExecutionSurface {
    /// `true` when the provider declares no execution domain at all — every set value is refused.
    pub const fn is_inert(&self) -> bool {
        !self.graph_eval_cadence_blocks.is_supported()
            && !self.ffn_chunk_rows.is_supported()
            && !self.cfg_batching.is_supported()
            && !self.prefix_kv_cache.is_supported()
    }

    /// Gate one request's execution selections against this surface.
    ///
    /// Called by the shared request floor
    /// ([`Capabilities::validate_request`](crate::Capabilities::validate_request)) rather than by
    /// each provider, so a provider cannot forget it and a knob cannot be silently dropped. Every
    /// rejection is the typed [`Error::Unsupported`] — these are capability gaps, not range
    /// violations (F-008), and a consumer distinguishes "this provider cannot schedule that way"
    /// from a malformed value by the variant.
    ///
    /// `None`/unset fields validate vacuously: that is the historical fast path, and it is what makes
    /// this addition byte-for-byte inert for every existing request.
    pub fn validate(&self, id: &str, memory: Option<&GenerationMemory>) -> Result<()> {
        let Some(memory) = memory else {
            return Ok(());
        };
        if let Some(cadence) = memory.graph_eval_cadence {
            self.check_numeric(
                id,
                "graph_eval_cadence",
                cadence.blocks(),
                "blocks per forced graph evaluation",
                &self.graph_eval_cadence_blocks,
            )?;
        }
        if let Some(chunk) = memory.ffn_chunk {
            self.check_numeric(
                id,
                "ffn_chunk",
                chunk.rows(),
                "sequence rows per chunked FFN",
                &self.ffn_chunk_rows,
            )?;
        }
        if let Some(mode) = memory.cfg_batching {
            if !self.cfg_batching.is_supported() {
                return Err(Error::Unsupported(format!(
                    "{id}: cfg_batching={} was selected, but this provider declares no selectable \
                     CFG batching. Leave memory.cfg_batching unset to run the provider's own \
                     convention.",
                    mode.label()
                )));
            }
            if !self.cfg_batching.accepts(mode) {
                return Err(Error::Unsupported(format!(
                    "{id}: cfg_batching={} is outside the declared production domain {}. Select a \
                     declared mode, or leave memory.cfg_batching unset for the provider's own \
                     convention.",
                    mode.label(),
                    self.cfg_batching.describe()
                )));
            }
        }
        if let Some(mode) = memory.prefix_kv_cache {
            if !self.prefix_kv_cache.is_supported() {
                return Err(Error::Unsupported(format!(
                    "{id}: prefix_kv_cache={} was selected, but this provider declares no \
                     selectable prefix KV cache. Leave memory.prefix_kv_cache unset to run the \
                     provider's own default.",
                    mode.label()
                )));
            }
            if !self.prefix_kv_cache.accepts(mode) {
                return Err(Error::Unsupported(format!(
                    "{id}: prefix_kv_cache={} is outside the declared production domain {}. \
                     Select a declared mode, or leave memory.prefix_kv_cache unset for the \
                     provider's own default.",
                    mode.label(),
                    self.prefix_kv_cache.describe()
                )));
            }
        }
        Ok(())
    }

    fn check_numeric(
        &self,
        id: &str,
        field: &str,
        value: u32,
        unit: &str,
        domain: &ExecutionValueDomain,
    ) -> Result<()> {
        if !domain.is_supported() {
            return Err(Error::Unsupported(format!(
                "{id}: {field}={value} was selected, but this provider declares no {field} \
                 mechanism ({unit}). Leave memory.{field} unset to run the provider's own default."
            )));
        }
        if !domain.accepts(value) {
            return Err(Error::Unsupported(format!(
                "{id}: {field}={value} is outside the declared production domain {}. Select a \
                 declared value, or leave memory.{field} unset for the provider's own default.",
                domain.describe()
            )));
        }
        Ok(())
    }

    /// Static declaration coherence. An empty result means the surface is internally consistent; it
    /// does **not** mean any measurement exists.
    ///
    /// The one defect this catches is an empty candidate list, which would read as a supported
    /// mechanism while refusing every value — a declaration that cannot be satisfied is worse than
    /// [`ExecutionValueDomain::Unsupported`], because a planner would keep offering the knob.
    pub fn declaration_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (field, domain) in [
            ("graph_eval_cadence_blocks", &self.graph_eval_cadence_blocks),
            ("ffn_chunk_rows", &self.ffn_chunk_rows),
        ] {
            if let ExecutionValueDomain::Candidates(candidates) = domain {
                if candidates.is_empty() {
                    errors.push(format!(
                        "{field} declares an empty candidate list; use Unsupported to declare the \
                         mechanism absent"
                    ));
                }
                if candidates.contains(&0) {
                    errors.push(format!("{field} declares a zero candidate"));
                }
                let mut unique = candidates.clone();
                unique.sort_unstable();
                unique.dedup();
                if unique.len() != candidates.len() {
                    errors.push(format!("{field} declares duplicate candidates"));
                }
            }
        }
        if let CfgBatchingDomain::Modes(modes) = &self.cfg_batching {
            if modes.is_empty() {
                errors.push(
                    "cfg_batching declares an empty mode list; use Unsupported to declare the \
                     mechanism absent"
                        .to_owned(),
                );
            }
            let mut unique = modes.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != modes.len() {
                errors.push("cfg_batching declares duplicate modes".to_owned());
            }
        }
        if let PrefixKvCacheDomain::Modes(modes) = &self.prefix_kv_cache {
            if modes.is_empty() {
                errors.push(
                    "prefix_kv_cache declares an empty mode list; use Unsupported to declare the \
                     mechanism absent"
                        .to_owned(),
                );
            }
            let mut unique = modes.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() != modes.len() {
                errors.push("prefix_kv_cache declares duplicate modes".to_owned());
            }
        }
        errors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> GenerationMemory {
        GenerationMemory::default()
    }

    #[test]
    fn default_memory_selects_no_execution_domain() {
        let m = memory();
        assert_eq!(m.graph_eval_cadence, None);
        assert_eq!(m.ffn_chunk, None);
        assert_eq!(m.cfg_batching, None);
        assert_eq!(m.prefix_kv_cache, None);
    }

    #[test]
    fn zero_is_not_a_cadence_or_a_chunk() {
        assert!(GraphEvalCadence::new(0).is_err());
        assert!(FfnChunk::new(0).is_err());
        assert_eq!(
            GraphEvalCadence::new(1).unwrap(),
            GraphEvalCadence::EVERY_BLOCK
        );
        assert_eq!(FfnChunk::new(7).unwrap().rows(), 7);
        assert_eq!(FfnChunk::new(7).unwrap().rows_usize(), 7);
    }

    #[test]
    fn every_block_cadence_is_a_boundary_at_every_index() {
        let cadence = GraphEvalCadence::EVERY_BLOCK;
        for index in 0..8 {
            assert!(
                cadence.evaluates_after_block(index),
                "EVERY_BLOCK must reproduce the pre-sc-18317 per-block branch at {index}"
            );
        }
    }

    #[test]
    fn wider_cadence_boundaries_land_on_multiples() {
        let cadence = GraphEvalCadence::new(3).unwrap();
        let boundaries: Vec<usize> = (0..9)
            .filter(|index| cadence.evaluates_after_block(*index))
            .collect();
        assert_eq!(boundaries, vec![2, 5, 8]);
    }

    #[test]
    fn cfg_batching_labels_and_batch_are_stable() {
        assert_eq!(CfgBatching::Batched.label(), "batched");
        assert_eq!(CfgBatching::Sequential.label(), "sequential");
        assert_eq!(CfgBatching::Batched.guided_batch(), 2);
        assert_eq!(CfgBatching::Sequential.guided_batch(), 1);
        assert!(CfgBatching::Batched.is_batched());
        assert!(!CfgBatching::Sequential.is_batched());
    }

    #[test]
    fn unsupported_domain_accepts_nothing() {
        let domain = ExecutionValueDomain::default();
        assert!(!domain.is_supported());
        assert!(!domain.accepts(1));
        assert!(!domain.accepts(1024));
        assert!(!CfgBatchingDomain::default().accepts(CfgBatching::Batched));
    }

    #[test]
    fn domain_shapes_admit_exactly_what_they_declare() {
        let any = ExecutionValueDomain::ANY_POSITIVE;
        assert!(any.accepts(1));
        assert!(any.accepts(u32::MAX));

        let floored = ExecutionValueDomain::AtLeast(NonZeroU32::new(64).unwrap());
        assert!(!floored.accepts(63));
        assert!(floored.accepts(64));

        let exact = ExecutionValueDomain::Candidates(vec![128, 256]);
        assert!(exact.accepts(256));
        assert!(!exact.accepts(255));
    }

    #[test]
    fn inert_surface_refuses_every_domain_by_name() {
        let surface = ExecutionSurface::default();
        assert!(surface.is_inert());

        let cadence = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::EVERY_BLOCK),
            ..memory()
        };
        let error = surface
            .validate("prov", Some(&cadence))
            .expect_err("an inert surface must refuse a cadence");
        assert!(matches!(error, Error::Unsupported(_)), "{error:?}");
        let text = error.to_string();
        assert!(text.contains("prov"), "{text}");
        assert!(text.contains("graph_eval_cadence"), "{text}");
        assert!(text.contains("unset"), "actionable remedy missing: {text}");
        // The reason must be the ABSENT MECHANISM, not an out-of-domain value: the two have different
        // fixes (implement/declare the mechanism vs pick another value), and a refusal that reports
        // the wrong one sends a caller down the wrong path. Pinning the wording is also what makes a
        // mutation that deletes the unsupported branch — leaving the range check to reject everything
        // by accident — a visible failure rather than a silent behaviour change.
        assert!(
            text.contains("declares no graph_eval_cadence mechanism"),
            "an inert surface must refuse for the absent mechanism: {text}"
        );

        let chunk = GenerationMemory {
            ffn_chunk: Some(FfnChunk::new(512).unwrap()),
            ..memory()
        };
        let text = surface
            .validate("prov", Some(&chunk))
            .expect_err("an inert surface must refuse an FFN chunk")
            .to_string();
        assert!(text.contains("ffn_chunk"), "{text}");
        assert!(text.contains("unset"), "{text}");
        assert!(
            text.contains("declares no ffn_chunk mechanism"),
            "an inert surface must refuse for the absent mechanism: {text}"
        );

        let cfg = GenerationMemory {
            cfg_batching: Some(CfgBatching::Batched),
            ..memory()
        };
        let text = surface
            .validate("prov", Some(&cfg))
            .expect_err("an inert surface must refuse a CFG mode")
            .to_string();
        assert!(text.contains("cfg_batching"), "{text}");
        assert!(text.contains("batched"), "{text}");
        assert!(
            text.contains("declares no selectable CFG batching"),
            "an inert surface must refuse for the absent mechanism: {text}"
        );
    }

    /// sc-25905: the prefix KV cache is refused by name where it is not declared, and a declared
    /// surface admits exactly its modes.
    ///
    /// *Mutations that red this:* dropping the `prefix_kv_cache` arm of `validate` (an undeclared
    /// selection is silently accepted); dropping the `accepts` check (an undeclared mode runs).
    #[test]
    fn prefix_kv_cache_is_refused_by_name_and_admitted_only_in_its_declared_modes() {
        let cached = GenerationMemory {
            prefix_kv_cache: Some(PrefixKvCacheMode::Cached),
            ..memory()
        };
        let text = ExecutionSurface::default()
            .validate("prov", Some(&cached))
            .expect_err("an inert surface must refuse a prefix KV cache selection")
            .to_string();
        assert!(text.contains("prov"), "{text}");
        assert!(text.contains("prefix_kv_cache=cached"), "{text}");
        assert!(text.contains("unset"), "actionable remedy missing: {text}");
        assert!(
            text.contains("declares no selectable prefix KV cache"),
            "an inert surface must refuse for the absent mechanism: {text}"
        );

        let prefill_only = ExecutionSurface {
            prefix_kv_cache: PrefixKvCacheDomain::Modes(vec![PrefixKvCacheMode::Prefill]),
            ..ExecutionSurface::default()
        };
        assert!(!prefill_only.is_inert());
        assert!(prefill_only.declaration_errors().is_empty());
        let text = prefill_only
            .validate("prov", Some(&cached))
            .expect_err("an undeclared mode must be refused")
            .to_string();
        assert!(text.contains("prefix_kv_cache=cached"), "{text}");
        assert!(text.contains("[prefill]"), "domain must be named: {text}");
        prefill_only
            .validate(
                "prov",
                Some(&GenerationMemory {
                    prefix_kv_cache: Some(PrefixKvCacheMode::Prefill),
                    ..memory()
                }),
            )
            .expect("a declared mode must be admitted");

        let both = ExecutionSurface {
            prefix_kv_cache: PrefixKvCacheDomain::Modes(PrefixKvCacheMode::ALL.to_vec()),
            ..ExecutionSurface::default()
        };
        both.validate("prov", Some(&cached))
            .expect("a declared mode must be admitted");
        assert_eq!(PrefixKvCacheMode::Cached.label(), "cached");
        assert_eq!(PrefixKvCacheMode::Prefill.label(), "prefill");
        assert!(PrefixKvCacheMode::Cached.is_cached());
        assert!(!PrefixKvCacheMode::Prefill.is_cached());

        let duplicated = ExecutionSurface {
            prefix_kv_cache: PrefixKvCacheDomain::Modes(vec![
                PrefixKvCacheMode::Cached,
                PrefixKvCacheMode::Cached,
            ]),
            ..ExecutionSurface::default()
        };
        assert!(
            duplicated
                .declaration_errors()
                .iter()
                .any(|error| error.contains("prefix_kv_cache") && error.contains("duplicate")),
            "{:?}",
            duplicated.declaration_errors()
        );
    }

    /// sc-25905: the two-way fit comparison, on its boundaries. Exact fits are accepted, exactly
    /// as [`MemoryBudget::fits`] accepts them.
    ///
    /// *Mutations that red this:* ignoring `cache_bytes` (the between-budget row answers
    /// `Cached`); answering `Prefill` when the shed demand does not fit (the last rows answer
    /// `Some`); always answering `Prefill` (the first rows answer `Prefill`).
    #[test]
    fn prefix_kv_cache_select_takes_the_cache_only_when_it_fits_and_refuses_when_neither_does() {
        let budget = |total_bytes| MemoryBudget {
            total_bytes,
            committed_bytes: 0,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 0,
        };
        let (shed, cache) = (1_000_u64, 300_u64);
        for (total, want) in [
            (u64::MAX, Some(PrefixKvCacheMode::Cached)),
            (shed + cache, Some(PrefixKvCacheMode::Cached)),
            (shed + cache - 1, Some(PrefixKvCacheMode::Prefill)),
            (shed, Some(PrefixKvCacheMode::Prefill)),
            (shed - 1, None),
            (0, None),
        ] {
            assert_eq!(
                PrefixKvCacheMode::select(budget(total), shed, cache),
                want,
                "budget {total} against shed {shed} + cache {cache}"
            );
        }
        // A zero-byte cache costs nothing: it is taken whenever the request itself fits.
        assert_eq!(
            PrefixKvCacheMode::select(budget(shed), shed, 0),
            Some(PrefixKvCacheMode::Cached)
        );
        // An overflowing sum cannot fit, and must not wrap into a small number that does.
        assert_eq!(
            PrefixKvCacheMode::select(budget(u64::MAX), u64::MAX, 1),
            Some(PrefixKvCacheMode::Prefill)
        );
        // Headroom and committed bytes are the budget's own arithmetic, not re-derived here.
        let constrained = MemoryBudget {
            total_bytes: 2_000,
            committed_bytes: 600,
            reclaimable_bytes: 0,
            reserved_headroom_bytes: 200,
        };
        assert_eq!(constrained.effective_bytes(), 1_200);
        assert_eq!(
            PrefixKvCacheMode::select(constrained, shed, cache),
            Some(PrefixKvCacheMode::Prefill)
        );
    }

    #[test]
    fn unset_fields_validate_vacuously_on_an_inert_surface() {
        let surface = ExecutionSurface::default();
        surface
            .validate("prov", None)
            .expect("an absent memory block must validate");
        surface
            .validate("prov", Some(&memory()))
            .expect("an unset execution selection must validate");
        // A ladder-only selection must not be caught by the execution gate.
        let ladder = GenerationMemory {
            stage_residency: true,
            chunk_attention: true,
            attention_chunk_size: Some(128),
            ..memory()
        };
        surface
            .validate("prov", Some(&ladder))
            .expect("the execution gate must ignore ladder parameters");
    }

    #[test]
    fn a_supporting_surface_admits_declared_values_and_refuses_the_rest() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(vec![1, 2]),
            ffn_chunk_rows: ExecutionValueDomain::ANY_POSITIVE,
            cfg_batching: CfgBatchingDomain::Modes(vec![CfgBatching::Batched]),
            prefix_kv_cache: PrefixKvCacheDomain::Modes(vec![PrefixKvCacheMode::Prefill]),
        };
        assert!(!surface.is_inert());
        assert!(surface.declaration_errors().is_empty());

        let admitted = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::new(2).unwrap()),
            ffn_chunk: Some(FfnChunk::new(4096).unwrap()),
            cfg_batching: Some(CfgBatching::Batched),
            ..memory()
        };
        surface
            .validate("prov", Some(&admitted))
            .expect("every declared value must be admitted");

        let off_grid = GenerationMemory {
            graph_eval_cadence: Some(GraphEvalCadence::new(3).unwrap()),
            ..memory()
        };
        let text = surface
            .validate("prov", Some(&off_grid))
            .expect_err("an out-of-domain cadence must be refused")
            .to_string();
        assert!(text.contains("[1, 2]"), "domain must be named: {text}");

        let missing_mode = GenerationMemory {
            cfg_batching: Some(CfgBatching::Sequential),
            ..memory()
        };
        let text = surface
            .validate("prov", Some(&missing_mode))
            .expect_err("an unimplemented CFG mode must be refused")
            .to_string();
        assert!(text.contains("sequential"), "{text}");
        assert!(text.contains("[batched]"), "domain must be named: {text}");
    }

    #[test]
    fn empty_and_malformed_declarations_are_declaration_errors() {
        let surface = ExecutionSurface {
            graph_eval_cadence_blocks: ExecutionValueDomain::Candidates(Vec::new()),
            ffn_chunk_rows: ExecutionValueDomain::Candidates(vec![0, 8, 8]),
            cfg_batching: CfgBatchingDomain::Modes(Vec::new()),
            prefix_kv_cache: PrefixKvCacheDomain::Modes(Vec::new()),
        };
        let errors = surface.declaration_errors();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("graph_eval_cadence_blocks")
                    && error.contains("empty candidate list")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("ffn_chunk_rows") && error.contains("zero candidate")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("ffn_chunk_rows") && error.contains("duplicate")),
            "{errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|error| error.contains("cfg_batching") && error.contains("empty mode list")),
            "{errors:?}"
        );
    }
}

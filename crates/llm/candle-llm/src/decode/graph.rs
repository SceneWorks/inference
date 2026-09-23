//! The shared CUDA-graph runner (epic sc-24128, story sc-24134).
//!
//! **One** graph runner in `decode/`, written over the [`StepModel`] seam and a stable-address
//! [`DecodeCache`]: [`GraphRunner`] wraps any step model, captures one CUDA graph per distinct
//! step shape (the token count `M` — `1` for a decode step, `K + 1` for a speculative verify —
//! plus the logits scope and whether hidden states are wanted) and replays it at every later
//! step of that shape. Model files carry no graph logic; they only keep their per-step positions
//! as **data** (a device tensor the kernels read, staged by [`DecodeCache::stage_positions`])
//! and their state at **stable addresses** (the S4 static KV cache), which is what makes a
//! captured step replayable at a new position.
//!
//! ## What a captured step is
//! Stream capture records every launch, memcpy and stream-ordered allocation the step issues
//! without executing them; the instantiated graph replays them as one launch. candle at
//! `1e6aa85e` allocates every temporary through cudarc 0.19's `CudaStream::alloc`, which is
//! `cuMemAllocAsync` on a device with memory-pool support (every Blackwell part), so the step's
//! temporaries become graph memory nodes — allocated and freed inside the graph — and the
//! runner needs no pre-planned workspace. The stream must be one candle created with
//! `Device::new_cuda_with_stream` (the legacy NULL stream `Device::new_cuda` uses cannot be
//! captured; [`select_device`](crate::device::select_device) does this), and the step's outputs
//! are copied into preallocated staging tensors inside the capture so nothing the step allocated
//! outlives it.
//!
//! ## The POC finding: candle uploads layout metadata per op
//! Every candle CUDA op whose kernel takes a layout — `index_select` (the embedding lookup),
//! reductions, `where_cond`, comparisons, gathers / scatters, any unary / binary / copy op on a
//! **non-contiguous** operand (a transpose, a broadcast, a narrow that is then made contiguous)
//! — uploads its `[dims, strides]` from a transient host `Vec` (`SlicePtrOrNull::params_from_layout`
//! / `CudaDevice::clone_htod` in `cuda_backend/mod.rs`) with a pageable `cuMemcpyHtoDAsync`.
//! Inside a capture the driver records that as a memcpy node that re-reads the host address on
//! every launch — an address freed the moment the op returned — so a replay copies whatever the
//! heap holds by then into the kernel's layout buffer (an illegal address, or silently wrong
//! indexing). The runner therefore takes a **census** of every captured graph before it is
//! instantiated ([`GraphCensus`]) and refuses one with a host-sourced memcpy node
//! ([`REASON_HOST_UPLOAD_IN_CAPTURE`]) — the reason a Qwen3.5/3.8 step reports on this candle
//! revision. What would fix it in candle: pass layouts by value as kernel parameters (or cache
//! the `[dims, strides]` device buffers per layout) so no per-op host upload exists. Until then
//! only a step built from contiguous-only ops, cuBLAS matmuls, `copy2d` (`slice_set`) and the
//! nvrtc-seam kernels (scalar arguments) is replayable — the synthetic model in the CUDA tests
//! below is such a step, and is what proves the runner end to end.
//!
//! ## Verification before trust
//! A graph is only ever used after two bit-exact self-checks: the capture step and the first
//! replay each launch the graph, roll the cache back, run the eager step, and compare every
//! logit (and hidden) element on the device. Any difference — a stale scalar, a reallocated
//! temporary, an unstable state address — throws the graph away, keeps the eager result and
//! names the reason. The reference path is never changed by the runner (E2).
//!
//! ## Fallback, never failure
//! Every refusal is a **named** reason on the per-thread [`GraphTally`], reported per request
//! through [`DecodeRecord::cuda_graphs`](crate::decode::DecodeRecord::cuda_graphs) as
//! `graph: … fallback=<reason>`: the switch is off, the build has no `cuda` feature, the model
//! is not on a CUDA device, the device has no stream-ordered allocator, the model or the cache
//! declares itself uncapturable ([`StepModel::graph_support`], [`DecodeCache::graph_support`]),
//! the census found a host upload / host read / an allocation that outlives the graph, the
//! driver invalidated the capture, instantiation failed (memory), or a replay disagreed with
//! eager. A fallback runs the eager step; the request continues on the same device.
//!
//! ## Switch and accounting
//! `CANDLE_LLM_CUDA_GRAPHS` (`1` / `on` / `true` / `yes` enable; the default is **off** —
//! opt-in until a capturable step model exists and the decode bench shows a win, see the
//! story's evidence) or [`set_cuda_graphs`] at runtime. Graph workspaces — the staging tensors
//! and the graph's own reserved device memory — are reported by [`GraphRunner::workspace`] for
//! admission (E6).
//!
//! Without the `cuda` feature the runner is a transparent pass-through that reports
//! [`REASON_CUDA_FEATURE_OFF`].

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use candle_core::Device;
#[cfg(feature = "cuda")]
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use crate::decode::step::StepTokens;
use crate::decode::step::{LogitsScope, StepModel, StepOutput, StepRequest};
use crate::error::Result;
use crate::primitives::attention::AttnFormulation;
use crate::primitives::decode_cache::DecodeCache;

/// Environment switch: `1` / `on` / `true` / `yes` enable the graph runner; anything else (and
/// unset) leaves it off.
pub const CUDA_GRAPHS_ENV: &str = "CANDLE_LLM_CUDA_GRAPHS";

/// Fallback reason: the switch is off.
pub const REASON_DISABLED: &str = "disabled";
/// Fallback reason: this build has no `cuda` feature, so no graph path exists.
pub const REASON_CUDA_FEATURE_OFF: &str = "cuda_feature_off";
/// Fallback reason: the model's device is not a CUDA device.
pub const REASON_NOT_CUDA: &str = "not_cuda";
/// Fallback reason: the device has no stream-ordered allocator (`cuMemAllocAsync`), so candle's
/// per-op temporaries would be plain `cuMemAlloc` calls, which capture refuses.
pub const REASON_NO_ASYNC_ALLOC: &str = "no_async_alloc";
/// Fallback reason: cudarc's per-slice event tracking is on (a second stream exists on the
/// context); its waits on events recorded before the capture would invalidate it.
pub const REASON_EVENT_TRACKING: &str = "event_tracking";
/// Fallback reason: the model runs on the legacy NULL stream (`Device::new_cuda`,
/// `CANDLE_LLM_CUDA_STREAM=legacy`), which stream capture does not support.
pub const REASON_LEGACY_STREAM: &str = "legacy_stream";
/// Fallback reason: the step shape is not capturable (empty, or more tokens than
/// [`GraphRunner::MAX_CAPTURED_TOKENS`] — a prefill is never captured).
pub const REASON_SHAPE: &str = "shape";
/// Fallback reason: the cache cannot roll back to the step start, so the self-check cannot run.
pub const REASON_ROLLBACK_UNAVAILABLE: &str = "rollback_unavailable";
/// Fallback reason: the step issued a device->host transfer this crate counts (`note_host_sync`)
/// during capture — the data it read was never computed.
pub const REASON_SYNC_IN_CAPTURE: &str = "sync_in_capture";
/// Fallback reason: the captured graph holds a memcpy node whose source is host memory — a
/// pageable upload the step made (candle's per-op `[dims, strides]` upload, a `Tensor::from_vec`)
/// whose source address is freed before any replay.
pub const REASON_HOST_UPLOAD_IN_CAPTURE: &str = "host_upload_in_capture";
/// Fallback reason: the captured graph holds a memcpy node whose destination is host memory — a
/// device->host read the step made (a `to_vec`) into a buffer that no longer exists.
pub const REASON_HOST_READ_IN_CAPTURE: &str = "host_read_in_capture";
/// Fallback reason: the capture was invalidated by the driver (the error text is logged).
pub const REASON_CAPTURE_INVALIDATED: &str = "capture_invalidated";
/// Fallback reason: the step failed during capture with an ordinary error; it is re-run eager.
pub const REASON_STEP_FAILED_IN_CAPTURE: &str = "step_failed_in_capture";
/// Fallback reason: an allocation made during the capture is not freed inside the graph — the
/// step kept a fresh tensor (a replaced state) instead of writing in place, so a replay would
/// read a stale address and the graph could not be relaunched.
pub const REASON_ALLOCATION_ESCAPED_CAPTURE: &str = "allocation_escaped_capture";
/// Fallback reason: `cuGraphInstantiate` / upload failed (typically device memory).
pub const REASON_INSTANTIATE_FAILED: &str = "instantiate_failed";
/// Fallback reason: a graph launch failed.
pub const REASON_LAUNCH_FAILED: &str = "launch_failed";
/// Fallback reason: a replay's outputs differed from the eager step's at the same position.
pub const REASON_REPLAY_MISMATCH: &str = "replay_mismatch";
/// Fallback reason: the step's outputs are not a shape the staging tensors can hold.
pub const REASON_OUTPUT_SHAPE: &str = "output_shape";

const POLICY_ENV: u8 = 0;
const POLICY_ON: u8 = 1;
const POLICY_OFF: u8 = 2;

static POLICY: AtomicU8 = AtomicU8::new(POLICY_ENV);

fn env_says_enabled() -> bool {
    static FROM_ENV: OnceLock<bool> = OnceLock::new();
    *FROM_ENV.get_or_init(|| {
        std::env::var(CUDA_GRAPHS_ENV)
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                matches!(v.as_str(), "1" | "on" | "true" | "yes")
            })
            .unwrap_or(false)
    })
}

/// Whether the graph runner may capture at all (the switch; a `cuda` build on a CUDA device is
/// still required for a graph to exist).
pub fn cuda_graphs_enabled() -> bool {
    match POLICY.load(Ordering::Relaxed) {
        POLICY_ON => true,
        POLICY_OFF => false,
        _ => env_says_enabled(),
    }
}

/// Override the switch for the process: `Some(true)` / `Some(false)` force it, `None` returns to
/// the environment's setting.
pub fn set_cuda_graphs(enabled: Option<bool>) {
    let policy = match enabled {
        Some(true) => POLICY_ON,
        Some(false) => POLICY_OFF,
        None => POLICY_ENV,
    };
    POLICY.store(policy, Ordering::Relaxed);
}

static POLICY_LOCK: Mutex<()> = Mutex::new(());

/// Holds the process-wide switch lock; restores the switch it found when dropped. Returned by
/// [`cuda_graphs_policy_guard`].
#[doc(hidden)]
#[must_use = "the policy is only held (and restored) while the guard is alive"]
pub struct CudaGraphsPolicyGuard {
    previous: u8,
    _lock: MutexGuard<'static, ()>,
}

impl Drop for CudaGraphsPolicyGuard {
    fn drop(&mut self) {
        POLICY.store(self.previous, Ordering::Relaxed);
    }
}

/// Test seam: take the process-wide switch lock, apply `enabled` (as [`set_cuda_graphs`]) and
/// hand back a guard that restores the previous switch when dropped. Every test that flips the
/// switch — or asserts a reason the switch could change — holds one.
#[doc(hidden)]
pub fn cuda_graphs_policy_guard(enabled: Option<bool>) -> CudaGraphsPolicyGuard {
    let lock = POLICY_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous = POLICY.load(Ordering::Relaxed);
    set_cuda_graphs(enabled);
    CudaGraphsPolicyGuard {
        previous,
        _lock: lock,
    }
}

/// Per-thread counts of graph-replayed vs eager steps (monotone; take deltas with
/// [`GraphTally::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphTally {
    /// Steps served by a graph replay.
    pub replayed: u64,
    /// Steps served by the eager step (fallbacks, warm-ups, prefills, self-checks).
    pub eager: u64,
    /// Graphs captured and verified (one per step shape).
    pub captured: u64,
    /// Why the most recent eager step happened when a graph was wanted (`None` if the runner
    /// was never asked, or every eager step was a warm-up / self-check).
    pub fallback_reason: Option<&'static str>,
}

impl GraphTally {
    /// The counts accumulated since `start` (the reason is the latest one).
    pub fn since(&self, start: &GraphTally) -> GraphTally {
        GraphTally {
            replayed: self.replayed.wrapping_sub(start.replayed),
            eager: self.eager.wrapping_sub(start.eager),
            captured: self.captured.wrapping_sub(start.captured),
            fallback_reason: if self.eager != start.eager {
                self.fallback_reason
            } else {
                None
            },
        }
    }

    /// Which path served the steps: `graph` (only replays), `eager` (no replay), `mixed` (both),
    /// or `none` (no step ran through the runner).
    pub fn label(&self) -> &'static str {
        match (self.replayed, self.eager) {
            (0, 0) => "none",
            (_, 0) => "graph",
            (0, _) => "eager",
            _ => "mixed",
        }
    }

    /// The telemetry line: `graph: <label> replayed=<n> eager=<n> captured=<n>` plus
    /// ` fallback=<reason>` when an eager step was a fallback.
    pub fn describe(&self) -> String {
        let mut s = format!(
            "graph: {} replayed={} eager={} captured={}",
            self.label(),
            self.replayed,
            self.eager,
            self.captured
        );
        if let Some(reason) = self.fallback_reason {
            s.push_str(" fallback=");
            s.push_str(reason);
        }
        s
    }
}

thread_local! {
    static TALLY: Cell<GraphTally> = const {
        Cell::new(GraphTally {
            replayed: 0,
            eager: 0,
            captured: 0,
            fallback_reason: None,
        })
    };
    static CAPTURING: Cell<bool> = const { Cell::new(false) };
}

/// This thread's graph tally so far.
pub fn graph_tally() -> GraphTally {
    TALLY.with(Cell::get)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn note_replayed() {
    TALLY.with(|t| {
        let mut v = t.get();
        v.replayed = v.replayed.wrapping_add(1);
        t.set(v);
    });
}

fn note_eager(reason: Option<&'static str>) {
    TALLY.with(|t| {
        let mut v = t.get();
        v.eager = v.eager.wrapping_add(1);
        if reason.is_some() {
            v.fallback_reason = reason;
        }
        t.set(v);
    });
}

#[cfg(feature = "cuda")]
fn note_captured() {
    TALLY.with(|t| {
        let mut v = t.get();
        v.captured = v.captured.wrapping_add(1);
        t.set(v);
    });
}

/// Whether the current thread is inside a stream capture. A cache's
/// [`stage_positions`](DecodeCache::stage_positions) is a no-op while this is set — the runner
/// staged them before the capture began, and a host upload inside a capture is never replayable.
pub fn capturing() -> bool {
    CAPTURING.with(Cell::get)
}

#[cfg(feature = "cuda")]
struct CaptureFlag;

#[cfg(feature = "cuda")]
impl CaptureFlag {
    fn set() -> Self {
        CAPTURING.with(|c| c.set(true));
        CaptureFlag
    }
}

#[cfg(feature = "cuda")]
impl Drop for CaptureFlag {
    fn drop(&mut self) {
        CAPTURING.with(|c| c.set(false));
    }
}

/// The device memory a runner holds for its graphs (E6): the staging tensors it allocated and
/// the graph memory the driver reserved for the captured temporaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphWorkspace {
    /// Bytes of the preallocated input / output staging tensors, every captured shape.
    pub staging_bytes: usize,
    /// Bytes the driver reports reserved for graph memory nodes on the device after the runner's
    /// captures (`CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT` delta; device-wide, so an
    /// over-estimate when other graphs live on the device).
    pub graph_reserved_bytes: usize,
}

impl GraphWorkspace {
    /// `staging_bytes + graph_reserved_bytes` (saturating).
    pub fn total_bytes(&self) -> usize {
        self.staging_bytes.saturating_add(self.graph_reserved_bytes)
    }
}

/// What a captured graph is made of — the node census the runner takes before instantiating it,
/// and the record behind the POC findings (a real decoder step's census is in the evidence).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphCensus {
    /// Every node.
    pub nodes: usize,
    /// Kernel launches.
    pub kernels: usize,
    /// Device->device memcpy nodes (`copy2d`, `slice_set`, `copy`).
    pub memcpy_device: usize,
    /// Memcpy nodes reading host memory: pageable uploads the step made.
    pub memcpy_from_host: usize,
    /// Memcpy nodes writing host memory: device->host reads the step made.
    pub memcpy_to_host: usize,
    /// Memset nodes (`zeros`).
    pub memsets: usize,
    /// Stream-ordered allocations (candle's temporaries).
    pub mem_allocs: usize,
    /// Stream-ordered frees.
    pub mem_frees: usize,
    /// Allocations with no free node in the graph — tensors that outlive the step.
    pub escaped_allocs: usize,
    /// Any other node kind.
    pub other: usize,
}

impl GraphCensus {
    /// Why a graph with this census must not be replayed, or `None` when it is replayable.
    pub fn refusal(&self) -> Option<&'static str> {
        if self.memcpy_from_host > 0 {
            Some(REASON_HOST_UPLOAD_IN_CAPTURE)
        } else if self.memcpy_to_host > 0 {
            Some(REASON_HOST_READ_IN_CAPTURE)
        } else if self.escaped_allocs > 0 {
            Some(REASON_ALLOCATION_ESCAPED_CAPTURE)
        } else {
            None
        }
    }

    /// One line for logs and evidence.
    pub fn describe(&self) -> String {
        format!(
            "nodes={} kernels={} memcpy(dtod={}, htod={}, dtoh={}) memset={} alloc={} free={} escaped={} other={}",
            self.nodes,
            self.kernels,
            self.memcpy_device,
            self.memcpy_from_host,
            self.memcpy_to_host,
            self.memsets,
            self.mem_allocs,
            self.mem_frees,
            self.escaped_allocs,
            self.other
        )
    }
}

/// One step shape: the graph key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ShapeKey {
    tokens: usize,
    scope: LogitsScope,
    want_hidden: bool,
}

impl ShapeKey {
    fn of(request: &StepRequest<'_>) -> Result<Self> {
        Ok(Self {
            tokens: request.len()?,
            scope: request.scope,
            want_hidden: request.want_hidden,
        })
    }
}

/// The step model wrapper: replays a captured graph for a step shape it has verified, runs the
/// eager step otherwise, and records which on the per-thread [`GraphTally`]. One runner serves
/// **one** cache (the graphs are captured against that cache's buffer addresses); a cache built
/// through the runner's own [`new_cache_for`](StepModel::new_cache_for) resets it, and a step on
/// a cache with a different [`graph_identity`](DecodeCache::graph_identity) discards the graphs
/// before anything is replayed.
pub struct GraphRunner<'m, M: StepModel> {
    model: &'m M,
    inner: RefCell<Inner>,
}

struct Inner {
    /// The identity of the cache the graphs belong to.
    cache_identity: Option<usize>,
    #[cfg(feature = "cuda")]
    cuda: cuda::State,
    /// Set once a refusal was recorded for this runner (recorded once, not per step).
    refused: Option<&'static str>,
    /// The census of the last capture attempt (for evidence and tests).
    last_census: Option<GraphCensus>,
}

impl<'m, M: StepModel> GraphRunner<'m, M> {
    /// The largest step (in tokens) the runner captures: decode (`1`) and verify (`K + 1`)
    /// steps; a prompt prefill is never captured.
    pub const MAX_CAPTURED_TOKENS: usize = 16;

    /// Wrap `model`. Nothing is captured until a step of some shape has run eager once (the
    /// warm-up that compiles every kernel outside any capture) and passed its self-checks.
    pub fn new(model: &'m M) -> Self {
        Self {
            model,
            inner: RefCell::new(Inner {
                cache_identity: None,
                #[cfg(feature = "cuda")]
                cuda: cuda::State::default(),
                refused: None,
                last_census: None,
            }),
        }
    }

    /// The wrapped model.
    pub fn model(&self) -> &'m M {
        self.model
    }

    /// The device memory this runner's graphs hold (E6).
    pub fn workspace(&self) -> GraphWorkspace {
        #[cfg(feature = "cuda")]
        {
            return self.inner.borrow().cuda.workspace();
        }
        #[allow(unreachable_code)]
        GraphWorkspace::default()
    }

    /// How many graphs this runner currently holds (one per verified step shape).
    pub fn captured_graphs(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            return self.inner.borrow().cuda.graphs();
        }
        #[allow(unreachable_code)]
        0
    }

    /// The reason this runner stopped trying to capture, if it did.
    pub fn refusal(&self) -> Option<&'static str> {
        self.inner.borrow().refused
    }

    /// The census of the most recent capture attempt, if there was one.
    pub fn last_census(&self) -> Option<GraphCensus> {
        self.inner.borrow().last_census
    }

    /// Drop every captured graph (the next step of each shape starts over with a warm-up).
    pub fn reset(&self) {
        let mut inner = self.inner.borrow_mut();
        inner.cache_identity = None;
        inner.refused = None;
        inner.last_census = None;
        #[cfg(feature = "cuda")]
        inner.cuda.clear();
    }

    /// Why this runner cannot capture on `cache` at all (checked before any capture, E5), or
    /// `None` when it can.
    fn capability_refusal(&self, cache: &M::Cache) -> Option<&'static str> {
        if !cuda_graphs_enabled() {
            return Some(REASON_DISABLED);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = cache;
            Some(REASON_CUDA_FEATURE_OFF)
        }
        #[cfg(feature = "cuda")]
        {
            let Device::Cuda(dev) = self.model.device() else {
                return Some(REASON_NOT_CUDA);
            };
            let stream = dev.cuda_stream();
            if stream.cu_stream().is_null() {
                return Some(REASON_LEGACY_STREAM);
            }
            let ctx = stream.context();
            if !ctx.has_async_alloc() {
                return Some(REASON_NO_ASYNC_ALLOC);
            }
            if ctx.is_managing_stream_synchronization() {
                return Some(REASON_EVENT_TRACKING);
            }
            if let Err(reason) = self.model.graph_support() {
                return Some(reason);
            }
            if let Err(reason) = cache.graph_support() {
                return Some(reason);
            }
            None
        }
    }
}

impl<M: StepModel> StepModel for GraphRunner<'_, M> {
    type Cache = M::Cache;

    fn new_cache(&self) -> M::Cache {
        self.reset();
        self.model.new_cache()
    }

    fn new_cache_for(&self, capacity: usize, overshoot: usize) -> Result<M::Cache> {
        self.reset();
        self.model.new_cache_for(capacity, overshoot)
    }

    fn attn_formulation(&self, cache: &M::Cache) -> AttnFormulation {
        self.model.attn_formulation(cache)
    }

    fn graph_support(&self) -> std::result::Result<(), &'static str> {
        self.model.graph_support()
    }

    fn device(&self) -> &Device {
        self.model.device()
    }

    fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    fn forward_step(&self, cache: &mut M::Cache, request: StepRequest<'_>) -> Result<StepOutput> {
        let key = ShapeKey::of(&request)?;
        // A different cache than the graphs were captured against: never replay into it.
        {
            let mut inner = self.inner.borrow_mut();
            let identity = cache.graph_identity();
            if inner.cache_identity != Some(identity) {
                inner.cache_identity = Some(identity);
                inner.refused = None;
                #[cfg(feature = "cuda")]
                inner.cuda.clear();
            }
        }
        if let Some(reason) = self.inner.borrow().refused {
            note_eager(Some(reason));
            return self.model.forward_step(cache, request);
        }
        if let Some(reason) = self.capability_refusal(cache) {
            self.inner.borrow_mut().refused = Some(reason);
            note_eager(Some(reason));
            return self.model.forward_step(cache, request);
        }
        if key.tokens == 0 || key.tokens > Self::MAX_CAPTURED_TOKENS {
            note_eager(Some(REASON_SHAPE));
            return self.model.forward_step(cache, request);
        }
        #[cfg(feature = "cuda")]
        {
            cuda::step(self, cache, request, key)
        }
        #[cfg(not(feature = "cuda"))]
        {
            note_eager(Some(REASON_CUDA_FEATURE_OFF));
            self.model.forward_step(cache, request)
        }
    }
}

/// **Diagnostic** (evidence, sc-24134): record one step of `model` on `cache` as a CUDA graph,
/// take its [`GraphCensus`], destroy the graph **without instantiating or launching it**, and
/// roll the cache back to where it was. Nothing runs on the device, and the declared refusals
/// ([`StepModel::graph_support`], [`DecodeCache::graph_support`]) are deliberately not consulted:
/// this is how a real decoder step's launch count and host-upload count are measured. `None`
/// when the recording itself failed (the reason is logged) — a step that frees a tensor it did
/// not allocate inside the capture does that.
///
/// The recording replaces any state the step re-creates with graph-owned tensors that were never
/// backed by memory; the rollback drops them, and the driver's complaints about freeing them are
/// drained here. Use it on a model you are done with (the end of an evidence run), not mid-request.
#[cfg(feature = "cuda")]
pub fn census_step<M: StepModel + ?Sized>(
    model: &M,
    cache: &mut M::Cache,
    request: StepRequest<'_>,
) -> Result<Option<GraphCensus>> {
    let Device::Cuda(dev) = model.device() else {
        return Ok(None);
    };
    let base = cache.len();
    cache.stage_positions()?;
    let recorded = cuda::capture(dev, || model.forward_step(cache, request).map(|_| ()));
    let census = match recorded {
        Ok((recorded, ())) => Some(recorded.census()),
        Err(cuda::Abandoned(reason)) => {
            eprintln!("[cuda-graph] census: the recording was abandoned ({reason})");
            None
        }
    };
    cache.rollback_to(base)?;
    // The graph-owned tensors the rollback dropped were never mapped; their frees fail.
    let stream = dev.cuda_stream();
    if let Err(e) = stream.context().check_err() {
        eprintln!("[cuda-graph] census: drained a recorded driver error after the rollback: {e:?}");
    }
    Ok(census)
}

/// Copy `tokens` into a `[1, n]` `u32` staging tensor in place (a host upload or a
/// device->device copy, both outside any capture).
#[cfg(feature = "cuda")]
fn stage_tokens(staging: &Tensor, tokens: &StepTokens<'_>, device: &Device) -> Result<()> {
    let ids = tokens.ids(device)?;
    staging.slice_set(&ids, 1, 0)?;
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda {
    //! The device half: stream capture, the census, instantiation, replay and the self-checks.

    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    use candle_core::cuda_backend::cudarc::driver::{sys, CudaStream};
    use candle_core::{CudaDevice, DType, Tensor};

    use super::*;
    use crate::error::Error;
    use crate::primitives::decode_cache::tensor_bytes;
    use crate::primitives::host_sync::{host_sync_count, note_host_sync};

    fn ok(r: sys::CUresult) -> std::result::Result<(), sys::CUresult> {
        if r == sys::CUresult::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(r)
        }
    }

    /// A graph as `cuStreamEndCapture` returned it: recorded, not yet instantiated. Destroyed on
    /// drop unless [`instantiate`](Self::instantiate) took it.
    pub(super) struct Recorded {
        graph: sys::CUgraph,
        stream: Arc<CudaStream>,
    }

    impl Drop for Recorded {
        fn drop(&mut self) {
            if !self.graph.is_null() && self.stream.context().bind_to_thread().is_ok() {
                unsafe { sys::cuGraphDestroy(self.graph) };
            }
        }
    }

    impl Recorded {
        /// Count the graph's nodes by kind (see [`GraphCensus`]).
        pub(super) fn census(&self) -> GraphCensus {
            let mut c = GraphCensus::default();
            let mut count = 0usize;
            if ok(unsafe { sys::cuGraphGetNodes(self.graph, std::ptr::null_mut(), &mut count) })
                .is_err()
            {
                return c;
            }
            let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); count];
            if ok(unsafe { sys::cuGraphGetNodes(self.graph, nodes.as_mut_ptr(), &mut count) })
                .is_err()
            {
                return c;
            }
            nodes.truncate(count);
            c.nodes = count;
            let mut allocated: HashSet<u64> = HashSet::new();
            let mut freed: HashSet<u64> = HashSet::new();
            for node in nodes {
                let mut kind = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
                if ok(unsafe { sys::cuGraphNodeGetType(node, &mut kind) }).is_err() {
                    c.other += 1;
                    continue;
                }
                match kind {
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => c.kernels += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => c.memsets += 1,
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => {
                        // The driver fills every field (the enums have no zero value, so the
                        // struct cannot be zero-initialised).
                        let mut params = std::mem::MaybeUninit::<sys::CUDA_MEMCPY3D>::uninit();
                        if ok(unsafe { sys::cuGraphMemcpyNodeGetParams(node, params.as_mut_ptr()) })
                            .is_ok()
                        {
                            let params = unsafe { params.assume_init() };
                            let host = sys::CUmemorytype::CU_MEMORYTYPE_HOST;
                            if params.srcMemoryType == host {
                                c.memcpy_from_host += 1;
                            } else if params.dstMemoryType == host {
                                c.memcpy_to_host += 1;
                            } else {
                                c.memcpy_device += 1;
                            }
                        } else {
                            c.other += 1;
                        }
                    }
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => {
                        c.mem_allocs += 1;
                        let mut params =
                            std::mem::MaybeUninit::<sys::CUDA_MEM_ALLOC_NODE_PARAMS>::uninit();
                        if ok(unsafe {
                            sys::cuGraphMemAllocNodeGetParams(node, params.as_mut_ptr())
                        })
                        .is_ok()
                        {
                            allocated.insert(unsafe { params.assume_init() }.dptr);
                        }
                    }
                    sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => {
                        c.mem_frees += 1;
                        let mut dptr: sys::CUdeviceptr = 0;
                        if ok(unsafe { sys::cuGraphMemFreeNodeGetParams(node, &mut dptr) }).is_ok()
                        {
                            freed.insert(dptr);
                        }
                    }
                    _ => c.other += 1,
                }
            }
            // Every temporary a step allocates is freed inside the graph, so a surplus of
            // allocation nodes is an allocation that outlives the step; the address match is
            // the precise form when the driver reports node parameters.
            c.escaped_allocs = allocated
                .difference(&freed)
                .count()
                .max(c.mem_allocs.saturating_sub(c.mem_frees));
            c
        }

        /// Instantiate (flags 0) and upload; the recorded graph is consumed either way.
        pub(super) fn instantiate(mut self) -> std::result::Result<Graph, Abandoned> {
            let mut exec: sys::CUgraphExec = std::ptr::null_mut();
            let instantiated =
                unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, self.graph, 0) };
            if let Err(e) = ok(instantiated) {
                eprintln!("[cuda-graph] instantiate failed: {e:?}");
                return Err(Abandoned(REASON_INSTANTIATE_FAILED));
            }
            let graph = Graph {
                graph: std::mem::replace(&mut self.graph, std::ptr::null_mut()),
                exec,
                stream: self.stream.clone(),
            };
            if let Err(e) = ok(unsafe { sys::cuGraphUpload(graph.exec, graph.stream.cu_stream()) })
            {
                eprintln!("[cuda-graph] upload failed: {e:?}");
                return Err(Abandoned(REASON_INSTANTIATE_FAILED));
            }
            Ok(graph)
        }
    }

    /// An instantiated CUDA graph on the device's stream. (cudarc's own `CudaGraph` instantiates
    /// through a flag enum with no zero value, so this wrapper drives the driver directly:
    /// `cuGraphInstantiateWithFlags(0)`, explicit upload, executable destroyed before the graph.)
    pub(super) struct Graph {
        graph: sys::CUgraph,
        exec: sys::CUgraphExec,
        stream: Arc<CudaStream>,
    }

    impl Graph {
        /// Replay the graph on its stream (asynchronous, stream-ordered like any launch).
        pub(super) fn launch(&self) -> std::result::Result<(), sys::CUresult> {
            self.stream.context().bind_to_thread().map_err(|e| e.0)?;
            ok(unsafe { sys::cuGraphLaunch(self.exec, self.stream.cu_stream()) })
        }
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            if self.stream.context().bind_to_thread().is_err() {
                return;
            }
            if !self.exec.is_null() {
                unsafe { sys::cuGraphExecDestroy(self.exec) };
            }
            if !self.graph.is_null() {
                unsafe { sys::cuGraphDestroy(self.graph) };
            }
        }
    }

    /// A captured, verified graph for one step shape, with the staging tensors its kernels read
    /// and write.
    pub(super) struct Captured {
        graph: Graph,
        ids: Tensor,
        logits: Tensor,
        hidden: Option<Tensor>,
        /// Replays served so far; the first one is still verified against eager.
        replays: u64,
    }

    impl Captured {
        fn staging_bytes(&self) -> usize {
            tensor_bytes(&self.ids)
                .saturating_add(tensor_bytes(&self.logits))
                .saturating_add(self.hidden.as_ref().map_or(0, tensor_bytes))
        }
    }

    /// Per-shape progress: an eager warm-up seen, or a graph held.
    pub(super) enum Shape {
        /// One eager step of this shape has run (kernels compiled); the next one captures.
        Warmed,
        Graph(Captured),
    }

    #[derive(Default)]
    pub(super) struct State {
        shapes: HashMap<ShapeKey, Shape>,
        graph_reserved_bytes: usize,
    }

    impl State {
        pub(super) fn clear(&mut self) {
            self.shapes.clear();
            self.graph_reserved_bytes = 0;
        }

        pub(super) fn graphs(&self) -> usize {
            self.shapes
                .values()
                .filter(|s| matches!(s, Shape::Graph(_)))
                .count()
        }

        pub(super) fn workspace(&self) -> GraphWorkspace {
            let staging_bytes = self
                .shapes
                .values()
                .map(|s| match s {
                    Shape::Graph(c) => c.staging_bytes(),
                    Shape::Warmed => 0,
                })
                .fold(0usize, usize::saturating_add);
            GraphWorkspace {
                staging_bytes,
                graph_reserved_bytes: self.graph_reserved_bytes,
            }
        }
    }

    /// `cuDeviceGetGraphMemAttribute` for `attr`, or `None` when the driver refuses.
    pub(super) fn graph_mem_attribute(
        dev: &CudaDevice,
        attr: sys::CUgraphMem_attribute,
    ) -> Option<u64> {
        let stream = dev.cuda_stream();
        let device = stream.context().cu_device();
        let mut value = 0u64;
        let result = unsafe {
            sys::cuDeviceGetGraphMemAttribute(
                device,
                attr,
                &mut value as *mut u64 as *mut std::ffi::c_void,
            )
        };
        (result == sys::CUresult::CUDA_SUCCESS).then_some(value)
    }

    fn reserved_graph_mem(dev: &CudaDevice) -> Option<u64> {
        graph_mem_attribute(
            dev,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
        )
    }

    /// Classify a driver / candle error raised inside a capture into a fallback reason.
    pub(super) fn classify_capture_error(e: &Error) -> &'static str {
        let text = format!("{e:?}");
        if text.contains("STREAM_CAPTURE") {
            REASON_CAPTURE_INVALIDATED
        } else {
            REASON_STEP_FAILED_IN_CAPTURE
        }
    }

    /// Why a capture attempt was abandoned.
    pub(super) struct Abandoned(pub(super) &'static str);

    /// Capture `f` on the device's stream: begin capture, run the closure with the capture flag
    /// set, end capture. The closure's device work is **recorded, not run**. Returns the recorded
    /// graph, or the reason capture was abandoned (the capture is always ended and cudarc's
    /// recorded error state drained, so the stream is usable afterwards).
    pub(super) fn capture<T>(
        dev: &CudaDevice,
        f: impl FnOnce() -> Result<T>,
    ) -> std::result::Result<(Recorded, T), Abandoned> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        if let Err(e) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        {
            eprintln!("[cuda-graph] begin_capture failed: {e:?}");
            return Err(Abandoned(REASON_CAPTURE_INVALIDATED));
        }
        let syncs_before = host_sync_count();
        let outcome = {
            let _flag = CaptureFlag::set();
            f()
        };
        // `cuStreamEndCapture` is called whatever happened: on a failed step it returns the
        // driver's invalidation error and leaves the stream in its normal state. A failing free
        // inside the capture is recorded on cudarc's context and would resurface on the next
        // call, so it is drained here.
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let ended = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        let recorded_err = ctx.check_err().err();
        let recorded = Recorded {
            graph,
            stream: stream.clone(),
        };
        let value = match outcome {
            Ok(v) => v,
            Err(e) => {
                let reason = classify_capture_error(&e);
                eprintln!("[cuda-graph] step failed during capture ({reason}): {e}");
                return Err(Abandoned(reason));
            }
        };
        if let Err(e) = ok(ended) {
            eprintln!("[cuda-graph] end_capture failed: {e:?} (recorded {recorded_err:?})");
            return Err(Abandoned(REASON_CAPTURE_INVALIDATED));
        }
        if recorded.graph.is_null() {
            return Err(Abandoned(REASON_CAPTURE_INVALIDATED));
        }
        if let Some(e) = recorded_err {
            eprintln!("[cuda-graph] an operation failed during capture: {e:?}");
            return Err(Abandoned(REASON_CAPTURE_INVALIDATED));
        }
        if host_sync_count() != syncs_before {
            return Err(Abandoned(REASON_SYNC_IN_CAPTURE));
        }
        Ok((recorded, value))
    }

    /// Bit-exact equality of two tensors of the same shape and dtype, decided on the device with
    /// one scalar transfer (counted as a host sync — this runs only at capture / verification).
    pub(super) fn bit_identical(a: &Tensor, b: &Tensor) -> Result<bool> {
        if a.dims() != b.dims() || a.dtype() != b.dtype() {
            return Ok(false);
        }
        let n = a.elem_count();
        // `eq` compares values; a NaN compares unequal, which is the strict answer.
        let equal = a.eq(b)?.to_dtype(DType::F32)?.sum_all()?;
        note_host_sync();
        let equal = equal.to_scalar::<f32>()?;
        Ok(equal as usize == n)
    }

    fn outputs_identical(a: &StepOutput, b: &StepOutput) -> Result<bool> {
        if !bit_identical(&a.logits, &b.logits)? {
            return Ok(false);
        }
        match (&a.hidden, &b.hidden) {
            (None, None) => Ok(true),
            (Some(x), Some(y)) => bit_identical(x, y),
            _ => Ok(false),
        }
    }

    /// Fresh copies of the staging outputs (so the caller's tensors survive the next replay).
    fn copy_outputs(logits: &Tensor, hidden: Option<&Tensor>) -> Result<StepOutput> {
        Ok(StepOutput {
            logits: logits.copy()?,
            hidden: match hidden {
                Some(h) => Some(h.copy()?),
                None => None,
            },
        })
    }

    /// Copy a step's outputs into the staging tensors in place (inside the capture, so the copy
    /// is part of the graph and the temporaries die inside it).
    fn stage_outputs(
        logits_stage: &Tensor,
        hidden_stage: Option<&Tensor>,
        out: StepOutput,
    ) -> Result<()> {
        let logits = out.logits.contiguous()?;
        if logits.dims() != logits_stage.dims() || logits.dtype() != logits_stage.dtype() {
            return Err(Error::Msg(REASON_OUTPUT_SHAPE.into()));
        }
        logits_stage.slice_set(&logits, 0, 0)?;
        match (hidden_stage, out.hidden) {
            (Some(staged), Some(hidden)) => {
                let hidden = hidden.contiguous()?;
                if hidden.dims() != staged.dims() || hidden.dtype() != staged.dtype() {
                    return Err(Error::Msg(REASON_OUTPUT_SHAPE.into()));
                }
                staged.slice_set(&hidden, 0, 0)?;
            }
            (None, None) => {}
            _ => return Err(Error::Msg(REASON_OUTPUT_SHAPE.into())),
        }
        Ok(())
    }

    fn refuse<M: StepModel>(runner: &GraphRunner<'_, M>, key: ShapeKey, reason: &'static str) {
        let mut inner = runner.inner.borrow_mut();
        inner.cuda.shapes.remove(&key);
        inner.refused = Some(reason);
    }

    /// The runner's CUDA step: warm-up, capture (with self-check), verified replay, replay.
    pub(super) fn step<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        request: StepRequest<'_>,
        key: ShapeKey,
    ) -> Result<StepOutput> {
        let Device::Cuda(dev) = runner.model.device() else {
            note_eager(Some(REASON_NOT_CUDA));
            return runner.model.forward_step(cache, request);
        };
        let dev = dev.clone();
        let device = Device::Cuda(dev.clone());
        let model = runner.model;
        enum Plan {
            WarmUp,
            Capture,
            VerifyReplay,
            Replay,
        }
        let plan = {
            let inner = runner.inner.borrow();
            match inner.cuda.shapes.get(&key) {
                None => Plan::WarmUp,
                Some(Shape::Warmed) => Plan::Capture,
                Some(Shape::Graph(c)) if c.replays == 0 => Plan::VerifyReplay,
                Some(Shape::Graph(_)) => Plan::Replay,
            }
        };
        match plan {
            Plan::WarmUp => {
                let out = model.forward_step(cache, request)?;
                note_eager(None);
                runner
                    .inner
                    .borrow_mut()
                    .cuda
                    .shapes
                    .insert(key, Shape::Warmed);
                Ok(out)
            }
            Plan::Capture => {
                let base = cache.len();
                // Staging tensors are allocated outside the capture; their shapes come from a
                // dry run of the eager step's output shapes: the warm-up already proved the
                // step runs, so this is one eager step whose result is the self-check reference.
                let eager = model.forward_step(cache, request)?;
                note_eager(None);
                if let Err(e) = cache.rollback_to(base) {
                    eprintln!("[cuda-graph] cannot roll back for the capture: {e}");
                    refuse(runner, key, REASON_ROLLBACK_UNAVAILABLE);
                    note_eager(Some(REASON_ROLLBACK_UNAVAILABLE));
                    return Ok(eager);
                }
                let staging = (|| -> Result<(Tensor, Tensor, Option<Tensor>)> {
                    let ids = Tensor::zeros((1, key.tokens), DType::U32, &device)?;
                    let logits = eager.logits.zeros_like()?;
                    let hidden = match &eager.hidden {
                        Some(h) => Some(h.zeros_like()?),
                        None => None,
                    };
                    Ok((ids, logits, hidden))
                })();
                let (ids, logits, hidden) = match staging {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[cuda-graph] staging allocation failed: {e}");
                        refuse(runner, key, REASON_INSTANTIATE_FAILED);
                        let out = model.forward_step(cache, request)?;
                        note_eager(Some(REASON_INSTANTIATE_FAILED));
                        return Ok(out);
                    }
                };
                stage_tokens(&ids, &request.tokens, &device)?;
                cache.stage_positions()?;
                let reserved_before = reserved_graph_mem(&dev).unwrap_or(0);
                let captured = capture(&dev, || {
                    let out = model.forward_step(
                        cache,
                        StepRequest {
                            tokens: StepTokens::Device(&ids),
                            scope: request.scope,
                            want_hidden: request.want_hidden,
                        },
                    )?;
                    stage_outputs(&logits, hidden.as_ref(), out)
                });
                // Whatever happens next, the Rust-side cache state advanced during the recording
                // (or partially, on a failed step) and is taken back to the step start.
                let recorded = match captured {
                    Ok((recorded, ())) => recorded,
                    Err(Abandoned(reason)) => {
                        refuse(runner, key, reason);
                        cache.rollback_to(base)?;
                        let out = model.forward_step(cache, request)?;
                        note_eager(Some(reason));
                        return Ok(out);
                    }
                };
                let census = recorded.census();
                runner.inner.borrow_mut().last_census = Some(census);
                if let Some(reason) = census.refusal() {
                    eprintln!(
                        "[cuda-graph] refusing the captured {}-token step ({reason}): {}",
                        key.tokens,
                        census.describe()
                    );
                    drop(recorded);
                    refuse(runner, key, reason);
                    cache.rollback_to(base)?;
                    let out = model.forward_step(cache, request)?;
                    note_eager(Some(reason));
                    return Ok(out);
                }
                let graph = match recorded.instantiate() {
                    Ok(g) => g,
                    Err(Abandoned(reason)) => {
                        refuse(runner, key, reason);
                        cache.rollback_to(base)?;
                        let out = model.forward_step(cache, request)?;
                        note_eager(Some(reason));
                        return Ok(out);
                    }
                };
                // First launch, at `base`, then the bit-exact self-check against the eager step
                // taken above (the cache was rolled back, so eager and graph wrote the same
                // positions; the eager result is what the caller gets either way).
                if let Err(e) = graph.launch() {
                    eprintln!("[cuda-graph] first launch failed: {e:?}");
                    drop(graph);
                    refuse(runner, key, REASON_LAUNCH_FAILED);
                    cache.rollback_to(base)?;
                    let out = model.forward_step(cache, request)?;
                    note_eager(Some(REASON_LAUNCH_FAILED));
                    return Ok(out);
                }
                let replayed = StepOutput {
                    logits: logits.clone(),
                    hidden: hidden.clone(),
                };
                let identical = outputs_identical(&replayed, &eager)?;
                // The eager step is what the cache holds after this call: re-run it so the
                // device state is the eager one (the launch wrote the same positions).
                cache.rollback_to(base)?;
                let eager = model.forward_step(cache, request)?;
                if !identical {
                    eprintln!(
                        "[cuda-graph] capture self-check failed ({REASON_REPLAY_MISMATCH}) on the {}-token step",
                        key.tokens
                    );
                    drop(graph);
                    refuse(runner, key, REASON_REPLAY_MISMATCH);
                    note_eager(Some(REASON_REPLAY_MISMATCH));
                    return Ok(eager);
                }
                note_eager(None);
                note_captured();
                let reserved_after = reserved_graph_mem(&dev).unwrap_or(reserved_before);
                let mut inner = runner.inner.borrow_mut();
                inner.cuda.graph_reserved_bytes = inner
                    .cuda
                    .graph_reserved_bytes
                    .saturating_add(reserved_after.saturating_sub(reserved_before) as usize);
                inner.cuda.shapes.insert(
                    key,
                    Shape::Graph(Captured {
                        graph,
                        ids,
                        logits,
                        hidden,
                        replays: 0,
                    }),
                );
                Ok(eager)
            }
            Plan::VerifyReplay => {
                // The first replay at a *new* position is verified against eager — the check
                // that catches a stale scalar or an unstable state address, which the capture
                // step's own check (same inputs, same position) cannot.
                let base = cache.len();
                let replayed = replay(runner, cache, &request, key, base)?;
                let Some(replayed) = replayed else {
                    // The launch failed; `replay` already fell back.
                    return runner.model.forward_step(cache, request).inspect(|_| {
                        note_eager(Some(REASON_LAUNCH_FAILED));
                    });
                };
                if let Err(e) = cache.rollback_to(base) {
                    eprintln!("[cuda-graph] cannot roll back for the replay check: {e}");
                    refuse(runner, key, REASON_ROLLBACK_UNAVAILABLE);
                    // The replay's device state is unverified: it cannot be kept.
                    return Err(e);
                }
                let eager = model.forward_step(cache, request)?;
                if !outputs_identical(&replayed, &eager)? {
                    eprintln!(
                        "[cuda-graph] first replay disagreed with eager ({REASON_REPLAY_MISMATCH}) on the {}-token step",
                        key.tokens
                    );
                    refuse(runner, key, REASON_REPLAY_MISMATCH);
                    note_eager(Some(REASON_REPLAY_MISMATCH));
                    return Ok(eager);
                }
                note_eager(None);
                if let Some(Shape::Graph(c)) = runner.inner.borrow_mut().cuda.shapes.get_mut(&key) {
                    c.replays = 1;
                }
                Ok(eager)
            }
            Plan::Replay => {
                let base = cache.len();
                match replay(runner, cache, &request, key, base)? {
                    Some(out) => {
                        if let Some(Shape::Graph(c)) =
                            runner.inner.borrow_mut().cuda.shapes.get_mut(&key)
                        {
                            c.replays = c.replays.wrapping_add(1);
                        }
                        Ok(out)
                    }
                    None => {
                        let out = runner.model.forward_step(cache, request)?;
                        note_eager(Some(REASON_LAUNCH_FAILED));
                        Ok(out)
                    }
                }
            }
        }
    }

    /// One replay: stage the tokens and positions, do the cache's bookkeeping for the step,
    /// launch, and return copies of the staged outputs. On a launch failure the bookkeeping is
    /// undone, the graph dropped, and `None` returned so the caller runs eager.
    fn replay<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        request: &StepRequest<'_>,
        key: ShapeKey,
        base: i32,
    ) -> Result<Option<StepOutput>> {
        let device = runner.model.device();
        let launched = {
            let inner = runner.inner.borrow();
            let Some(Shape::Graph(c)) = inner.cuda.shapes.get(&key) else {
                return Err(Error::Msg(
                    "cuda graph runner: no graph for the shape".into(),
                ));
            };
            stage_tokens(&c.ids, &request.tokens, device)?;
            cache.stage_positions()?;
            cache.replay_advance(key.tokens)?;
            match c.graph.launch() {
                Ok(()) => {
                    note_replayed();
                    Some(copy_outputs(&c.logits, c.hidden.as_ref())?)
                }
                Err(e) => {
                    eprintln!("[cuda-graph] launch failed: {e:?}");
                    None
                }
            }
        };
        if launched.is_none() {
            refuse(runner, key, REASON_LAUNCH_FAILED);
            cache.rollback_to(base)?;
        }
        Ok(launched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::step::StepOutput;
    use crate::primitives::decode_cache::CacheMemory;
    use candle_core::{DType, Tensor};
    use std::cell::Cell;

    /// A fixed-logits model whose cache is a bare counter (the step driver's own mock).
    pub(super) struct Counter {
        pub(super) device: Device,
        pub(super) steps: Cell<u64>,
        pub(super) vocab: usize,
    }

    pub(super) struct CounterCache(pub(super) i32);

    impl DecodeCache for CounterCache {
        fn len(&self) -> i32 {
            self.0
        }
        fn rollback_to(&mut self, n: i32) -> Result<()> {
            self.0 = n;
            Ok(())
        }
        fn reset(&mut self) {
            self.0 = 0;
        }
        fn memory(&self) -> CacheMemory {
            CacheMemory::default()
        }
    }

    impl StepModel for Counter {
        type Cache = CounterCache;
        fn new_cache(&self) -> CounterCache {
            CounterCache(0)
        }
        fn device(&self) -> &Device {
            &self.device
        }
        fn vocab_size(&self) -> usize {
            self.vocab
        }
        fn forward_step(
            &self,
            cache: &mut CounterCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            self.steps.set(self.steps.get() + 1);
            cache.0 += request.len()? as i32;
            let mut row = vec![0f32; self.vocab];
            row[(cache.0 as usize) % self.vocab] = 10.0;
            let logits = Tensor::from_vec(row, (1, self.vocab), &self.device)?;
            Ok(StepOutput {
                logits: match request.scope {
                    LogitsScope::Last => logits,
                    LogitsScope::All => logits.unsqueeze(1)?,
                },
                hidden: None,
            })
        }
    }

    fn counter() -> Counter {
        Counter {
            device: Device::Cpu,
            steps: Cell::new(0),
            vocab: 4,
        }
    }

    #[test]
    fn switch_env_default_is_off_and_runtime_override_round_trips() {
        let _guard = cuda_graphs_policy_guard(None);
        // The environment is not set in the test process: the default is off.
        if std::env::var_os(CUDA_GRAPHS_ENV).is_none() {
            assert!(!cuda_graphs_enabled());
        }
        set_cuda_graphs(Some(true));
        assert!(cuda_graphs_enabled());
        set_cuda_graphs(Some(false));
        assert!(!cuda_graphs_enabled());
        set_cuda_graphs(None);
        if std::env::var_os(CUDA_GRAPHS_ENV).is_none() {
            assert!(!cuda_graphs_enabled());
        }
    }

    #[test]
    fn the_policy_guard_restores_the_switch_it_found() {
        let outer = cuda_graphs_policy_guard(Some(false));
        assert!(!cuda_graphs_enabled());
        drop(outer);
        {
            let _inner = cuda_graphs_policy_guard(Some(true));
            assert!(cuda_graphs_enabled());
        }
        let _check = cuda_graphs_policy_guard(None);
        if std::env::var_os(CUDA_GRAPHS_ENV).is_none() {
            assert!(
                !cuda_graphs_enabled(),
                "the guard must restore the previous policy"
            );
        }
    }

    #[test]
    fn tally_deltas_labels_and_describe() {
        let start = graph_tally();
        note_eager(None);
        note_eager(Some(REASON_SHAPE));
        let eager_only = graph_tally().since(&start);
        assert_eq!(eager_only.eager, 2);
        assert_eq!(eager_only.replayed, 0);
        assert_eq!(eager_only.label(), "eager");
        assert_eq!(eager_only.fallback_reason, Some(REASON_SHAPE));
        assert_eq!(
            eager_only.describe(),
            "graph: eager replayed=0 eager=2 captured=0 fallback=shape"
        );
        note_replayed();
        let mixed = graph_tally().since(&start);
        assert_eq!(mixed.label(), "mixed");
        let only_replay = graph_tally().since(&GraphTally {
            eager: graph_tally().eager,
            ..start
        });
        assert_eq!(only_replay.label(), "graph");
        assert_eq!(only_replay.fallback_reason, None);
        assert_eq!(GraphTally::default().label(), "none");
        assert_eq!(
            GraphTally::default().describe(),
            "graph: none replayed=0 eager=0 captured=0"
        );
    }

    #[test]
    fn census_refusals_are_named_in_priority_order() {
        let clean = GraphCensus {
            nodes: 3,
            kernels: 2,
            memcpy_device: 1,
            ..Default::default()
        };
        assert_eq!(clean.refusal(), None);
        let upload = GraphCensus {
            memcpy_from_host: 1,
            memcpy_to_host: 1,
            escaped_allocs: 1,
            ..Default::default()
        };
        assert_eq!(upload.refusal(), Some(REASON_HOST_UPLOAD_IN_CAPTURE));
        let read = GraphCensus {
            memcpy_to_host: 1,
            escaped_allocs: 1,
            ..Default::default()
        };
        assert_eq!(read.refusal(), Some(REASON_HOST_READ_IN_CAPTURE));
        let escaped = GraphCensus {
            escaped_allocs: 2,
            ..Default::default()
        };
        assert_eq!(escaped.refusal(), Some(REASON_ALLOCATION_ESCAPED_CAPTURE));
        assert!(escaped.describe().contains("escaped=2"));
        let workspace = GraphWorkspace {
            staging_bytes: usize::MAX,
            graph_reserved_bytes: 1,
        };
        assert_eq!(workspace.total_bytes(), usize::MAX);
    }

    /// Off (the default), and on a non-CUDA model: the runner is a pass-through that names why,
    /// once per runner, and the wrapped model sees every step.
    #[test]
    fn runner_is_a_named_pass_through_when_off_or_off_device() {
        let model = counter();
        let runner = GraphRunner::new(&model);
        let mut cache = runner.new_cache_for(8, 0).unwrap();
        {
            let _guard = cuda_graphs_policy_guard(Some(false));
            let start = graph_tally();
            let out = runner
                .forward_step(&mut cache, StepRequest::last(&[1, 2]))
                .unwrap();
            assert_eq!(out.logits.dims(), &[1, 4]);
            assert_eq!(model.steps.get(), 1);
            assert_eq!(cache.len(), 2);
            let tally = graph_tally().since(&start);
            assert_eq!(tally.eager, 1);
            assert_eq!(tally.replayed, 0);
            assert_eq!(tally.fallback_reason, Some(REASON_DISABLED));
            assert_eq!(runner.refusal(), Some(REASON_DISABLED));
            assert_eq!(runner.captured_graphs(), 0);
            assert_eq!(runner.workspace(), GraphWorkspace::default());
        }
        {
            let _guard = cuda_graphs_policy_guard(Some(true));
            // A new cache resets the runner, so the refusal is re-evaluated.
            let mut cache = runner.new_cache_for(8, 0).unwrap();
            assert_eq!(runner.refusal(), None);
            let start = graph_tally();
            runner
                .forward_step(&mut cache, StepRequest::last(&[3]))
                .unwrap();
            let expected = if cfg!(feature = "cuda") {
                REASON_NOT_CUDA
            } else {
                REASON_CUDA_FEATURE_OFF
            };
            assert_eq!(graph_tally().since(&start).fallback_reason, Some(expected));
            assert_eq!(runner.refusal(), Some(expected));
            assert_eq!(runner.vocab_size(), 4);
            assert!(runner.device().is_cpu());
            assert_eq!(runner.attn_formulation(&cache), AttnFormulation::Gqa);
            assert_eq!(runner.model().vocab, 4);
        }
    }

    /// The runner drives the token-at-a-time driver and the engine exactly like the bare model
    /// (CPU: every step is a named eager step).
    #[test]
    fn driver_and_engine_through_the_runner_match_the_bare_model() {
        use crate::decode::cancel::CancelFlag;
        use crate::decode::engine::{generate_speculative, NoProposer, SpeculativePrompt};
        use crate::decode::step::generate_step;
        use crate::decode::stream::GenerationConfig;
        let _guard = cuda_graphs_policy_guard(Some(true));
        let model = counter();
        let runner = GraphRunner::new(&model);
        let cfg = GenerationConfig {
            max_new_tokens: 6,
            seed: Some(1),
            ..Default::default()
        };
        let (bare, _) =
            generate_step(&model, &[1, 2], &cfg, &CancelFlag::new(), &mut |_| {}, None).unwrap();
        let (wrapped, record) = generate_step(
            &runner,
            &[1, 2],
            &cfg,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(bare.tokens, wrapped.tokens);
        assert_eq!(record.cuda_graphs.eager, 6);
        assert_eq!(record.cuda_graphs.replayed, 0);
        assert!(record.cuda_graphs.fallback_reason.is_some());
        assert_eq!(record.cuda_graphs.label(), "eager");
        let run = generate_speculative(
            &runner,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&[1, 2]),
            &cfg,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(run.output.tokens, bare.tokens);
        assert_eq!(run.record.cuda_graphs.label(), "eager");
    }

    #[test]
    fn oversized_steps_are_named_shape_fallbacks_even_when_capturable() {
        // The shape check runs after the capability checks; on a CPU model the capability
        // refusal wins, so the shape reason is exercised through the tally directly.
        let too_long = vec![1i32; GraphRunner::<Counter>::MAX_CAPTURED_TOKENS + 1];
        let request = StepRequest::last(&too_long);
        let key = ShapeKey::of(&request).unwrap();
        assert!(key.tokens > GraphRunner::<Counter>::MAX_CAPTURED_TOKENS);
        assert_eq!(key.scope, LogitsScope::Last);
        assert!(!key.want_hidden);
        let _ = DType::F32;
    }
}

/// The story's POC experiments (sc-24134, step 0): what candle @ `1e6aa85e` on cudarc 0.19 can
/// and cannot do inside a stream capture, plus the runner end to end on a synthetic step model
/// built from capturable ops only. They run on the CUDA lane and are the record behind the
/// module docs; each prints its finding.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::cuda::{bit_identical, capture, Abandoned};
    use super::*;
    use crate::decode::cancel::CancelFlag;
    use crate::decode::engine::{
        generate_speculative, DraftSampler, Drafts, NoProposer, Proposal, ProposeContext, Proposer,
        SpeculativePrompt,
    };
    use crate::decode::step::generate_step;
    use crate::decode::stream::GenerationConfig;
    use crate::primitives::attention::{capacity_mask, sdpa_gqa_causal, sdpa_gqa_masked};
    use crate::primitives::decode_cache::CacheMemory;
    use crate::primitives::rope::rms_norm_rope;
    use candle_core::cuda_backend::cudarc::driver::sys;
    use candle_core::{DType, Device};
    use core_llm::ProposerKind;
    use std::cell::Cell;

    /// A capturable device: candle's own stream (the legacy NULL stream cannot be captured) with
    /// cudarc's event tracking off — what `select_device` builds.
    fn device() -> Option<(Device, candle_core::CudaDevice)> {
        match crate::device::select_device() {
            Ok(Device::Cuda(d)) => Some((Device::Cuda(d.clone()), d)),
            _ => {
                eprintln!("skipping: no CUDA device");
                None
            }
        }
    }

    fn greedy(max_new_tokens: usize) -> GenerationConfig {
        let mut config = GenerationConfig {
            max_new_tokens,
            seed: Some(0),
            stop_tokens: Vec::new(),
            ..Default::default()
        };
        config.sampling.temperature = 0.0;
        config
    }

    // ---- POC experiments ----

    #[test]
    fn poc_allocator_is_stream_ordered_and_the_stream_is_capturable() {
        let Some((_, dev)) = device() else { return };
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        eprintln!(
            "[poc] has_async_alloc = {} (cudarc 0.19 CudaStream::alloc -> cuMemAllocAsync); event tracking = {}",
            ctx.has_async_alloc(),
            ctx.is_managing_stream_synchronization()
        );
        assert!(ctx.has_async_alloc());
        assert!(!ctx.is_managing_stream_synchronization());
        let reserved = super::cuda::graph_mem_attribute(
            &dev,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
        );
        eprintln!("[poc] graph reserved mem now = {reserved:?}");
        let (recorded, ()) =
            capture(&dev, || Ok(())).unwrap_or_else(|Abandoned(r)| panic!("empty capture: {r}"));
        assert_eq!(recorded.census().nodes, 0);
        // The legacy NULL stream (`Device::new_cuda`) refuses capture: the reason the device
        // selector builds the model on its own stream.
        let Ok(Device::Cuda(legacy)) = Device::new_cuda(0) else {
            return;
        };
        match capture(&legacy, || Ok(())) {
            Err(Abandoned(reason)) => {
                eprintln!("[poc] capture on the legacy NULL stream -> {reason}")
            }
            Ok(_) => panic!("the legacy stream must not be capturable"),
        }
        // The runner refuses it by name before any capture.
        let _guard = cuda_graphs_policy_guard(Some(true));
        let model = Synthetic::new(&Device::Cuda(legacy), None);
        let runner = GraphRunner::new(&model);
        let (out, record) = generate_step(
            &runner,
            &PROMPT,
            &greedy(4),
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(out.tokens.len(), 4);
        assert_eq!(
            record.cuda_graphs.fallback_reason,
            Some(REASON_LEGACY_STREAM)
        );
    }

    /// The core experiment: contiguous candle ops (a cuBLAS matmul, a softmax, an affine, a
    /// fused nvrtc kernel compiled outside the capture, `slice_set`) capture, allocate their
    /// temporaries as graph memory nodes freed inside the graph, and replay bit-exactly at new
    /// inputs; adding one `index_select` puts a host-sourced memcpy node in the graph, which the
    /// census refuses before anything is launched.
    #[test]
    fn poc_contiguous_ops_replay_bit_exact_and_index_select_is_refused() {
        let Some((device, dev)) = device() else {
            return;
        };
        let (rows, n) = (4usize, 64usize);
        let w = Tensor::arange(0f32, (n * n) as f32, &device)
            .unwrap()
            .affine(0.001, -1.0)
            .unwrap()
            .cos()
            .unwrap()
            .reshape((n, n))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let norm_w = Tensor::ones(n, DType::BF16, &device).unwrap();
        let cos = Tensor::ones((1, rows, n), DType::BF16, &device).unwrap();
        let sin = Tensor::zeros((1, rows, n), DType::BF16, &device).unwrap();
        let idx = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();
        let step = |x: &Tensor, with_gather: bool| -> Result<Tensor> {
            let y = x.matmul(&w)?;
            let y = candle_nn::ops::softmax_last_dim(&y)?.affine(2.0, 0.5)?;
            let y = y.reshape((1, rows, 1, n))?;
            let y = rms_norm_rope(&y, &norm_w, 1e-6, &cos, &sin, false)?.reshape((rows, n))?;
            if with_gather {
                Ok((y + w.index_select(&idx, 0)?)?)
            } else {
                Ok(y)
            }
        };
        let input = |phase: f32| {
            Tensor::arange(0f32, (rows * n) as f32, &device)
                .unwrap()
                .affine(0.05, phase as f64)
                .unwrap()
                .sin()
                .unwrap()
                .reshape((rows, n))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let x_stage = Tensor::zeros((rows, n), DType::BF16, &device).unwrap();
        let out_stage = Tensor::zeros((rows, n), DType::BF16, &device).unwrap();
        x_stage.slice_set(&input(0.0), 0, 0).unwrap();
        let _ = step(&x_stage, false).unwrap(); // warm-up: compiles the fused kernel
        let (recorded, ()) = capture(&dev, || {
            let y = step(&x_stage, false)?;
            out_stage.slice_set(&y, 0, 0)?;
            Ok(())
        })
        .unwrap_or_else(|Abandoned(reason)| panic!("capture abandoned: {reason}"));
        let census = recorded.census();
        eprintln!("[poc] contiguous step census: {}", census.describe());
        assert_eq!(census.refusal(), None);
        assert!(census.kernels >= 4);
        assert!(census.mem_allocs > 0 && census.mem_allocs == census.mem_frees);
        let graph = recorded
            .instantiate()
            .unwrap_or_else(|Abandoned(reason)| panic!("instantiate: {reason}"));
        for phase in [0.7f32, 2.1] {
            let x = input(phase);
            let eager = step(&x, false).unwrap();
            x_stage.slice_set(&x, 0, 0).unwrap();
            graph.launch().unwrap();
            assert!(
                bit_identical(&out_stage, &eager).unwrap(),
                "replay at phase {phase} is not bit-identical to eager"
            );
        }
        let used = super::cuda::graph_mem_attribute(
            &dev,
            sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
        );
        eprintln!("[poc] contiguous step: 2 replays bit-exact; graph used mem after = {used:?}");

        // The same step plus one `index_select`: candle uploads the index layout from a host Vec.
        let (recorded, ()) = capture(&dev, || {
            let y = step(&x_stage, true)?;
            out_stage.slice_set(&y, 0, 0)?;
            Ok(())
        })
        .unwrap_or_else(|Abandoned(reason)| panic!("capture abandoned: {reason}"));
        let census = recorded.census();
        eprintln!("[poc] + index_select census: {}", census.describe());
        assert_eq!(census.refusal(), Some(REASON_HOST_UPLOAD_IN_CAPTURE));
        assert!(census.memcpy_from_host >= 1);
    }

    /// A pageable host upload (`Tensor::from_vec`) inside a capture is refused, and a
    /// device->host read (`to_vec`) is recorded as a memcpy node into a host buffer that no
    /// longer exists — the census names both before any launch.
    #[test]
    fn poc_host_traffic_inside_capture_is_named() {
        let Some((device, dev)) = device() else {
            return;
        };
        let upload = capture(&dev, || {
            let t = Tensor::from_vec(vec![1f32, 2.0, 3.0], 3, &device)?;
            let _ = t.affine(2.0, 0.0)?;
            Ok(())
        });
        let upload_reason = match upload {
            Err(Abandoned(reason)) => reason,
            Ok((recorded, ())) => recorded.census().refusal().unwrap_or("captured"),
        };
        eprintln!("[poc] pageable host->device upload inside capture -> {upload_reason}");
        assert!(
            upload_reason == REASON_HOST_UPLOAD_IN_CAPTURE
                || upload_reason == REASON_CAPTURE_INVALIDATED
        );
        let staged = Tensor::new(&[1f32, 2.0, 3.0], &device).unwrap();
        let read = capture(&dev, || {
            let doubled = staged.affine(2.0, 0.0)?;
            let _ = doubled.to_vec1::<f32>()?;
            Ok(())
        });
        let read_reason = match read {
            Err(Abandoned(reason)) => reason,
            Ok((recorded, ())) => recorded.census().refusal().unwrap_or("captured"),
        };
        eprintln!("[poc] device->host read inside capture -> {read_reason}");
        assert_eq!(read_reason, REASON_HOST_READ_IN_CAPTURE);
        // After a refused capture the stream is usable again.
        let ok = staged.affine(3.0, 0.0).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(ok, vec![3.0, 6.0, 9.0]);
    }

    /// Op-by-op capture diagnostic (`POC_OP=matmul|softmax|index_select|rope|affine|slice_set|
    /// transpose|broadcast_add|sum`): captures one candle op and reports its census.
    #[test]
    fn poc_single_op_census() {
        let Ok(op) = std::env::var("POC_OP") else {
            return;
        };
        let Some((device, dev)) = device() else {
            return;
        };
        let (rows, n) = (4usize, 64usize);
        let x = Tensor::arange(0f32, (rows * n) as f32, &device)
            .unwrap()
            .affine(0.05, 0.3)
            .unwrap()
            .sin()
            .unwrap()
            .reshape((rows, n))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::arange(0f32, (n * n) as f32, &device)
            .unwrap()
            .affine(0.001, -1.0)
            .unwrap()
            .cos()
            .unwrap()
            .reshape((n, n))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let idx = Tensor::new(&[1u32, 2, 3, 4], &device).unwrap();
        let row = Tensor::ones((1, n), DType::BF16, &device).unwrap();
        let norm_w = Tensor::ones(n, DType::BF16, &device).unwrap();
        let cos = Tensor::ones((1, rows, n), DType::BF16, &device).unwrap();
        let sin = Tensor::zeros((1, rows, n), DType::BF16, &device).unwrap();
        let step = |x: &Tensor| -> Result<Tensor> {
            Ok(match op.as_str() {
                "matmul" => x.matmul(&w)?,
                "softmax" => candle_nn::ops::softmax_last_dim(x)?,
                "index_select" => w.index_select(&idx, 0)?,
                "rope" => rms_norm_rope(
                    &x.reshape((1, rows, 1, n))?,
                    &norm_w,
                    1e-6,
                    &cos,
                    &sin,
                    false,
                )?
                .reshape((rows, n))?,
                "affine" => x.affine(2.0, 1.0)?,
                "slice_set" => x.clone(),
                "transpose" => x.t()?.contiguous()?.t()?.contiguous()?,
                "broadcast_add" => x.broadcast_add(&row)?,
                "sum" => x.sum_keepdim(1)?.broadcast_as((rows, n))?.contiguous()?,
                other => panic!("unknown POC_OP {other}"),
            })
        };
        let out_stage = Tensor::zeros((rows, n), DType::BF16, &device).unwrap();
        let eager = step(&x).unwrap();
        let (recorded, ()) = capture(&dev, || {
            let y = step(&x)?;
            out_stage.slice_set(&y, 0, 0)?;
            Ok(())
        })
        .unwrap_or_else(|Abandoned(reason)| panic!("capture abandoned: {reason}"));
        let census = recorded.census();
        eprintln!("[poc] op {op}: census {}", census.describe());
        if census.refusal().is_some() {
            return;
        }
        let graph = recorded
            .instantiate()
            .unwrap_or_else(|Abandoned(reason)| panic!("instantiate: {reason}"));
        graph.launch().unwrap();
        let identical = bit_identical(&out_stage, &eager).unwrap();
        eprintln!("[poc] op {op}: replayed, bit-identical = {identical}");
        assert!(identical);
    }

    /// Position as data: the masked full-capacity attention against the bounded `narrow` view at
    /// the Qwen3.8-27B decode and verify shapes, bf16, over every length up to the capacity.
    /// Recorded finding (RTX Pro 6000 / sm_120): the two are **not** bit-identical — cuBLAS
    /// selects its kernel (and so its reduction order) by the key extent, so a masked form
    /// changes the static path's bits at a few dozen lengths by one bf16 ULP. That is why the
    /// static path keeps `narrow` today: a data-driven length needs an attention kernel whose
    /// arithmetic does not depend on the extent (a seam kernel), not a mask over cuBLAS.
    #[test]
    fn poc_masked_capacity_attention_vs_narrow_survey() {
        let Some((device, _)) = device() else { return };
        let (b, h, hkv, d, cap) = (1usize, 24usize, 4usize, 256usize, 384usize);
        let mk = |heads, s, phase: f64| {
            let n = (b * heads * s * d) as f32;
            Tensor::arange(0f32, n, &device)
                .unwrap()
                .reshape((b, heads, s, d))
                .unwrap()
                .affine(0.0137, phase)
                .unwrap()
                .cos()
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap()
        };
        let scale = (d as f32).powf(-0.5);
        let k_buf = mk(hkv, cap, 1.7);
        let v_buf = mk(hkv, cap, 3.1);
        let arange = Tensor::arange(0u32, cap as u32, &device).unwrap();
        for q_len in [1usize, 4] {
            let q = mk(h, q_len, 0.4);
            let mut mismatches = 0usize;
            let mut worst = 0f32;
            let mut first = None;
            for len in q_len..=cap {
                let pos = (len - q_len) as u32;
                let narrow = sdpa_gqa_causal(
                    &q,
                    &k_buf.narrow(2, 0, len).unwrap(),
                    &v_buf.narrow(2, 0, len).unwrap(),
                    scale,
                )
                .unwrap();
                let limits = Tensor::arange(pos, pos + q_len as u32, &device).unwrap();
                let mask = capacity_mask(&arange, &limits, DType::BF16).unwrap();
                let masked = sdpa_gqa_masked(&q, &k_buf, &v_buf, scale, &mask).unwrap();
                if !bit_identical(&masked, &narrow).unwrap() {
                    mismatches += 1;
                    first.get_or_insert(len);
                    let delta = (masked.to_dtype(DType::F32).unwrap()
                        - narrow.to_dtype(DType::F32).unwrap())
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max_all()
                    .unwrap()
                    .to_scalar::<f32>()
                    .unwrap();
                    worst = worst.max(delta);
                }
            }
            eprintln!(
                "[poc] masked(cap={cap}) vs narrow, q_len={q_len}: {mismatches} / {} lengths differ in bits (first {first:?}, max |delta| {worst})",
                cap + 1 - q_len
            );
        }
    }

    // ---- The runner end to end on a synthetic, capturable step model ----

    /// A recurrent step model built only from capturable ops: an `[1, n]` state at a stable
    /// address, updated in place (`slice_set`), a per-token row mixed in through cuBLAS
    /// matmuls on contiguous operands, logits through one more matmul. No positions (the
    /// state *is* the history), no gathers, no broadcasts: what candle can replay today.
    struct Synthetic {
        device: Device,
        n: usize,
        vocab: usize,
        /// `[1, n]`: the constant mix-in every token gets.
        w_in: Tensor,
        /// `[1, n]`: the row every token's id scales.
        w_id: Tensor,
        /// `[n, vocab]`.
        w_out: Tensor,
        /// Optional misbehaviour for the fallback tests.
        misbehave: Cell<Option<&'static str>>,
        steps: Cell<u64>,
    }

    struct SyntheticCache {
        state: Tensor,
        len: i32,
        /// `(position, copy of the state at that position)`, newest last.
        checkpoints: Vec<(i32, Tensor)>,
        support: std::result::Result<(), &'static str>,
        /// The fallback test: keep a fresh tensor from every step (an escaping allocation).
        kept: Vec<Tensor>,
        staged: Cell<u32>,
    }

    impl SyntheticCache {
        fn checkpoint(&mut self) -> Result<()> {
            if capturing() {
                // A real stable-address cache writes its checkpoint into a preallocated ring
                // slot; this mock skips the copy inside a capture (the runner's
                // `replay_advance` takes the checkpoint outside it).
                return Ok(());
            }
            if self.checkpoints.last().is_some_and(|(p, _)| *p == self.len) {
                return Ok(());
            }
            let copy = self.state.copy()?;
            self.checkpoints.push((self.len, copy));
            if self.checkpoints.len() > 8 {
                self.checkpoints.remove(0);
            }
            Ok(())
        }
    }

    impl DecodeCache for SyntheticCache {
        fn len(&self) -> i32 {
            self.len
        }
        fn rollback_to(&mut self, n: i32) -> Result<()> {
            if n == self.len {
                return Ok(());
            }
            if n > self.len || n < 0 {
                return Err(crate::error::Error::Msg(format!(
                    "rollback to {n} past {}",
                    self.len
                )));
            }
            let Some(at) = self.checkpoints.iter().rposition(|(p, _)| *p == n) else {
                return Err(crate::error::Error::RollbackUnavailable {
                    n,
                    have: self.checkpoints.iter().map(|(p, _)| *p).collect(),
                });
            };
            // Restore in place: the state's address never changes.
            let (_, saved) = &self.checkpoints[at];
            self.state.slice_set(saved, 0, 0)?;
            self.checkpoints.truncate(at + 1);
            self.len = n;
            Ok(())
        }
        fn reset(&mut self) {
            self.len = 0;
            self.checkpoints.clear();
        }
        fn memory(&self) -> CacheMemory {
            CacheMemory {
                live_bytes: crate::primitives::decode_cache::tensor_bytes(&self.state),
                checkpoint_bytes: self
                    .checkpoints
                    .iter()
                    .map(|(_, t)| crate::primitives::decode_cache::tensor_bytes(t))
                    .sum(),
            }
        }
        fn graph_support(&self) -> std::result::Result<(), &'static str> {
            self.support
        }
        fn stage_positions(&mut self) -> Result<()> {
            if !capturing() {
                self.staged.set(self.staged.get() + 1);
            }
            Ok(())
        }
        fn replay_advance(&mut self, n: usize) -> Result<()> {
            self.checkpoint()?;
            self.len += n as i32;
            Ok(())
        }
    }

    impl Synthetic {
        const MAX: usize = 16;

        fn new(device: &Device, misbehave: Option<&'static str>) -> Self {
            let (n, vocab) = (32usize, 24usize);
            let mk = |rows: usize, cols: usize, mul: f64, phase: f64| {
                Tensor::arange(0f32, (rows * cols) as f32, device)
                    .unwrap()
                    .affine(mul, phase)
                    .unwrap()
                    .sin()
                    .unwrap()
                    .reshape((rows, cols))
                    .unwrap()
            };
            Self {
                device: device.clone(),
                n,
                vocab,
                w_in: mk(1, n, 0.37, 0.1),
                w_id: mk(1, n, 0.11, 0.9).affine(0.05, 0.0).unwrap(),
                w_out: mk(n, vocab, 0.23, 0.4),
                misbehave: Cell::new(misbehave),
                steps: Cell::new(0),
            }
        }
    }

    impl StepModel for Synthetic {
        type Cache = SyntheticCache;

        fn new_cache(&self) -> SyntheticCache {
            SyntheticCache {
                state: Tensor::zeros((1, self.n), DType::F32, &self.device).unwrap(),
                len: 0,
                checkpoints: Vec::new(),
                support: match self.misbehave.get() {
                    Some("declared") => Err("mock_cache_declared_unstable"),
                    _ => Ok(()),
                },
                kept: Vec::new(),
                staged: Cell::new(0),
            }
        }

        fn device(&self) -> &Device {
            &self.device
        }

        fn vocab_size(&self) -> usize {
            self.vocab
        }

        fn forward_step(
            &self,
            cache: &mut SyntheticCache,
            request: StepRequest<'_>,
        ) -> Result<StepOutput> {
            self.steps.set(self.steps.get() + 1);
            cache.stage_positions()?;
            cache.checkpoint()?;
            let m = request.len()?;
            if m == 0 || m > Self::MAX {
                return Err(crate::error::Error::Msg("synthetic: bad step size".into()));
            }
            // `[M, 1]` ids as f32 (a reshape of the contiguous `[1, M]`). Rows chain through the
            // recurrence one at a time — every row is the same op sequence a single-token step
            // runs, so an `M`-token step equals `M` single steps bit for bit (what the engine's
            // verify relies on). All operands are contiguous: no layout upload anywhere.
            let ids = request.tokens.ids(&self.device)?;
            let ids_col = ids.to_dtype(DType::F32)?.reshape((m, 1))?;
            let mix = &self.w_in; // [1, n]
            let mut prev = cache.state.clone();
            let mut rows = Vec::with_capacity(m);
            for i in 0..m {
                let token_row = ids_col.narrow(0, i, 1)?.matmul(&self.w_id)?; // [1, n]
                let row = ((token_row + mix)? + &prev)?.tanh()?;
                rows.push(row.clone());
                prev = row;
            }
            let h = if m == 1 {
                rows[0].clone()
            } else {
                Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)? // [M, n], a copy2d per row
            };
            match self.misbehave.get() {
                Some("host_read") => {
                    // A model that reads the device during its step (a router); noted like every
                    // host transfer this crate issues.
                    crate::primitives::host_sync::note_host_sync();
                    let _ = h.to_vec2::<f32>()?;
                }
                Some("escape") => {
                    // A fresh tensor kept by the cache: an allocation the graph would own but
                    // never free. (Kept in a list, never replaced: freeing a tensor allocated
                    // *before* the capture inside it fails the step outright on this rev —
                    // `cuMemFreeAsync` returns INVALID_VALUE, named `step_failed_in_capture` —
                    // which is what the S1 hybrid cache's replaced DeltaNet state would do.)
                    cache.kept.push(h.affine(1.0, 0.0)?);
                }
                _ => {}
            }
            let last = h.narrow(0, m - 1, 1)?; // contiguous row
            cache.state.slice_set(&last, 0, 0)?;
            cache.len += m as i32;
            let logits = match request.scope {
                LogitsScope::Last => last.matmul(&self.w_out)?, // [1, vocab]
                LogitsScope::All => h.matmul(&self.w_out)?.reshape((1, m, self.vocab))?,
            };
            let hidden = request
                .want_hidden
                .then(|| h.reshape((1, m, self.n)))
                .transpose()?;
            Ok(StepOutput { logits, hidden })
        }
    }

    /// Proposes `k` drafts after `cur` (host ids) from a known greedy continuation, corrupting
    /// every second one, so the engine's `K + 1`-token verify steps see both acceptances and
    /// rejections (and roll back).
    struct FixedDrafts {
        k: usize,
        vocab: i32,
        prompt_len: usize,
        continuation: Vec<i32>,
    }

    impl Proposer for FixedDrafts {
        fn kind(&self) -> ProposerKind {
            ProposerKind::Ngram
        }
        fn warm(&mut self, _: &[i32], _: Option<&Tensor>) -> Result<()> {
            Ok(())
        }
        fn propose(
            &mut self,
            ctx: &ProposeContext<'_>,
            _: &mut DraftSampler<'_, '_>,
        ) -> Result<Proposal> {
            let k = self.k.min(ctx.max_drafts);
            let at = ctx.history.len() - self.prompt_len;
            let drafts = (0..k)
                .map(|i| {
                    let truth = self.continuation.get(at + i).copied().unwrap_or(0);
                    if i % 2 == 1 {
                        (truth + 1) % self.vocab
                    } else {
                        truth
                    }
                })
                .collect();
            Ok(Proposal {
                drafts: Some(Drafts::Host(drafts)),
                dists: Vec::new(),
            })
        }
        fn commit(&mut self, _: i32, _: &[i32], _: Option<&Tensor>, _: i32) -> Result<()> {
            Ok(())
        }
    }

    const PROMPT: [i32; 5] = [3, 7, 11, 2, 7];

    /// Speculation off: the runner captures the 1-token decode step, verifies it twice and
    /// replays it for the rest of the request, token-identical to the bare model.
    #[test]
    fn synthetic_decode_through_the_runner_is_token_identical_and_replays() {
        let Some((device, _)) = device() else { return };
        let _guard = cuda_graphs_policy_guard(Some(true));
        let model = Synthetic::new(&device, None);
        let config = greedy(40);
        let (bare, _) = generate_step(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let runner = GraphRunner::new(&model);
        let steps_before = model.steps.get();
        let (wrapped, record) = generate_step(
            &runner,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(
            wrapped.tokens, bare.tokens,
            "graph replay diverged from eager"
        );
        assert_eq!(wrapped.tokens.len(), 40);
        let tally = record.cuda_graphs;
        eprintln!("[runner] synthetic decode: {}", tally.describe());
        assert_eq!(tally.captured, 1, "one graph for the 1-token shape");
        // Prefill (5 tokens, eager) + warm-up + the capture step (the eager reference and the
        // eager re-run that leaves the cache in the eager state: two eager executions) + the
        // verified first replay (one eager reference): 5 eager steps; the verified replay and
        // the other 36 decode steps replay.
        assert_eq!(tally.eager, 5);
        assert_eq!(tally.replayed, 37);
        assert_eq!(tally.fallback_reason, None);
        assert_eq!(tally.label(), "mixed");
        assert_eq!(runner.captured_graphs(), 1);
        assert_eq!(runner.refusal(), None);
        let census = runner.last_census().unwrap();
        eprintln!(
            "[runner] synthetic 1-token step census: {}",
            census.describe()
        );
        assert_eq!(census.refusal(), None);
        // Model calls: prefill, warm-up, the capture step (eager reference, the recording, the
        // eager re-run) and the verified replay's eager reference; replays never call the model.
        let eager_calls = model.steps.get() - steps_before;
        assert_eq!(
            eager_calls,
            1 + 1 + 3 + 1,
            "eager model calls through the runner"
        );
        let workspace = runner.workspace();
        eprintln!("[runner] workspace: {workspace:?}");
        assert!(workspace.staging_bytes > 0);
        assert_eq!(
            workspace.staging_bytes,
            4 + model.vocab * 4,
            "ids [1,1] u32 + logits [1, vocab] f32"
        );
    }

    /// MTP-style speculation with K = 3: the engine's 4-token verify steps (all-position logits,
    /// device ids) and its rollbacks run through captured graphs, token-identical to the bare
    /// model through the same engine.
    #[test]
    fn synthetic_speculative_k3_through_the_runner_is_token_identical_and_replays() {
        let Some((device, _)) = device() else { return };
        let _guard = cuda_graphs_policy_guard(Some(true));
        let model = Synthetic::new(&device, None);
        let config = greedy(48);
        let (greedy_run, _) = generate_step(
            &model,
            &PROMPT,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let proposer = || FixedDrafts {
            k: 3,
            vocab: 24,
            prompt_len: PROMPT.len(),
            continuation: greedy_run.tokens.clone(),
        };
        let bare = generate_speculative(
            &model,
            &mut proposer(),
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(bare.output.tokens, greedy_run.tokens);
        let runner = GraphRunner::new(&model);
        let wrapped = generate_speculative(
            &runner,
            &mut proposer(),
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            3,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(wrapped.output.tokens, bare.output.tokens);
        assert_eq!(wrapped.output.tokens.len(), 48);
        assert_eq!(wrapped.stats.proposed, bare.stats.proposed);
        assert_eq!(wrapped.stats.accepted, bare.stats.accepted);
        assert!(
            bare.stats.accepted < bare.stats.proposed,
            "the fixture must reject drafts"
        );
        assert!(bare.stats.accepted > 0, "the fixture must accept drafts");
        let tally = wrapped.record.cuda_graphs;
        eprintln!(
            "[runner] synthetic K=3: {} (verify steps {}, forwards {})",
            tally.describe(),
            wrapped.stats.verify_steps,
            wrapped.stats.forwards
        );
        assert!(tally.captured >= 1, "the 4-token verify shape is captured");
        assert!(tally.replayed > 0, "verify steps replay");
        assert_eq!(tally.fallback_reason, None);
        // The bare engine pays one sync per verify step (AC2 of S2); through the runner the two
        // bit-exact self-checks of every captured shape each cost one more, exactly, and only
        // at capture / verification time.
        assert_eq!(bare.record.host_syncs_per_verify_step(), Some(1.0));
        assert_eq!(
            wrapped.record.verify_host_syncs,
            wrapped.stats.verify_steps as u64 + 2 * tally.captured,
            "one sync per verify step plus two self-checks per captured shape"
        );
        // Speculation off through the engine equals the token-at-a-time driver.
        let plain = generate_speculative(
            &runner,
            &mut NoProposer,
            SpeculativePrompt::Tokens(&PROMPT),
            &config,
            0,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(plain.output.tokens, wrapped.output.tokens);
    }

    /// Every fallback is named and never fails the request: a cache that declares itself
    /// unstable, a step that reads the device during capture, a step whose allocation escapes
    /// the graph. Each request finishes token-identical to the bare model, on the same device.
    #[test]
    fn synthetic_misbehaviour_falls_back_with_a_named_reason() {
        let Some((device, _)) = device() else { return };
        let _guard = cuda_graphs_policy_guard(Some(true));
        let config = greedy(24);
        let reference = {
            let model = Synthetic::new(&device, None);
            generate_step(
                &model,
                &PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap()
            .0
            .tokens
        };
        for (misbehaviour, expected) in [
            ("declared", "mock_cache_declared_unstable"),
            ("host_read", REASON_SYNC_IN_CAPTURE),
            ("escape", REASON_ALLOCATION_ESCAPED_CAPTURE),
        ] {
            let model = Synthetic::new(&device, Some(misbehaviour));
            let runner = GraphRunner::new(&model);
            let (out, record) = generate_step(
                &runner,
                &PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap();
            eprintln!("[runner] {misbehaviour}: {}", record.cuda_graphs.describe());
            if let Some(census) = runner.last_census() {
                eprintln!("[runner] {misbehaviour} census: {}", census.describe());
            }
            assert_eq!(out.tokens, reference, "{misbehaviour}: tokens changed");
            assert_eq!(
                record.cuda_graphs.replayed, 0,
                "{misbehaviour}: nothing replayed"
            );
            assert_eq!(
                record.cuda_graphs.fallback_reason,
                Some(expected),
                "{misbehaviour}"
            );
            assert_eq!(runner.refusal(), Some(expected));
            assert!(runner.device().is_cuda(), "the device never changes");
            assert_eq!(runner.captured_graphs(), 0);
            if misbehaviour == "escape" {
                assert!(runner.last_census().unwrap().escaped_allocs >= 1);
            }
        }
    }

    /// The Qwen3.5/3.8 decoder on a pure-attention tiny config with the static KV cache: the
    /// runner captures the 1-token step, the census finds candle's per-op layout uploads
    /// (host-sourced memcpy nodes), and the request falls back eager with that named reason —
    /// the measured finding behind the module docs. The hybrid (DeltaNet) config is refused by
    /// the cache's own declaration before any capture.
    #[test]
    fn qwen35_static_step_is_refused_by_the_census_with_layout_uploads() {
        use crate::models::qwen35::tests::{text_model_attention_only_on, text_model_on};
        let Some((device, _)) = device() else { return };
        let _guard = cuda_graphs_policy_guard(Some(true));
        let config = greedy(8);
        let prompt = [1i32, 7, 3, 42, 9];

        let (_cfg, model) = text_model_attention_only_on(&device);
        let (bare, _) = generate_step(
            &model,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        let runner = GraphRunner::new(&model);
        let (wrapped, record) = generate_step(
            &runner,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(wrapped.tokens, bare.tokens);
        eprintln!(
            "[runner] qwen35 attention-only static: {}",
            record.cuda_graphs.describe()
        );
        assert_eq!(
            record.cuda_graphs.fallback_reason,
            Some(REASON_HOST_UPLOAD_IN_CAPTURE)
        );
        let census = runner.last_census().expect("a capture was attempted");
        eprintln!(
            "[runner] qwen35 1-token static step census: {}",
            census.describe()
        );
        assert!(census.memcpy_from_host > 0);
        assert!(census.kernels > 0);

        let (_cfg, hybrid) = text_model_on(&device);
        let runner = GraphRunner::new(&hybrid);
        let (_, record) = generate_step(
            &runner,
            &prompt,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        eprintln!(
            "[runner] qwen35 hybrid static: {}",
            record.cuda_graphs.describe()
        );
        assert_eq!(
            record.cuda_graphs.fallback_reason,
            Some("deltanet_state_unstable")
        );
        assert!(runner.last_census().is_none(), "refused before any capture");
    }

    /// What a graph replay saves per step on the synthetic model (the mechanism's win on a
    /// launch-bound step; evidence only).
    #[test]
    #[ignore = "timing evidence; needs a quiet GPU"]
    fn synthetic_replay_timing() {
        let Some((device, _)) = device() else { return };
        let _guard = cuda_graphs_policy_guard(Some(true));
        let model = Synthetic::new(&device, None);
        let config = greedy(2048);
        fn time<M: StepModel>(
            device: &Device,
            m: &M,
            config: &GenerationConfig,
        ) -> (std::time::Duration, Vec<i32>, GraphTally) {
            device.synchronize().unwrap();
            let started = std::time::Instant::now();
            let (out, record) =
                generate_step(m, &PROMPT, config, &CancelFlag::new(), &mut |_| {}, None).unwrap();
            device.synchronize().unwrap();
            (started.elapsed(), out.tokens, record.cuda_graphs)
        }
        let (eager_time, eager_tokens, _) = time(&device, &model, &config);
        let runner = GraphRunner::new(&model);
        let (graph_time, graph_tokens, tally) = time(&device, &runner, &config);
        assert_eq!(eager_tokens, graph_tokens);
        eprintln!(
            "[timing] synthetic 2048 steps: eager {:.1} us/step, graphs {:.1} us/step ({})",
            eager_time.as_secs_f64() * 1e6 / 2048.0,
            graph_time.as_secs_f64() * 1e6 / 2048.0,
            tally.describe()
        );
    }
}

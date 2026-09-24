//! The shared CUDA-graph runner (epic sc-24128, story sc-24134).
//!
//! **One** graph runner in `decode/`, written over the [`StepModel`] seam and a stable-address
//! [`DecodeCache`]: [`GraphRunner`] wraps any step model, captures one CUDA graph per distinct
//! step shape (the token count `M` — `1` for a decode step, `K + 1` for a speculative verify —
//! plus the logits scope and whether hidden states are wanted) and replays it at every later
//! step of that shape. Model files carry no graph logic. A captured step is replayable at a new
//! position only when every per-step position its kernels read is **device data** (staged by
//! [`DecodeCache::stage_positions`]) and its state lives at **stable addresses** (a static KV
//! cache, and — since sc-24131 — the per-token DeltaNet checkpoint ring, whose live state is a view
//! of a preallocated slot written in place). No Qwen3.5/3.8 step meets that on this revision —
//! `Qwen35Model` keeps its positions as Rust-side scalars and declares so
//! ([`StepModel::graph_support`] → `positions_host_scalar`); a ring-less hybrid cache still
//! replaces its DeltaNet state per step (`deltanet_state_unstable`) — so the runner refuses those
//! steps by name before any capture. The synthetic step model in the CUDA tests below meets it,
//! and is what proves the runner end to end.
//!
//! ## What a captured step is
//! Stream capture records every launch, memcpy and stream-ordered allocation the step issues
//! without executing them; the instantiated graph replays them as one launch. candle at
//! `1e6aa85e` allocates every temporary through cudarc 0.19's `CudaStream::alloc`, which is
//! `cuMemAllocAsync` on a device with memory-pool support (every Blackwell part), so the step's
//! temporaries become graph memory nodes — allocated and freed inside the graph — and the
//! runner needs no pre-planned workspace. The stream must be one candle created with
//! `Device::new_cuda_with_stream`: the legacy NULL stream `Device::new_cuda` uses cannot be
//! captured, and [`select_device`](crate::device::select_device) only builds the model's own
//! stream when this runner is switched on at load time (or `CANDLE_LLM_CUDA_STREAM=own`). The
//! step's outputs are copied into preallocated staging tensors inside the capture so nothing
//! the step allocated outlives it.
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
//! ([`REASON_HOST_UPLOAD_IN_CAPTURE`]). A Qwen3.5/3.8 step recorded with the evidence-only
//! `census_step` shows exactly that on this candle revision (the runner itself never records
//! one: the declarations above refuse it first). What would fix it in candle: pass layouts by
//! value as kernel parameters (or cache the `[dims, strides]` device buffers per layout) so no
//! per-op host upload exists. Until then only a step built from contiguous-only ops, cuBLAS
//! matmuls, `copy2d` (`slice_set`) and the nvrtc-seam kernels (scalar arguments) is replayable.
//!
//! ## Verification before trust
//! A graph is only ever used after two bit-exact self-checks, at the capture step and at the
//! first replay (a *new* position): each runs the eager step, rolls the cache back, launches
//! the graph at the same position, compares every logit (and hidden) element on the device,
//! then rolls back and re-runs the eager step so the cache holds the eager state. Any
//! difference — a stale scalar, a reallocated temporary, an unstable state address — throws
//! the graph away, keeps the eager result and names the reason. The reference path is never
//! changed by the runner (E2).
//!
//! ## Fallback, never failure
//! Every refusal is a **named** reason on the per-thread [`GraphTally`], reported per request
//! through [`DecodeRecord::cuda_graphs`](crate::decode::DecodeRecord::cuda_graphs) as
//! `graph: … fallback=<reason>`: the switch is off, the build has no `cuda` feature or uses
//! `flash-attn`, the model is not on a CUDA device or is on the legacy stream, the device has
//! no stream-ordered allocator, the cache or the model declares itself uncapturable
//! ([`DecodeCache::graph_support`], then [`StepModel::graph_support`]), the step shape is not
//! captured, the staging failed, the census found a host upload / host read / an allocation
//! that outlives the graph, the driver invalidated the capture, instantiation or a launch
//! failed, or a self-check could not run or disagreed with eager. A refusal drops every graph
//! the runner holds; the step rolls the cache back and runs eager, and the request continues
//! on the same device. The one case that fails the step is a cache that cannot roll back to a
//! position it restored or checkpointed earlier in the same step (`rollback_unavailable`):
//! then neither the graph's state nor the eager one can be trusted.
//!
//! ## Switch and accounting
//! `CANDLE_LLM_CUDA_GRAPHS` (`1` / `on` / `true` / `yes` enable; the default is **off** —
//! opt-in until a capturable step model exists and the decode bench shows a win, see the
//! story's evidence) or [`set_cuda_graphs`] at runtime; a load's own `LoadSpec::cuda_graphs`
//! overrides both for that model — its load (the stream it lands on) and every generation on it
//! ([`cuda_graphs_scope`], sc-24139). Admission prices graph memory up front with
//! [`graph_workspace_admission_bytes`] (E6); [`GraphRunner::workspace`] reports what a runner
//! holds (telemetry). A destroyed graph's memory is trimmed back from the device's graph pool.
//!
//! Without the `cuda` feature the runner is a transparent pass-through that reports
//! [`REASON_CUDA_FEATURE_OFF`].

use std::cell::{Cell, RefCell};

use candle_core::Device;
#[cfg(feature = "cuda")]
use candle_core::Tensor;

#[cfg(feature = "cuda")]
use crate::decode::step::StepTokens;
use crate::decode::step::{LogitsScope, StepModel, StepOutput, StepRequest};
use crate::error::Result;
use crate::primitives::attention::AttnFormulation;
use crate::primitives::decode_cache::DecodeCache;
use crate::primitives::switch::{ProcessSwitch, SwitchGuard};

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
/// Fallback reason: the model runs on the legacy NULL stream (`Device::new_cuda`), which stream
/// capture does not support — the device was selected while the switch was off, or with
/// `CANDLE_LLM_CUDA_STREAM=legacy`.
pub const REASON_LEGACY_STREAM: &str = "legacy_stream";
/// Fallback reason: a `flash-attn` build. candle-flash-attn launches its kernels on stream 0,
/// which cannot be captured and does not order against the model's own stream, so such a build
/// keeps the legacy stream and never captures.
pub const REASON_FLASH_ATTN_STREAM: &str = "flash_attn_stream";
/// Fallback reason: the step shape is not capturable (empty, or more tokens than
/// [`GraphRunner::MAX_CAPTURED_TOKENS`] — a prefill is never captured).
pub const REASON_SHAPE: &str = "shape";
/// Fallback reason: the cache cannot roll back to the step start, so the self-check cannot run.
pub const REASON_ROLLBACK_UNAVAILABLE: &str = "rollback_unavailable";
/// Fallback reason: the runner could not prepare or read back a graph step — allocating or
/// filling the staging tensors (tokens, positions), the cache's replay bookkeeping, or copying
/// the staged outputs out.
pub const REASON_STAGING_FAILED: &str = "staging_failed";
/// Fallback reason: a bit-exact self-check could not run (comparing the outputs failed).
pub const REASON_SELF_CHECK_FAILED: &str = "self_check_failed";
/// Why a request's record shows no runner step with the switch on: it decoded on a reference
/// path the runner does not wrap (it serves the step-seam engine only).
pub const REASON_REFERENCE_PATH: &str = "reference_path";
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

fn env_value_enables(v: &str) -> bool {
    matches!(v, "1" | "on" | "true" | "yes")
}

/// The process CUDA-graph switch (off unless the environment turns it on), on the crate's one
/// switch implementation, [`ProcessSwitch`].
static SWITCH: ProcessSwitch = ProcessSwitch::new(CUDA_GRAPHS_ENV, false, env_value_enables);

thread_local! {
    /// The thread-scoped switch ([`cuda_graphs_scope`]); `None` defers to the process switch.
    static REQUEST_POLICY: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Whether the graph runner may capture at all (the switch; a `cuda` build on a CUDA device is
/// still required for a graph to exist). A loaded model's own policy — `LoadSpec::cuda_graphs`,
/// scoped over its load and each of its generations by [`cuda_graphs_scope`] — wins over the
/// process switch ([`set_cuda_graphs`] / [`CUDA_GRAPHS_ENV`]).
pub fn cuda_graphs_enabled() -> bool {
    if let Some(enabled) = REQUEST_POLICY.with(Cell::get) {
        return enabled;
    }
    SWITCH.enabled()
}

/// Override the switch for the process: `Some(true)` / `Some(false)` force it, `None` returns to
/// the environment's setting.
pub fn set_cuda_graphs(enabled: Option<bool>) {
    SWITCH.set(enabled);
}

/// Restores the calling thread's previous scoped switch when dropped. Returned by
/// [`cuda_graphs_scope`].
#[must_use = "the graph policy only applies while the scope is alive"]
pub struct CudaGraphsScope {
    previous: Option<bool>,
}

impl Drop for CudaGraphsScope {
    fn drop(&mut self) {
        REQUEST_POLICY.with(|p| p.set(self.previous));
    }
}

/// Apply a loaded model's graph policy (`LoadSpec::cuda_graphs`, sc-24139) on the current
/// thread until the returned scope drops: `Some(true)` / `Some(false)` force the switch on / off
/// for everything done on this thread meanwhile — the device selection at load, and per request
/// admission's workspace pricing, the choice to wrap the step model and the runner's own switch
/// check — and `None` keeps the process switch. A load and its decodes run on the calling thread
/// (the tallies are thread-local for the same reason), so work on another thread is unaffected.
pub fn cuda_graphs_scope(enabled: Option<bool>) -> CudaGraphsScope {
    let previous = REQUEST_POLICY.with(|p| p.replace(enabled));
    CudaGraphsScope { previous }
}

/// Holds the process-wide switch lock; restores the switch it found when dropped. Returned by
/// [`cuda_graphs_policy_guard`].
#[doc(hidden)]
pub type CudaGraphsPolicyGuard = SwitchGuard;

/// Test seam: take the process-wide switch lock, apply `enabled` (as [`set_cuda_graphs`]) and
/// hand back a guard that restores the previous switch when dropped. Every test that flips the
/// switch — or asserts a reason the switch could change — holds one.
#[doc(hidden)]
pub fn cuda_graphs_policy_guard(enabled: Option<bool>) -> CudaGraphsPolicyGuard {
    SWITCH.guard(enabled)
}

/// Per-thread counts of graph replays vs eager step executions (monotone; take deltas with
/// [`GraphTally::since`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphTally {
    /// Graph launches that ran a step: every replay, the first-replay self-check included.
    pub replayed: u64,
    /// Eager executions of a step through the runner: fallbacks, warm-ups, prefills, and each
    /// self-check's eager reference and eager re-run.
    pub eager: u64,
    /// Graphs that passed both self-checks (at most one per step shape).
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

/// The device memory a runner holds for its graphs (telemetry; admission prices graph memory
/// with [`graph_workspace_admission_bytes`], E6): the staging tensors it allocated and the growth
/// of the device's graph-memory reservation across its captures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GraphWorkspace {
    /// Bytes of the preallocated input / output staging tensors, every captured shape.
    pub staging_bytes: usize,
    /// How much the device's graph-memory reservation (`CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT`)
    /// grew across this runner's captures. The pool is device-wide and a graph reuses what is
    /// already reserved, so this is an **approximation**: it under-reports (down to `0`) when
    /// another live graph's reservation already covers the step, and it includes whatever
    /// another thread reserved during the capture. Destroyed graphs are trimmed from the pool
    /// (`cuDeviceGraphMemTrim`), so a runner that drops its graphs gives the reservation back.
    pub graph_reserved_bytes: usize,
}

impl GraphWorkspace {
    /// `staging_bytes + graph_reserved_bytes` (saturating).
    pub fn total_bytes(&self) -> usize {
        self.staging_bytes.saturating_add(self.graph_reserved_bytes)
    }
}

/// Bytes admission charges for the CUDA graphs one request may hold (E6, story sc-24134), on the
/// same basis the request estimator prices a step (`core_llm::estimate_chunked_request_bytes`).
///
/// A graph's memory nodes are served from a device-wide graph pool the driver keeps reserved while
/// the graph lives, *beside* the stream-ordered pool the eager step uses, so each captured shape
/// costs its own copy of one step's working set: the projections, MLP intermediates and residuals
/// (`M · (3·intermediate + 8·hidden + vocab)`), the attention scores / weights over every position
/// the request can reach (`3 · M · query_heads · total_positions`), plus the runner's staging
/// tensors (ids, all-position logits, hidden rows). The speculative engine can present
/// `max_step_tokens` token counts (`1 ..= K + 1`: the verify and every replay length) in two logits
/// scopes each, so every one is priced. `None` on overflow (the caller fails closed).
pub fn graph_workspace_admission_bytes(
    geometry: &core_llm::LlmMemoryGeometry,
    total_positions: u64,
    max_step_tokens: u32,
) -> Option<u64> {
    let e = geometry.element_bytes;
    let per_token = geometry
        .intermediate_size
        .checked_mul(3)?
        .checked_add(geometry.hidden_size.checked_mul(8)?)?
        .checked_add(geometry.vocab_size)?
        .checked_mul(e)?
        .checked_add(
            geometry
                .query_heads
                .checked_mul(total_positions)?
                .checked_mul(e)?
                .checked_mul(3)?,
        )?
        // Staging: the id, a logits row and a hidden row per token.
        .checked_add(4)?
        .checked_add(
            geometry
                .vocab_size
                .checked_add(geometry.hidden_size)?
                .checked_mul(e)?,
        )?;
    // Σ_{M=1}^{N} M = N (N + 1) / 2 token-rows, in two scopes.
    let n = u64::from(max_step_tokens);
    let rows = n.checked_mul(n.checked_add(1)?)? / 2;
    per_token.checked_mul(rows)?.checked_mul(2)
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
            if cfg!(feature = "flash-attn") {
                return Some(REASON_FLASH_ATTN_STREAM);
            }
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
            // The cache's declaration first: it names the state a cache change lifts (a
            // ring-less DeltaNet cache's replaced state, a growing KV), the gate ahead of the
            // model's own (positions).
            if let Err(reason) = cache.graph_support() {
                return Some(reason);
            }
            if let Err(reason) = self.model.graph_support() {
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

/// **Evidence only** (sc-24134), not serving API: record one step of `model` on `cache` as a
/// CUDA graph, take its [`GraphCensus`], destroy the graph **without instantiating or launching
/// it**, and roll the cache back to where it was. Nothing runs on the device, and the declared
/// refusals ([`StepModel::graph_support`], [`DecodeCache::graph_support`]) are deliberately not
/// consulted: this is how a real decoder step's launch count and host-upload count are
/// measured. `None` when the recording itself failed (the reason is logged) — a step that frees
/// a tensor it did not allocate inside the capture does that.
///
/// **Precondition: discard `cache` (and do not decode with `model` on it) after this call.** The
/// recording replaces any state the step re-creates with graph-owned tensors that were never
/// backed by memory; the rollback drops the ones it restores, but a cache that keeps any other
/// such tensor holds an address nothing backs, and the driver errors from freeing them are only
/// drained here. Call it at the end of an evidence run on a model and cache you are done with.
/// The model's device must be on its own stream (the graph runner on at load time).
#[doc(hidden)]
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
    /// `cuGraphInstantiateWithFlags(0)`, explicit upload, executable destroyed before the graph,
    /// and the device's graph-memory pool trimmed once both are gone.)
    pub(super) struct Graph {
        graph: sys::CUgraph,
        exec: sys::CUgraphExec,
        stream: Arc<CudaStream>,
    }

    #[cfg(test)]
    thread_local! {
        /// Test seam: fail this thread's next graph launch (the `launch_failed` fallback test).
        pub(super) static FAIL_NEXT_LAUNCH: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    impl Graph {
        /// Replay the graph on its stream (asynchronous, stream-ordered like any launch).
        pub(super) fn launch(&self) -> std::result::Result<(), sys::CUresult> {
            #[cfg(test)]
            if FAIL_NEXT_LAUNCH.with(|f| f.replace(false)) {
                return Err(sys::CUresult::CUDA_ERROR_LAUNCH_FAILED);
            }
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
            // Graph memory stays reserved in the device-wide pool after its graph is gone until
            // it is trimmed; give it back (only memory no live, scheduled graph uses is freed).
            unsafe { sys::cuDeviceGraphMemTrim(self.stream.context().cu_device()) };
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
        /// Drop every graph and staging tensor (each graph trims the pool as it goes).
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

    pub(super) fn reserved_graph_mem(dev: &CudaDevice) -> Option<u64> {
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

    /// An open stream capture on this thread. [`end`](Self::end) ends it; dropping it without
    /// that — the step panicked — ends it too and destroys whatever was recorded, so the stream
    /// never stays in capture mode. The capture flag is cleared after the capture ends.
    struct CaptureSession {
        stream: Arc<CudaStream>,
        open: bool,
        _flag: CaptureFlag,
    }

    impl CaptureSession {
        fn begin(stream: &Arc<CudaStream>) -> std::result::Result<Self, Abandoned> {
            if let Err(e) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            {
                eprintln!("[cuda-graph] begin_capture failed: {e:?}");
                return Err(Abandoned(REASON_CAPTURE_INVALIDATED));
            }
            Ok(Self {
                stream: stream.clone(),
                open: true,
                _flag: CaptureFlag::set(),
            })
        }

        /// `cuStreamEndCapture`: its result and the recorded graph (null when invalidated).
        fn end(mut self) -> (sys::CUresult, sys::CUgraph) {
            self.open = false;
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            let ended = unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
            (ended, graph)
        }
    }

    impl Drop for CaptureSession {
        fn drop(&mut self) {
            if !self.open {
                return;
            }
            let ctx = self.stream.context();
            if ctx.bind_to_thread().is_err() {
                return;
            }
            let mut graph: sys::CUgraph = std::ptr::null_mut();
            unsafe { sys::cuStreamEndCapture(self.stream.cu_stream(), &mut graph) };
            if !graph.is_null() {
                unsafe { sys::cuGraphDestroy(graph) };
            }
            // Drain what the interrupted step left recorded on cudarc's context.
            let _ = ctx.check_err();
        }
    }

    /// Capture `f` on the device's stream: begin capture, run the closure with the capture flag
    /// set, end capture. The closure's device work is **recorded, not run**. Returns the recorded
    /// graph, or the reason capture was abandoned (the capture is always ended — also when `f`
    /// panics — and cudarc's recorded error state drained, so the stream is usable afterwards).
    pub(super) fn capture<T>(
        dev: &CudaDevice,
        f: impl FnOnce() -> Result<T>,
    ) -> std::result::Result<(Recorded, T), Abandoned> {
        let stream = dev.cuda_stream();
        let ctx = stream.context();
        let session = CaptureSession::begin(&stream)?;
        let syncs_before = host_sync_count();
        let outcome = f();
        // `cuStreamEndCapture` is called whatever happened: on a failed step it returns the
        // driver's invalidation error and leaves the stream in its normal state. A failing free
        // inside the capture is recorded on cudarc's context and would resurface on the next
        // call, so it is drained here.
        let (ended, graph) = session.end();
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

    /// Runner-wide refusal: record the reason and drop **every** graph and staging tensor the
    /// runner holds (a refused runner never replays again, so nothing it captured is kept).
    fn refuse<M: StepModel>(runner: &GraphRunner<'_, M>, reason: &'static str) {
        let mut inner = runner.inner.borrow_mut();
        inner.cuda.clear();
        inner.refused = Some(reason);
    }

    /// The end of every failed graph attempt: refuse with `reason`, take the cache back to `base`
    /// (the attempt may have advanced it — a recording, a launch, the replay bookkeeping) and run
    /// the eager step there. When the cache cannot return to `base` — a position it restored or
    /// checkpointed earlier in this same step — neither the graph's state nor the eager one can
    /// be trusted: the step fails with the cache's error (`rollback_unavailable`).
    fn fall_back<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        request: StepRequest<'_>,
        base: i32,
        reason: &'static str,
    ) -> Result<StepOutput> {
        refuse(runner, reason);
        restore(runner, cache, base, reason)?;
        // A rollback after a recording drops the graph-owned tensors the step made, whose
        // addresses nothing ever backed; cudarc records the failed frees on the context, where
        // they would fail the eager step's first op. Drain them (as `census_step` does).
        if let Device::Cuda(dev) = runner.model.device() {
            if let Err(e) = dev.cuda_stream().context().check_err() {
                eprintln!("[cuda-graph] drained a recorded driver error after `{reason}`: {e:?}");
            }
        }
        let out = runner.model.forward_step(cache, request)?;
        note_eager(Some(reason));
        Ok(out)
    }

    /// Roll the cache back to `base` after the runner advanced it (see [`fall_back`]); `after`
    /// names what happened, for the log.
    fn restore<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        base: i32,
        after: &str,
    ) -> Result<()> {
        cache.rollback_to(base).map_err(|e| {
            eprintln!(
                "[cuda-graph] cannot roll back to {base} after {after} ({REASON_ROLLBACK_UNAVAILABLE}): {e}"
            );
            refuse(runner, REASON_ROLLBACK_UNAVAILABLE);
            e
        })
    }

    /// The eager reference of a self-check: run the eager step at `base`, then roll the cache
    /// back to `base` for the graph to run the same position. `Err(out)` when the cache cannot
    /// roll back: the runner is refused and `out` — the eager step, which the cache holds — is
    /// the step's answer.
    fn eager_reference<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        request: StepRequest<'_>,
        base: i32,
    ) -> Result<std::result::Result<StepOutput, StepOutput>> {
        let eager = runner.model.forward_step(cache, request)?;
        if let Err(e) = cache.rollback_to(base) {
            eprintln!("[cuda-graph] cannot roll back for the self-check: {e}");
            refuse(runner, REASON_ROLLBACK_UNAVAILABLE);
            note_eager(Some(REASON_ROLLBACK_UNAVAILABLE));
            return Ok(Err(eager));
        }
        note_eager(None);
        Ok(Ok(eager))
    }

    /// Compare the graph's outputs with the eager reference: `None` when bit-identical, the
    /// fallback reason otherwise.
    fn self_check(graph: &StepOutput, eager: &StepOutput, what: &str) -> Option<&'static str> {
        match outputs_identical(graph, eager) {
            Ok(true) => None,
            Ok(false) => {
                eprintln!("[cuda-graph] {what} disagreed with eager ({REASON_REPLAY_MISMATCH})");
                Some(REASON_REPLAY_MISMATCH)
            }
            Err(e) => {
                eprintln!(
                    "[cuda-graph] {what}: the self-check failed ({REASON_SELF_CHECK_FAILED}): {e}"
                );
                Some(REASON_SELF_CHECK_FAILED)
            }
        }
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
        let base = cache.len();
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
                // The eager reference first: the warm-up proved the step runs, its outputs give
                // the staging tensors their shapes, and it is what the caller gets either way.
                let eager = match eager_reference(runner, cache, request, base)? {
                    Ok(eager) => eager,
                    Err(eager) => return Ok(eager),
                };
                // Staging, outside the capture: the id / output tensors the graph reads and
                // writes, the step's tokens and positions. Nothing here advances the cache.
                let staged = (|| -> Result<(Tensor, Tensor, Option<Tensor>)> {
                    let ids = Tensor::zeros((1, key.tokens), DType::U32, &device)?;
                    let logits = eager.logits.zeros_like()?;
                    let hidden = match &eager.hidden {
                        Some(h) => Some(h.zeros_like()?),
                        None => None,
                    };
                    stage_tokens(&ids, &request.tokens, &device)?;
                    cache.stage_positions()?;
                    Ok((ids, logits, hidden))
                })();
                let (ids, logits, hidden) = match staged {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("[cuda-graph] staging the capture failed: {e}");
                        return fall_back(runner, cache, request, base, REASON_STAGING_FAILED);
                    }
                };
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
                // From here every failure falls back through `fall_back`, which takes the cache
                // back to the step start: the recording advanced its Rust-side state (or part of
                // it, on a failed step), and a launch wrote its device state.
                let recorded = match captured {
                    Ok((recorded, ())) => recorded,
                    Err(Abandoned(reason)) => {
                        return fall_back(runner, cache, request, base, reason)
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
                    return fall_back(runner, cache, request, base, reason);
                }
                let graph = match recorded.instantiate() {
                    Ok(g) => g,
                    Err(Abandoned(reason)) => {
                        return fall_back(runner, cache, request, base, reason)
                    }
                };
                if let Err(e) = graph.launch() {
                    eprintln!("[cuda-graph] first launch failed: {e:?}");
                    drop(graph);
                    return fall_back(runner, cache, request, base, REASON_LAUNCH_FAILED);
                }
                let launched = StepOutput {
                    logits: logits.clone(),
                    hidden: hidden.clone(),
                };
                let what = format!("the capture check of the {}-token step", key.tokens);
                if let Some(reason) = self_check(&launched, &eager, &what) {
                    drop(graph);
                    return fall_back(runner, cache, request, base, reason);
                }
                // Verified: leave the eager step's state in the cache (the launch wrote the same
                // positions) and keep the graph for its first replay.
                restore(runner, cache, base, "the capture check")?;
                let eager = model.forward_step(cache, request)?;
                note_eager(None);
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
                // step's own check (same inputs, same position) cannot. Eager first: until the
                // replay runs, a cache that cannot roll back still holds the eager step.
                let eager = match eager_reference(runner, cache, request, base)? {
                    Ok(eager) => eager,
                    Err(eager) => return Ok(eager),
                };
                let replayed = match replay(runner, cache, &request, key) {
                    Ok(out) => out,
                    Err(reason) => return fall_back(runner, cache, request, base, reason),
                };
                let what = format!("the first replay of the {}-token step", key.tokens);
                if let Some(reason) = self_check(&replayed, &eager, &what) {
                    return fall_back(runner, cache, request, base, reason);
                }
                // Verified: leave the eager step's state in the cache — a replay that wrote a
                // stale position can still match eager's outputs this once.
                restore(runner, cache, base, "the first-replay check")?;
                let eager = model.forward_step(cache, request)?;
                note_eager(None);
                note_captured();
                if let Some(Shape::Graph(c)) = runner.inner.borrow_mut().cuda.shapes.get_mut(&key) {
                    c.replays = 1;
                }
                Ok(eager)
            }
            Plan::Replay => match replay(runner, cache, &request, key) {
                Ok(out) => {
                    if let Some(Shape::Graph(c)) =
                        runner.inner.borrow_mut().cuda.shapes.get_mut(&key)
                    {
                        c.replays = c.replays.wrapping_add(1);
                    }
                    Ok(out)
                }
                Err(reason) => fall_back(runner, cache, request, base, reason),
            },
        }
    }

    /// One replay: stage the tokens and positions, do the cache's bookkeeping for the step,
    /// launch, and return copies of the staged outputs. `Err` names why not; the cache may have
    /// advanced (the caller falls back through [`fall_back`], which rolls it back).
    fn replay<M: StepModel>(
        runner: &GraphRunner<'_, M>,
        cache: &mut M::Cache,
        request: &StepRequest<'_>,
        key: ShapeKey,
    ) -> std::result::Result<StepOutput, &'static str> {
        let inner = runner.inner.borrow();
        let Some(Shape::Graph(c)) = inner.cuda.shapes.get(&key) else {
            eprintln!(
                "[cuda-graph] no graph to launch for the {}-token step",
                key.tokens
            );
            return Err(REASON_LAUNCH_FAILED);
        };
        let staged = stage_tokens(&c.ids, &request.tokens, runner.model.device())
            .and_then(|()| cache.stage_positions())
            .and_then(|()| cache.replay_advance(key.tokens));
        if let Err(e) = staged {
            eprintln!("[cuda-graph] staging the replay failed: {e}");
            return Err(REASON_STAGING_FAILED);
        }
        if let Err(e) = c.graph.launch() {
            eprintln!("[cuda-graph] launch failed: {e:?}");
            return Err(REASON_LAUNCH_FAILED);
        }
        match copy_outputs(&c.logits, c.hidden.as_ref()) {
            Ok(out) => {
                note_replayed();
                Ok(out)
            }
            Err(e) => {
                eprintln!("[cuda-graph] copying the replay's outputs failed: {e}");
                Err(REASON_STAGING_FAILED)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::step::StepOutput;
    use crate::primitives::decode_cache::CacheMemory;
    use candle_core::Tensor;
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
    fn a_scope_overrides_the_process_switch_on_its_thread_only() {
        let _guard = cuda_graphs_policy_guard(Some(false));
        {
            let _on = cuda_graphs_scope(Some(true));
            assert!(cuda_graphs_enabled(), "the scope's Some(true) wins");
            // Another thread sees the process switch, not this scope's policy.
            let elsewhere = std::thread::spawn(cuda_graphs_enabled).join().unwrap();
            assert!(!elsewhere);
            {
                let _inner = cuda_graphs_scope(Some(false));
                assert!(!cuda_graphs_enabled());
            }
            assert!(
                cuda_graphs_enabled(),
                "a nested scope restores the outer one"
            );
            // `None` defers to the process switch.
            let _defer = cuda_graphs_scope(None);
            assert!(!cuda_graphs_enabled());
        }
        assert!(!cuda_graphs_enabled(), "the scope is gone once dropped");
        set_cuda_graphs(Some(true));
        let _off = cuda_graphs_scope(Some(false));
        assert!(
            !cuda_graphs_enabled(),
            "Some(false) wins over a process switch that is on"
        );
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
    fn graph_admission_prices_every_step_shape_of_a_request() {
        let geometry = core_llm::LlmMemoryGeometry {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            layers: 4,
            element_bytes: 4,
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 50,
            recurrent_bytes: 0,
        };
        let per_token = (3 * 64 + 8 * 32 + 50) * 4 + 4 * 100 * 4 * 3 + 4 + (50 + 32) * 4;
        // A decode-only request (one 1-token shape, two scopes).
        assert_eq!(
            graph_workspace_admission_bytes(&geometry, 100, 1),
            Some(2 * per_token)
        );
        // K = 3: token counts 1..=4 → 10 token-rows per scope.
        assert_eq!(
            graph_workspace_admission_bytes(&geometry, 100, 4),
            Some(2 * 10 * per_token)
        );
        assert_eq!(graph_workspace_admission_bytes(&geometry, 100, 0), Some(0));
        let huge = core_llm::LlmMemoryGeometry {
            vocab_size: u64::MAX,
            ..geometry
        };
        assert_eq!(graph_workspace_admission_bytes(&huge, 100, 4), None);
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
    use crate::primitives::decode_cache::CacheMemory;
    use crate::primitives::rope::rms_norm_rope;
    use candle_core::cuda_backend::cudarc::driver::sys;
    use candle_core::{DType, Device};
    use core_llm::ProposerKind;
    use std::cell::Cell;

    /// A capturable device the way a graphs-on deployment gets one: the switch on when
    /// `select_device` runs, so the model gets its own stream with cudarc's event tracking off.
    /// The returned guard keeps the switch on — and serializes the tests that capture — for
    /// the whole test; bind it first so it is dropped last.
    fn device() -> Option<(CudaGraphsPolicyGuard, Device, candle_core::CudaDevice)> {
        let guard = cuda_graphs_policy_guard(Some(true));
        match crate::device::select_device() {
            Ok(Device::Cuda(d)) if !d.cuda_stream().cu_stream().is_null() => {
                Some((guard, Device::Cuda(d.clone()), d))
            }
            Ok(Device::Cuda(_)) => {
                eprintln!(
                    "skipping: the legacy stream was selected (CANDLE_LLM_CUDA_STREAM, flash-attn)"
                );
                None
            }
            _ => {
                eprintln!("skipping: no CUDA device");
                None
            }
        }
    }

    /// The device's graph-memory reservation now.
    fn reserved(dev: &candle_core::CudaDevice) -> u64 {
        super::cuda::reserved_graph_mem(dev).expect("the graph-memory attribute")
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

    /// The device `select_device` builds with the switch on is refused by name when it cannot
    /// be captured: in a `flash-attn` build (always the legacy stream) as `flash_attn_stream`,
    /// otherwise not at all at the capability check (the own stream).
    #[test]
    fn a_flash_attn_build_is_refused_by_name() {
        let _guard = cuda_graphs_policy_guard(Some(true));
        let Ok(device @ Device::Cuda(_)) = crate::device::select_device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let model = Synthetic::new(&device, None);
        let runner = GraphRunner::new(&model);
        let (_, record) = generate_step(
            &runner,
            &PROMPT,
            &greedy(4),
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        eprintln!(
            "[runner] select_device, switch on: {}",
            record.cuda_graphs.describe()
        );
        if cfg!(feature = "flash-attn") {
            assert_eq!(
                record.cuda_graphs.fallback_reason,
                Some(REASON_FLASH_ATTN_STREAM)
            );
            assert_eq!(runner.captured_graphs(), 0);
        } else {
            assert_eq!(record.cuda_graphs.fallback_reason, None);
            assert_eq!(runner.captured_graphs(), 1);
        }
    }

    // ---- POC experiments ----

    #[test]
    fn poc_allocator_is_stream_ordered_and_the_stream_is_capturable() {
        let Some((_guard, _, dev)) = device() else {
            return;
        };
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
        let Some((_guard, device, dev)) = device() else {
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
        let Some((_guard, device, dev)) = device() else {
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

    /// Op-by-op capture diagnostic (evidence only): captures one candle op at a time — `POC_OP`
    /// names one of `matmul|softmax|index_select|rope|affine|slice_set|transpose|broadcast_add|
    /// sum`, unset runs them all — reports its census, and replays the ones the census accepts
    /// (bit-exact against eager).
    #[test]
    #[ignore = "evidence only: the per-op census behind the module docs (POC_OP selects one op)"]
    fn poc_single_op_census() {
        let Some((_guard, device, dev)) = device() else {
            return;
        };
        let ops: Vec<String> = match std::env::var("POC_OP") {
            Ok(op) => vec![op],
            Err(_) => [
                "matmul",
                "softmax",
                "index_select",
                "rope",
                "affine",
                "slice_set",
                "transpose",
                "broadcast_add",
                "sum",
            ]
            .map(String::from)
            .to_vec(),
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
        for op in &ops {
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
                continue;
            }
            let graph = recorded
                .instantiate()
                .unwrap_or_else(|Abandoned(reason)| panic!("instantiate: {reason}"));
            graph.launch().unwrap();
            let identical = bit_identical(&out_stage, &eager).unwrap();
            eprintln!("[poc] op {op}: replayed, bit-identical = {identical}");
            assert!(identical, "{op}");
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
        /// Longer than the runner captures, so the shape fallback is reachable.
        const MAX: usize = 32;

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
            // A Rust-side scalar folded into a kernel argument (what a model with host-side
            // positions does): the capture bakes in the position it was recorded at, so the
            // first replay at a new position disagrees with eager.
            let h = match self.misbehave.get() {
                Some("scalar") => h.affine(1.0, f64::from(cache.len) * 0.25)?,
                _ => h,
            };
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
        let Some((_guard, device, _)) = device() else {
            return;
        };
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
        // Prefill (5 tokens, eager) + warm-up + the capture step and the verified first replay
        // (each an eager reference and the eager re-run that leaves the cache in the eager
        // state): 6 eager executions; the verified replay and the other 36 decode steps replay.
        assert_eq!(tally.eager, 6);
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
        // eager re-run) and the verified replay's eager reference and re-run; replays never call
        // the model.
        let eager_calls = model.steps.get() - steps_before;
        assert_eq!(
            eager_calls,
            1 + 1 + 3 + 2,
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
        let Some((_guard, device, _)) = device() else {
            return;
        };
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
    /// the graph, a step that bakes a Rust-side scalar into the capture (caught by the first
    /// replay's self-check). Each request finishes token-identical to the same model run bare,
    /// on the same device, with no graph kept.
    #[test]
    fn synthetic_misbehaviour_falls_back_with_a_named_reason() {
        let Some((_guard, _, _)) = device() else {
            return;
        };
        let config = greedy(24);
        // (misbehaviour, reason, graph launches: the first-replay check's own replay)
        for (misbehaviour, expected, launches) in [
            ("declared", "mock_cache_declared_unstable", 0),
            ("host_read", REASON_SYNC_IN_CAPTURE, 0),
            ("escape", REASON_ALLOCATION_ESCAPED_CAPTURE, 0),
            ("scalar", REASON_REPLAY_MISMATCH, 1),
        ] {
            // A device per case: the `escape` mock cache keeps the tensor its recording made
            // outside any rollback, and freeing that never-backed address when the cache drops
            // leaves an error on the device's context for its next op.
            let device = crate::device::select_device().unwrap();
            let bare = generate_step(
                &Synthetic::new(&device, Some(misbehaviour)),
                &PROMPT,
                &config,
                &CancelFlag::new(),
                &mut |_| {},
                None,
            )
            .unwrap()
            .0
            .tokens;
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
            assert_eq!(out.tokens, bare, "{misbehaviour}: tokens changed");
            assert_eq!(
                record.cuda_graphs.replayed, launches,
                "{misbehaviour}: graph launches"
            );
            assert_eq!(
                record.cuda_graphs.captured, 0,
                "{misbehaviour}: no graph passed both checks"
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

    /// A step longer than the runner captures (a prefill) runs eager with the named reason
    /// `shape`; it is a per-step fallback, not a refusal, so later decode steps still capture.
    #[test]
    fn oversized_steps_are_named_shape_fallbacks() {
        let Some((_guard, device, _)) = device() else {
            return;
        };
        let model = Synthetic::new(&device, None);
        let runner = GraphRunner::new(&model);
        let mut cache = runner.new_cache_for(64, 0).unwrap();
        let long: Vec<i32> = (0..GraphRunner::<Synthetic>::MAX_CAPTURED_TOKENS as i32 + 1)
            .map(|i| i % 20)
            .collect();
        let start = graph_tally();
        let out = runner
            .forward_step(&mut cache, StepRequest::last(&long))
            .unwrap();
        let tally = graph_tally().since(&start);
        assert_eq!(out.logits.dims(), &[1, model.vocab]);
        assert_eq!(cache.len(), long.len() as i32);
        assert_eq!(tally.fallback_reason, Some(REASON_SHAPE));
        assert_eq!((tally.eager, tally.replayed), (1, 0));
        assert_eq!(
            runner.refusal(),
            None,
            "a shape fallback does not refuse the runner"
        );
        for t in [1, 2, 3, 4] {
            runner
                .forward_step(&mut cache, StepRequest::last(&[t]))
                .unwrap();
        }
        assert_eq!(runner.captured_graphs(), 1, "decode steps still capture");
    }

    /// A graph launch that fails mid-request: the replay's bookkeeping is rolled back, the step
    /// runs eager with the named reason `launch_failed`, the runner drops its graph, and every
    /// step's logits and the cache length stay identical to the bare model's.
    #[test]
    fn a_failed_launch_rolls_back_and_falls_back_eager() {
        let Some((_guard, device, _)) = device() else {
            return;
        };
        let model = Synthetic::new(&device, None);
        let runner = GraphRunner::new(&model);
        let mut cache = runner.new_cache_for(64, 0).unwrap();
        let mut bare = model.new_cache();
        let start = graph_tally();
        for (i, t) in [3i32, 7, 11, 2, 7, 5, 9, 4, 8, 6].into_iter().enumerate() {
            if i == 6 {
                // Steps 0–1 warm up and capture, 2 is the verified replay, 3–5 replay.
                assert_eq!(runner.captured_graphs(), 1);
                super::cuda::FAIL_NEXT_LAUNCH.with(|f| f.set(true));
            }
            let before = graph_tally();
            let graph = runner
                .forward_step(&mut cache, StepRequest::last(&[t]))
                .unwrap();
            if i == 6 {
                assert_eq!(
                    graph_tally().since(&before).fallback_reason,
                    Some(REASON_LAUNCH_FAILED),
                    "the failing step's reason"
                );
            }
            let eager = model
                .forward_step(&mut bare, StepRequest::last(&[t]))
                .unwrap();
            assert!(
                bit_identical(&graph.logits, &eager.logits).unwrap(),
                "step {i}: logits"
            );
            assert_eq!(cache.len(), bare.len(), "step {i}: cache length");
        }
        let tally = graph_tally().since(&start);
        eprintln!("[runner] failed launch: {}", tally.describe());
        assert_eq!(
            tally.replayed, 4,
            "the verified replay and three replays before the failure"
        );
        assert_eq!(runner.refusal(), Some(REASON_LAUNCH_FAILED));
        assert_eq!(runner.captured_graphs(), 0);
    }

    /// A runner-wide refusal drops every graph the runner holds, not only the refused shape's:
    /// a verified 1-token graph is released (with its staging) when the 2-token shape is refused.
    #[test]
    fn a_refusal_drops_every_captured_shape() {
        let Some((_guard, device, _)) = device() else {
            return;
        };
        let model = Synthetic::new(&device, None);
        let runner = GraphRunner::new(&model);
        let mut cache = runner.new_cache_for(64, 0).unwrap();
        for t in [3, 7, 11, 2] {
            runner
                .forward_step(&mut cache, StepRequest::last(&[t]))
                .unwrap();
        }
        assert_eq!(runner.captured_graphs(), 1);
        assert!(runner.workspace().staging_bytes > 0);
        // The 2-token shape keeps a fresh tensor from every step: its capture is refused.
        model.misbehave.set(Some("escape"));
        for pair in [[1, 2], [3, 4]] {
            runner
                .forward_step(&mut cache, StepRequest::last(&pair))
                .unwrap();
        }
        assert_eq!(runner.refusal(), Some(REASON_ALLOCATION_ESCAPED_CAPTURE));
        assert_eq!(
            runner.captured_graphs(),
            0,
            "the 1-token graph is dropped too"
        );
        assert_eq!(runner.workspace(), GraphWorkspace::default());
    }

    /// A step that panics inside a capture still ends it: the stream leaves capture mode, the
    /// capture flag is cleared, and the device keeps working.
    #[test]
    fn a_panic_inside_the_capture_still_ends_it() {
        let Some((_guard, device, dev)) = device() else {
            return;
        };
        let x = Tensor::new(&[1f32, 2.0, 3.0], &device).unwrap();
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = capture(&dev, || -> Result<()> {
                let _doubled = x.affine(2.0, 0.0)?;
                panic!("a step panics inside the capture");
            });
        }));
        assert!(panicked.is_err());
        assert!(!capturing(), "the capture flag is cleared");
        // A failed query leaves `ACTIVE`, which the assertion rejects.
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
        let _ = unsafe { sys::cuStreamIsCapturing(dev.cuda_stream().cu_stream(), &mut status) };
        assert_eq!(
            status,
            sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
            "the stream left capture mode"
        );
        // And the stream runs work again.
        x.affine(3.0, 0.0).unwrap().to_vec1::<f32>().unwrap();
    }

    /// Graph memory (E6): the runner reports the reservation its capture added to a trimmed
    /// pool, and gives it back — the pool is trimmed — when it drops its graphs.
    #[test]
    fn graph_memory_is_reported_and_trimmed_when_the_graphs_go() {
        let Some((_guard, device, dev)) = device() else {
            return;
        };
        // Earlier tests' graphs are gone (each trims as it goes); start from an empty pool.
        unsafe { sys::cuDeviceGraphMemTrim(dev.cuda_stream().context().cu_device()) };
        assert_eq!(reserved(&dev), 0, "no graph memory reserved at the start");
        let model = Synthetic::new(&device, None);
        let runner = GraphRunner::new(&model);
        generate_step(
            &runner,
            &PROMPT,
            &greedy(8),
            &CancelFlag::new(),
            &mut |_| {},
            None,
        )
        .unwrap();
        assert_eq!(runner.captured_graphs(), 1);
        let workspace = runner.workspace();
        eprintln!(
            "[runner] graph memory: {workspace:?}, device reserved {}",
            reserved(&dev)
        );
        assert!(
            workspace.graph_reserved_bytes > 0,
            "the capture's reservation is reported"
        );
        assert_eq!(workspace.graph_reserved_bytes as u64, reserved(&dev));
        runner.reset();
        assert_eq!(runner.workspace(), GraphWorkspace::default());
        assert_eq!(reserved(&dev), 0, "dropping the graphs trims the pool");
    }

    /// The Qwen3.5/3.8 decoder is refused by declaration before any capture: on the pure-attention
    /// and the hybrid tiny configs alike, on the engine's cache (static KV, and on the hybrid the
    /// per-token DeltaNet ring of sc-24131, whose state stays at stable addresses), by the model's
    /// Rust-scalar positions (`positions_host_scalar`). What a recording of its step would hold is
    /// measured with `census_step`: candle's per-op layout uploads (host-sourced memcpy nodes),
    /// and — on the hybrid — no allocation outliving the step (`escaped=0`: the ring is written
    /// in place, where the S1 cache replaced every linear layer's states).
    #[test]
    fn qwen35_steps_are_refused_by_declaration_and_the_census_finds_layout_uploads() {
        use crate::models::qwen35::tests::{text_model_attention_only_on, text_model_on};
        let Some((_guard, device, _)) = device() else {
            return;
        };
        let config = greedy(8);
        let prompt = [1i32, 7, 3, 42, 9];

        let (_cfg, model) = text_model_attention_only_on(&device);
        assert_eq!(model.graph_support(), Err("positions_host_scalar"));
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
            Some("positions_host_scalar")
        );
        assert!(runner.last_census().is_none(), "refused before any capture");

        // The measurement: record one warmed 1-token step (evidence only; the cache is
        // discarded afterwards).
        let mut cache = model.new_cache_for(prompt.len() + 8, 0).unwrap();
        model
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        model
            .forward_step(&mut cache, StepRequest::last(&[3]))
            .unwrap();
        let census = census_step(&model, &mut cache, StepRequest::last(&[5]))
            .unwrap()
            .expect("the step records");
        eprintln!(
            "[runner] qwen35 1-token static step census: {}",
            census.describe()
        );
        assert!(census.memcpy_from_host > 0);
        assert!(census.kernels > 0);
        assert_eq!(census.refusal(), Some(REASON_HOST_UPLOAD_IN_CAPTURE));
        drop(cache);

        let (_cfg, hybrid) = text_model_on(&device);
        assert_eq!(hybrid.graph_support(), Err("positions_host_scalar"));
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
            Some("positions_host_scalar")
        );
        assert!(runner.last_census().is_none(), "refused before any capture");

        // The hybrid step on the engine's cache (static KV + per-token DeltaNet ring), decode and
        // a 4-token verify: the ring keeps the recurrent state in place, so nothing the step
        // allocates outlives it; the layout uploads remain.
        let mut cache = hybrid.new_cache_for(prompt.len() + 8, 3).unwrap();
        assert_eq!(cache.graph_support(), Ok(()));
        hybrid
            .forward_step(&mut cache, StepRequest::last(&prompt))
            .unwrap();
        hybrid
            .forward_step(&mut cache, StepRequest::last(&[3]))
            .unwrap();
        hybrid
            .forward_step(&mut cache, StepRequest::all(&[4, 5, 6, 7]))
            .unwrap();
        let decode = census_step(&hybrid, &mut cache, StepRequest::last(&[5]))
            .unwrap()
            .expect("the decode step records");
        let verify = census_step(&hybrid, &mut cache, StepRequest::all(&[4, 5, 6, 7]))
            .unwrap()
            .expect("the verify step records");
        for (name, census) in [("1-token", decode), ("4-token", verify)] {
            eprintln!(
                "[runner] qwen35 hybrid {name} static step census: {}",
                census.describe()
            );
            assert_eq!(
                census.escaped_allocs, 0,
                "{name}: the ring is written in place"
            );
            assert!(census.mem_allocs > 0 && census.kernels > 0);
            assert!(census.memcpy_from_host > 0);
            assert_eq!(census.refusal(), Some(REASON_HOST_UPLOAD_IN_CAPTURE));
        }
    }

    /// What a graph replay saves per step on the synthetic model (the mechanism's win on a
    /// launch-bound step; evidence only).
    #[test]
    #[ignore = "timing evidence; needs a quiet GPU"]
    fn synthetic_replay_timing() {
        let Some((_guard, device, _)) = device() else {
            return;
        };
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

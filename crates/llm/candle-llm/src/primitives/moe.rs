//! The sparse Mixture-of-Experts feed-forward — the **one** routing module every MoE family in this
//! crate reaches (sc-24440): Qwen2-MoE and DeepSeek-V2 through the generic
//! [`CausalLm`](crate::models::CausalLm), Qwen3.6-MoE (`qwen3_5_moe`) through
//! [`Qwen35Model`](crate::models::Qwen35Model).
//!
//! **Routing is on the device.** The router's softmax, the top-k (`arg_sort` + `narrow` +
//! `gather`) and the renormalization / routed scaling are tensor ops; the router this replaced
//! pulled every token's probabilities to the host each layer to pick the experts there, which
//! drained the GPU pipeline once per MoE layer and made every MoE model refuse CUDA-graph capture
//! (`moe_router_host_read`).
//!
//! **Dispatch.** A decode-sized step — at most [`DEVICE_DISPATCH_MAX_ROWS`] tokens — runs every
//! (token, slot) pair's expert on its one row, with the routes as device data:
//!
//! * **Indexed** (CUDA): one launch per projection of [`IndexedExperts`] — each pair's expert id
//!   is read on the device and its weight read **in place** through a load-time device table of
//!   expert addresses. No host read, no copy of an expert matrix, and launch shapes fixed by the step, so
//!   the step is graph-capturable. Every expert format the crate loads into a bank has a kernel:
//!   dense (f32 / bf16 / f16), GGML blocks (Q4_0 / Q8_0 / Q4_K — what quantize-on-load and a
//!   prepared snapshot store — and every other type candle's decode MMVQ serves), the MLX-affine Q8
//!   tier (Q8_0 dequantized per forward) and NVFP4. A GGML or NVFP4 pair's output is bit-identical
//!   to that expert's own decode forward on that row (the kernels run candle's MMVQ / the NVFP4
//!   decode GEMV's core); dense and dequant pairs agree with the per-expert matmul to its rounding.
//! * **Gathered** (a dense bank off CUDA, or where the indexed kernel is unavailable): each pair's
//!   stacked expert weights are gathered by the device index (`index_select` — a copy of `slots`
//!   expert matrices per projection) and multiplied in one batched matmul.
//!
//! Every pair is its own one-row product, so a row never depends on its batch; the products are
//! accumulated in ascending expert order, the order the grouped loop below adds them, so a
//! one-token step reproduces the grouped dispatch's sums.
//!
//! * **Grouped**: larger batches (a prefill), and a bank no device dispatch serves — Prism-packed
//!   experts (no Prism MoE checkpoint is in the supported set: Prism GGUF synthesizes a dense
//!   Qwen3.5 config, the MLX Prism release is the dense Bonsai), a bank whose experts mix formats
//!   or carry a bias, NVFP4 with the decode GEMV switched off or a non-bf16 activation, an indexed
//!   kernel that does not compile on the device — read the device-computed routes back once (one
//!   counted host sync), run each expert on just its tokens, and `index_add` the weighted outputs
//!   back. A decode step through such a bank reads the device, so the model says so in
//!   [`SparseMoe::graph_refusal`] (`moe_expert_host_dispatch`).
//! * On the **CPU** every step is dispatched grouped. A host read costs nothing there (the tensors
//!   already live in host memory, and there is no graph to break), and the grouped dispatch reads
//!   each routed expert's weights in place where the gathered one first copies them: measured
//!   1.5–1.8x slower gathered at 64 experts / k = 8 / hidden 1024 / expert FFN 512, t = 1–8
//!   (`tests::stacked_vs_grouped_cpu_timing`). [`with_device_dispatch`] forces the GPU choice on
//!   the CPU so tests exercise the dispatch a GPU step takes; [`with_grouped_dispatch`] forces the
//!   grouped one anywhere, the reference a GPU test compares against.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, Ordering};

use candle_core::{DType, Device, Tensor};
use candle_quant_kernels::{IndexedExperts, IndexedFormat, MoeGemvError, MoeRows};

use crate::error::{Error, Result};
use crate::primitives::host_sync::note_host_sync;
use crate::primitives::nn::swiglu;
use crate::primitives::projection::{Projection, WeightCensus};

/// The most tokens a step may route through a device dispatch. Decode steps (one token per
/// sequence, a handful of sequences) fit; a prefill goes grouped, where each expert's weights are
/// read once for all of its tokens instead of once per (token, slot) pair.
pub const DEVICE_DISPATCH_MAX_ROWS: usize = 8;

/// The graph-capture refusal of a model whose MoE bank is dispatched from host-read routes. A
/// refusal is always composed with why the indexed kernels do not serve the bank
/// ([`HOST_DISPATCH_REASONS`]); the bare reason is only the fallback for a cause the table does
/// not list.
pub const REASON_EXPERT_HOST_DISPATCH: &str = "moe_expert_host_dispatch";

/// Every reason an MoE bank can be refused for graph capture, composed as
/// `moe_expert_host_dispatch:<cause>`: the cause is why the indexed kernels do not serve the bank —
/// not on CUDA, Prism-packed experts, a bank mixing formats or with an unserved one, an activation
/// dtype the format refuses, NVFP4 with the decode GEMV switched off, or the indexed kernel's
/// refusal / compile label (`compute_floor`, `nvrtc`, `load`, `device`, `missing_function`).
pub const HOST_DISPATCH_REASONS: &[(&str, &str)] = &[
    ("not_cuda", "moe_expert_host_dispatch:not_cuda"),
    ("prism", "moe_expert_host_dispatch:prism"),
    ("mixed", "moe_expert_host_dispatch:mixed"),
    ("format", "moe_expert_host_dispatch:format"),
    ("empty", "moe_expert_host_dispatch:empty"),
    ("input", "moe_expert_host_dispatch:input"),
    ("dtype", "moe_expert_host_dispatch:dtype"),
    (
        "nvfp4_gemv_disabled",
        "moe_expert_host_dispatch:nvfp4_gemv_disabled",
    ),
    ("compute_floor", "moe_expert_host_dispatch:compute_floor"),
    ("nvrtc", "moe_expert_host_dispatch:nvrtc"),
    ("load", "moe_expert_host_dispatch:load"),
    ("device", "moe_expert_host_dispatch:device"),
    (
        "missing_function",
        "moe_expert_host_dispatch:missing_function",
    ),
    ("candle", "moe_expert_host_dispatch:candle"),
];

/// The composed graph refusal for an indexed-kernel `cause` ([`HOST_DISPATCH_REASONS`]).
fn host_dispatch_reason(cause: Option<&'static str>) -> &'static str {
    HOST_DISPATCH_REASONS
        .iter()
        .find(|(label, _)| Some(*label) == cause)
        .map_or(REASON_EXPERT_HOST_DISPATCH, |(_, reason)| reason)
}

thread_local! {
    static FORCE_DEVICE_DISPATCH: Cell<bool> = const { Cell::new(false) };
    static FORCE_GROUPED_DISPATCH: Cell<bool> = const { Cell::new(false) };
    static DEVICE_DISPATCHES: Cell<u64> = const { Cell::new(0) };
    static INDEXED_DISPATCHES: Cell<u64> = const { Cell::new(0) };
    static EXPERT_GATHERS: Cell<u64> = const { Cell::new(0) };
}

/// Run `f` with `flag` set on this thread, restoring it after (also on unwind).
fn with_flag<R>(flag: &'static std::thread::LocalKey<Cell<bool>>, f: impl FnOnce() -> R) -> R {
    struct Restore(&'static std::thread::LocalKey<Cell<bool>>, bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            self.0.with(|c| c.set(self.1));
        }
    }
    let _restore = Restore(flag, flag.with(|c| c.replace(true)));
    f()
}

/// Run `f` with this thread's MoE steps dispatched as on a GPU even when the tensors live on the
/// CPU (which otherwise dispatches every step grouped — see the module docs). Test support: it lets
/// a CPU test drive the device dispatch through a whole model.
pub fn with_device_dispatch<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&FORCE_DEVICE_DISPATCH, f)
}

/// Run `f` with this thread's MoE steps dispatched grouped on every device — the per-expert
/// dispatch the device dispatches are measured against. Test and evidence support.
pub fn with_grouped_dispatch<R>(f: impl FnOnce() -> R) -> R {
    with_flag(&FORCE_GROUPED_DISPATCH, f)
}

fn bump(counter: &'static std::thread::LocalKey<Cell<u64>>) {
    counter.with(|c| c.set(c.get().wrapping_add(1)));
}

/// MoE layer steps this thread has run through a device dispatch, indexed or gathered (monotone;
/// take deltas). With [`host_sync_count`](crate::primitives::host_sync_count) it says which
/// dispatch a step's MoE layers took: every layer on the device, none reading back.
pub fn moe_device_dispatch_count() -> u64 {
    DEVICE_DISPATCHES.with(Cell::get)
}

/// MoE layer steps this thread has run through the indexed kernels (monotone; take deltas): every
/// expert weight read in place, nothing gathered.
pub fn moe_indexed_dispatch_count() -> u64 {
    INDEXED_DISPATCHES.with(Cell::get)
}

/// MoE layer steps this thread has run through the gathered dispatch (monotone; take deltas): each
/// one copied `slots` whole expert matrices per projection per token by a device index.
pub fn moe_expert_gather_count() -> u64 {
    EXPERT_GATHERS.with(Cell::get)
}

/// Whether reading routes back to the host would cost `device` a pipeline drain: every GPU, and
/// the CPU only when [`with_device_dispatch`] says to act like one.
fn host_reads_stall(device: &Device) -> bool {
    !device.is_cpu() || FORCE_DEVICE_DISPATCH.with(Cell::get)
}

/// A SwiGLU MLP over three projections: a routed expert of a bank Candle cannot stack, or the
/// always-on shared expert.
pub struct SwiGlu {
    /// Gate projection.
    pub gate: Projection,
    /// Up projection.
    pub up: Projection,
    /// Down projection.
    pub down: Projection,
}

impl SwiGlu {
    /// `down(silu(gate(x)) · up(x))`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let g = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        self.down.forward(&swiglu(&g, &up)?)
    }

    /// Add the three projections to a census.
    pub fn record(&self, census: &mut WeightCensus) {
        for p in [&self.gate, &self.up, &self.down] {
            census.projections.record(p);
        }
    }

    fn part(&self, part: ExpertPart) -> &Projection {
        match part {
            ExpertPart::Gate => &self.gate,
            ExpertPart::Up => &self.up,
            ExpertPart::Down => &self.down,
        }
    }
}

/// One projection of a routed expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertPart {
    /// The gate projection.
    Gate,
    /// The up projection.
    Up,
    /// The down projection.
    Down,
}

const PARTS: [ExpertPart; 3] = [ExpertPart::Gate, ExpertPart::Up, ExpertPart::Down];

/// How the routed experts are held.
enum Layout {
    /// Dense experts stacked per projection: `gate` / `up` `[experts, inter, hidden]`, `down`
    /// `[experts, hidden, inter]`.
    Stacked {
        gate: Tensor,
        up: Tensor,
        down: Tensor,
    },
    /// Experts of any other representation (quantized / NVFP4 / Prism, or a mixed bank).
    PerExpert(Vec<SwiGlu>),
}

/// The routed experts: their layout, plus — for a CUDA bank the indexed kernels serve — the
/// gate / up / down expert-address tables they read the weights through.
struct ExpertBank {
    layout: Layout,
    /// The indexed tables, or why the bank has none (a stable label: `not_cuda`, `prism`,
    /// `mixed`, `format`, ...).
    indexed: std::result::Result<Box<[IndexedExperts; 3]>, &'static str>,
}

/// How a step's routed experts are dispatched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dispatch {
    /// The indexed kernels (CUDA): weights read in place by the device ids.
    Indexed,
    /// Stacked expert weights gathered by the device ids, then a batched matmul.
    Gathered,
    /// Routes read back to the host; each expert run on its own tokens.
    Grouped,
}

/// The indexed tables for one projection of a per-expert bank, or why there are none.
fn indexed_part(
    experts: &[SwiGlu],
    part: ExpertPart,
) -> std::result::Result<std::result::Result<IndexedExperts, &'static str>, MoeGemvError> {
    let projections = experts.iter().map(|e| e.part(part)).collect::<Vec<_>>();
    let built = match projections.first() {
        Some(Projection::Quantized(_)) => {
            let mut sources = Vec::with_capacity(projections.len());
            for p in &projections {
                match p {
                    Projection::Quantized(q) => match q.indexed_source() {
                        Some(source) => sources.push(source),
                        None => return Ok(Err("format")),
                    },
                    _ => return Ok(Err("mixed")),
                }
            }
            let dequant = sources[0].1;
            if sources.iter().any(|(_, d)| *d != dequant) {
                return Ok(Err("mixed"));
            }
            let tensors = sources.into_iter().map(|(q, _)| q).collect::<Vec<_>>();
            IndexedExperts::ggml(&tensors, dequant)
        }
        Some(Projection::Nvfp4(_)) => {
            let mut weights = Vec::with_capacity(projections.len());
            for p in &projections {
                match p {
                    Projection::Nvfp4(w) => weights.push(w.clone()),
                    _ => return Ok(Err("mixed")),
                }
            }
            IndexedExperts::nvfp4(&weights)
        }
        Some(Projection::Prism(_)) => return Ok(Err("prism")),
        // A dense projection in a per-expert bank: a biased or mixed bank (an all-dense, bias-free
        // bank is stacked instead).
        Some(Projection::Dense(_)) | None => return Ok(Err("format")),
    };
    match built {
        Ok(bank) => Ok(Ok(bank)),
        Err(MoeGemvError::Refused(r)) => Ok(Err(r.label())),
        Err(e @ (MoeGemvError::Compile(_) | MoeGemvError::Candle(_))) => Err(e),
    }
}

/// The three tables from per-projection results: the first refusal wins.
fn indexed_bank(
    parts: [std::result::Result<std::result::Result<IndexedExperts, &'static str>, MoeGemvError>;
        3],
) -> Result<std::result::Result<Box<[IndexedExperts; 3]>, &'static str>> {
    let [g, u, d] = parts;
    Ok(
        match (
            g.map_err(gemv_err)?,
            u.map_err(gemv_err)?,
            d.map_err(gemv_err)?,
        ) {
            (Ok(g), Ok(u), Ok(d)) => Ok(Box::new([g, u, d])),
            (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => Err(why),
        },
    )
}

fn gemv_err(e: MoeGemvError) -> Error {
    Error::Candle(e.into())
}

impl ExpertBank {
    fn new(experts: Vec<SwiGlu>) -> Result<Self> {
        let dense = |p: &Projection| match p {
            Projection::Dense(l) if l.bias().is_none() => Some(l.weight().clone()),
            _ => None,
        };
        let stackable = experts.iter().all(|e| {
            [&e.gate, &e.up, &e.down]
                .into_iter()
                .all(|p| dense(p).is_some())
        });
        if !stackable {
            let indexed = indexed_bank(PARTS.map(|part| indexed_part(&experts, part)))?;
            return Ok(Self {
                layout: Layout::PerExpert(experts),
                indexed,
            });
        }
        let mut parts: [Vec<Tensor>; 3] = Default::default();
        for e in &experts {
            for (i, p) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
                parts[i].extend(dense(p));
            }
        }
        let [gate, up, down] = parts;
        let (gate, up, down) = (
            Tensor::stack(&gate, 0)?,
            Tensor::stack(&up, 0)?,
            Tensor::stack(&down, 0)?,
        );
        let indexed = indexed_bank([&gate, &up, &down].map(|w| match IndexedExperts::dense(w) {
            Ok(bank) => Ok(Ok(bank)),
            Err(MoeGemvError::Refused(r)) => Ok(Err(r.label())),
            Err(e) => Err(e),
        }))?;
        Ok(Self {
            layout: Layout::Stacked { gate, up, down },
            indexed,
        })
    }

    /// Expert `e`'s `part` applied to `x` `[n, in]`.
    fn project(&self, e: usize, part: ExpertPart, x: &Tensor) -> Result<Tensor> {
        match &self.layout {
            Layout::Stacked { gate, up, down } => {
                let w = match part {
                    ExpertPart::Gate => gate,
                    ExpertPart::Up => up,
                    ExpertPart::Down => down,
                };
                // The same `x · wᵀ` a dense `Linear` runs on its own `[out, in]` weight.
                Ok(x.matmul(&w.get(e)?.t()?)?)
            }
            Layout::PerExpert(experts) => experts[e].part(part).forward(x),
        }
    }

    /// Expert `e`'s SwiGLU on `x` `[n, hidden]`.
    fn expert(&self, e: usize, x: &Tensor) -> Result<Tensor> {
        match &self.layout {
            Layout::Stacked { .. } => {
                let g = self.project(e, ExpertPart::Gate, x)?;
                let up = self.project(e, ExpertPart::Up, x)?;
                self.project(e, ExpertPart::Down, &swiglu(&g, &up)?)
            }
            Layout::PerExpert(experts) => experts[e].forward(x),
        }
    }

    /// Why the indexed kernels cannot serve a step of `act` activations, or `None` when they can:
    /// the bank has tables, their kernels compile on the device (the nvrtc seam's cached outcome)
    /// and serve `act`, and — NVFP4 — the decode GEMV switch is on (the indexed NVFP4 kernel is
    /// that GEMV's core; with the switch off every NVFP4 projection runs cuBLASLt).
    fn indexed_refusal(&self, act: DType) -> Option<&'static str> {
        let banks = match &self.indexed {
            Ok(banks) => banks,
            Err(why) => return Some(why),
        };
        for bank in banks.iter() {
            if bank.format().io_dtype(act).is_none() {
                return Some("dtype");
            }
            if bank.format() == IndexedFormat::Nvfp4
                && !crate::primitives::nvfp4_path::nvfp4_gemv_enabled()
            {
                return Some("nvfp4_gemv_disabled");
            }
            if let Err(e) = bank.available() {
                return Some(e.label());
            }
        }
        None
    }

    /// Device bytes a decode step's routed dispatch allocates beyond its input, per token row, for
    /// activations of `act` on this bank's device: the largest of the dispatches it may take
    /// (grouped; indexed; for a stacked bank on a device, gathered), so admission covers whichever
    /// runs.
    fn step_bytes_per_token(&self, act: DType, slots: usize) -> u64 {
        let e = act.size_in_bytes();
        let (inter, hidden) = self.dims();
        // The grouped dispatch: each routed expert's three projections and SwiGLU on its row.
        let mut bytes = slots * (3 * inter + 2 * hidden) * e;
        if let Ok(banks) = &self.indexed {
            // Each projection's own workspace, plus the activation casts around it (GGML widens
            // to f32 and narrows back) and the SwiGLU / weighting temporaries.
            let mut indexed = 0usize;
            for (bank, rows) in
                banks
                    .iter()
                    .zip([MoeRows::PerToken, MoeRows::PerToken, MoeRows::PerSlot])
            {
                let (n, k) = bank.shape();
                let input = match rows {
                    MoeRows::PerToken => k,
                    MoeRows::PerSlot => slots * k,
                };
                indexed += bank.workspace_bytes(act, 1, slots, rows) + input * 4 + slots * n * e;
            }
            bytes = bytes.max(indexed);
        }
        if matches!(self.layout, Layout::Stacked { .. }) && !self.on_cpu() {
            // A stacked bank on a device may also run gathered (its indexed kernel refused):
            // `slots` copied expert matrices per projection.
            bytes = bytes.max(slots * 3 * inter * hidden * e + slots * (3 * inter + hidden) * e);
        }
        bytes as u64
    }

    /// `(expert FFN width, hidden)`.
    fn dims(&self) -> (usize, usize) {
        match &self.layout {
            Layout::Stacked { gate, .. } => (gate.dim(1).unwrap_or(0), gate.dim(2).unwrap_or(0)),
            Layout::PerExpert(experts) => experts.first().map_or((0, 0), |e| e.gate.dims()),
        }
    }

    fn on_cpu(&self) -> bool {
        match &self.layout {
            Layout::Stacked { gate, .. } => gate.device().is_cpu(),
            // A per-expert bank has tables only on CUDA.
            Layout::PerExpert(_) => self.indexed.is_err(),
        }
    }
}

/// How the router turns probabilities into expert weights.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeRouting {
    /// Experts each token is routed to.
    pub experts_per_tok: usize,
    /// Renormalize the top-k probabilities to sum to 1 (Qwen3.6-MoE; Qwen2-MoE when its config
    /// says so). Otherwise they are multiplied by `routed_scaling_factor`.
    pub norm_topk_prob: bool,
    /// Multiplier on the un-normalized routed weights (DeepSeek-V2); `1.0` for Qwen2-MoE. Ignored
    /// when `norm_topk_prob`.
    pub routed_scaling_factor: f32,
}

/// A sparse Mixture-of-Experts FFN: a softmax router over the routed experts (top-k per token)
/// plus an always-on shared expert, sigmoid-gated (Qwen2-MoE, Qwen3.6-MoE) or added ungated
/// (DeepSeek-V2). `n_group` / `topk_group` group-limited routing (DeepSeek-V2-236B / V3) is not
/// modelled — the verification model (V2-Lite) uses plain greedy top-k.
pub struct SparseMoe {
    /// Router weight `[experts, hidden]`, in the model's activation dtype.
    router: Tensor,
    bank: ExpertBank,
    num_experts: usize,
    shared: SwiGlu,
    /// Shared-expert sigmoid gate `[1, hidden]`; `None` ⇒ the shared expert is added ungated.
    shared_gate: Option<Tensor>,
    routing: MoeRouting,
    /// Whether a gathered dispatch on CUDA (the indexed kernels refused) was logged.
    gather_noted: AtomicBool,
}

impl SparseMoe {
    /// Assemble the block from the router `[experts, hidden]` (in the model's activation dtype),
    /// the per-expert SwiGLUs (stacked here when every projection is dense; on CUDA, indexed
    /// through load-time expert-address tables when their format has a kernel), the shared expert
    /// and its optional gate.
    pub fn new(
        router: Tensor,
        experts: Vec<SwiGlu>,
        shared: SwiGlu,
        shared_gate: Option<Tensor>,
        routing: MoeRouting,
    ) -> Result<Self> {
        let num_experts = experts.len();
        if num_experts == 0 || router.dim(0)? != num_experts {
            return Err(Error::Config(format!(
                "an MoE router over {} experts needs as many experts, got {num_experts}",
                router.dim(0)?
            )));
        }
        Ok(Self {
            router,
            bank: ExpertBank::new(experts)?,
            num_experts,
            shared,
            shared_gate,
            routing,
            gather_noted: AtomicBool::new(false),
        })
    }

    /// The routed-expert count.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    fn top_k(&self) -> usize {
        self.routing.experts_per_tok.clamp(1, self.num_experts)
    }

    /// The device dispatch a `t`-row step takes on a GPU, or [`Dispatch::Grouped`]: indexed when
    /// the indexed kernels serve the bank, else gathered for a stacked bank — both only for a
    /// decode-sized step. The one dispatch decision — [`forward`](Self::forward) and
    /// [`graph_refusal`](Self::graph_refusal) both read it.
    fn device_dispatch(&self, t: usize) -> Dispatch {
        if t > DEVICE_DISPATCH_MAX_ROWS {
            return Dispatch::Grouped;
        }
        if self.bank.indexed_refusal(self.router.dtype()).is_none() {
            return Dispatch::Indexed;
        }
        match self.bank.layout {
            Layout::Stacked { .. } => Dispatch::Gathered,
            Layout::PerExpert(_) => Dispatch::Grouped,
        }
    }

    /// Why a decode step through this block cannot be captured as a CUDA graph, if it cannot: a
    /// decode-sized step (up to [`DEVICE_DISPATCH_MAX_ROWS`] tokens) that is not dispatched on the
    /// device reads its routes back to the host. The reason names why the indexed kernels do not
    /// serve the bank, `moe_expert_host_dispatch:<cause>` ([`HOST_DISPATCH_REASONS`]), so it reaches
    /// `graph_support`, the runner's fallback reason and the decode report by name.
    pub fn graph_refusal(&self) -> Option<&'static str> {
        (self.device_dispatch(DEVICE_DISPATCH_MAX_ROWS) == Dispatch::Grouped)
            .then(|| host_dispatch_reason(self.indexed_refusal()))
    }

    /// Why the indexed kernels do not serve this block's experts (`not_cuda`, `prism`, `mixed`,
    /// `format`, `dtype`, `nvfp4_gemv_disabled`, a compile label), `None` when they do. A stacked
    /// dense bank they refuse still dispatches on the device (gathered); any other bank is
    /// dispatched grouped and refused for graph capture with this cause.
    pub fn indexed_refusal(&self) -> Option<&'static str> {
        self.bank.indexed_refusal(self.router.dtype())
    }

    /// Device bytes of this block's indexed expert-address tables (the three projections'
    /// [`IndexedExperts::table_bytes`]) — `0` without them. A load prices at most
    /// `3 · experts · MAX_TABLE_BYTES_PER_EXPERT` per MoE layer for them.
    pub fn indexed_table_bytes(&self) -> usize {
        self.bank.indexed.as_ref().map_or(0, |banks| {
            banks.iter().map(IndexedExperts::table_bytes).sum()
        })
    }

    /// Device bytes a decode step through this block's routed experts allocates per token row
    /// (the dispatch's workspace and temporaries, `experts_per_tok` slots) — what admission prices
    /// per token on top of the dense step working set.
    pub fn step_bytes_per_token(&self) -> u64 {
        self.bank
            .step_bytes_per_token(self.router.dtype(), self.top_k())
    }

    /// Route every token on the device: the top-k expert ids `[t, k]` (u32, most probable first)
    /// and their f32 weights `[t, k]`, from `xf` `[t, hidden]`.
    ///
    /// The arithmetic is the pre-change host router's, op for op: an f32 softmax, the k most
    /// probable experts, then either `p / Σ top-k p` (the sum taken in the same most-probable-first
    /// order) or `p · routed_scaling_factor`. The host router clamped the sum at
    /// `f32::MIN_POSITIVE`; it can never reach it — the top-k include the largest probability,
    /// which is at least `1 / experts` — so no clamp is taken here.
    pub fn route(&self, xf: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.top_k();
        let logits = xf.matmul(&self.router.t()?)?; // [t, E]
        let probs = candle_nn::ops::softmax_last_dim(&logits.to_dtype(DType::F32)?)?;
        let ids = probs
            .arg_sort_last_dim(false)?
            .narrow(1, 0, k)?
            .contiguous()?; // [t, k]
        let top = probs.gather(&ids, 1)?; // [t, k]
        let weights = if self.routing.norm_topk_prob {
            let mut denom = top.narrow(1, 0, 1)?;
            for j in 1..k {
                denom = (denom + top.narrow(1, j, 1)?)?;
            }
            top.broadcast_div(&denom)?
        } else {
            (top * f64::from(self.routing.routed_scaling_factor))?
        };
        Ok((ids, weights))
    }

    /// The block's output for `x` `[batch, seq, hidden]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let xf = x.reshape((t, h))?;
        let (ids, weights) = self.route(&xf)?;
        let dispatch = if FORCE_GROUPED_DISPATCH.with(Cell::get) || !host_reads_stall(xf.device()) {
            Dispatch::Grouped
        } else {
            self.device_dispatch(t)
        };
        let routed = match (dispatch, &self.bank.indexed, &self.bank.layout) {
            (Dispatch::Indexed, Ok(banks), _) => {
                bump(&DEVICE_DISPATCHES);
                bump(&INDEXED_DISPATCHES);
                dispatch_indexed(&xf, &ids, &weights, banks)?
            }
            (Dispatch::Gathered, _, Layout::Stacked { gate, up, down }) => {
                if xf.device().is_cuda() && !self.gather_noted.swap(true, Ordering::Relaxed) {
                    // On CUDA the indexed kernels serve every stacked bank; a gather there means
                    // they were refused (a compile failure) — said once per block, by name.
                    tracing::warn!(
                        cause = self.indexed_refusal().unwrap_or("unknown"),
                        "MoE experts gathered by index_select on CUDA: the indexed kernels are \
                         unavailable"
                    );
                }
                bump(&DEVICE_DISPATCHES);
                bump(&EXPERT_GATHERS);
                dispatch_stacked(&xf, &ids, &weights, [gate, up, down])?
            }
            _ => self.dispatch_grouped(&xf, &ids, &weights)?,
        };

        // Always-on shared expert: Qwen2 / Qwen3.6 gate it by sigmoid(x · shared_gateᵀ);
        // DeepSeek packs several shared experts into one MLP and adds them ungated.
        let shared = self.shared.forward(&xf)?;
        let shared = match &self.shared_gate {
            Some(g) => {
                let sg = candle_nn::ops::sigmoid(&xf.matmul(&g.t()?)?)?; // [t, 1]
                shared.broadcast_mul(&sg)?
            }
            None => shared,
        };
        Ok((routed + shared)?.reshape((b, s, h))?)
    }

    /// Group the routed (token, weight) pairs per expert on the host, run each expert on just its
    /// tokens, and `index_add` the weighted outputs back.
    fn dispatch_grouped(&self, xf: &Tensor, ids: &Tensor, weights: &Tensor) -> Result<Tensor> {
        let (t, h) = xf.dims2()?;
        let (dtype, device) = (xf.dtype(), xf.device());
        let mut routed: Vec<Vec<(u32, f32)>> = vec![Vec::new(); self.num_experts];
        for (ti, row) in read_routes(ids, weights)?.into_iter().enumerate() {
            for (e, w) in row {
                routed[e as usize].push((ti as u32, w));
            }
        }
        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for (e, toks) in routed.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx = Tensor::from_vec(
                toks.iter().map(|&(ti, _)| ti).collect::<Vec<u32>>(),
                (n,),
                device,
            )?;
            let wts = Tensor::from_vec(
                toks.iter().map(|&(_, w)| w).collect::<Vec<f32>>(),
                (n, 1),
                device,
            )?
            .to_dtype(dtype)?;
            let xe = xf.index_select(&idx, 0)?; // [n, h]
            let ye = self.bank.expert(e, &xe)?.broadcast_mul(&wts)?; // [n, h]
            out = out.index_add(&idx, &ye, 0)?;
        }
        Ok(out)
    }

    /// Expert `e`'s `part` applied to `x` `[n, in]` — the per-expert view a test compares
    /// representations through, whether the bank is stacked or not.
    #[cfg(test)]
    pub(crate) fn expert_projection(
        &self,
        e: usize,
        part: ExpertPart,
        x: &Tensor,
    ) -> Result<Tensor> {
        self.bank.project(e, part, x)
    }

    /// Which representation expert `e`'s `part` holds.
    #[cfg(test)]
    pub(crate) fn expert_kind(
        &self,
        e: usize,
        part: ExpertPart,
    ) -> crate::primitives::projection::ProjectionKind {
        match &self.bank.layout {
            Layout::Stacked { .. } => crate::primitives::projection::ProjectionKind::Dense,
            Layout::PerExpert(experts) => experts[e].part(part).kind(),
        }
    }

    /// Add the block's weights to a census. A stacked dense bank counts as its per-expert dense
    /// projections, exactly as the un-stacked bank did.
    pub fn record(&self, census: &mut WeightCensus) {
        census.record_tensor(&self.router);
        if let Some(gate) = &self.shared_gate {
            census.record_tensor(gate);
        }
        match &self.bank.layout {
            Layout::Stacked { gate, up, down } => {
                for w in [gate, up, down] {
                    let experts = w.dim(0).unwrap_or(0) as u64;
                    let tally = &mut census.projections.dense;
                    tally.count += experts;
                    tally.params += w.elem_count() as u64;
                    tally.resident_bytes += (w.elem_count() * w.dtype().size_in_bytes()) as u64;
                }
            }
            Layout::PerExpert(experts) => {
                for expert in experts {
                    expert.record(census);
                }
            }
        }
        self.shared.record(census);
    }
}

/// The indexed dispatch: each projection one launch over every (token, slot) pair, the expert
/// read in place by its device id ([`IndexedExperts`]). Pairs are visited in ascending expert
/// order per token and their weighted outputs summed in that order, as the grouped dispatch sums
/// them.
fn dispatch_indexed(
    xf: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    [gate, up, down]: &[IndexedExperts; 3],
) -> Result<Tensor> {
    let (t, h) = xf.dims2()?;
    let k = ids.dim(1)?;
    // `ids` is `route`'s fresh `[t, k]` tensor (see `dispatch_stacked` on why a view would not
    // do on the CPU).
    let (ids, perm) = ids.sort_last_dim(true)?;
    let weights = weights.gather(&perm, 1)?.to_dtype(xf.dtype())?; // [t, k]
    let g = project_indexed(gate, xf, &ids, MoeRows::PerToken)?; // [t·k, inter]
    let u = project_indexed(up, xf, &ids, MoeRows::PerToken)?;
    let y = project_indexed(down, &swiglu(&g, &u)?, &ids, MoeRows::PerSlot)?; // [t·k, h]
    let y = y
        .reshape((t, k, h))?
        .broadcast_mul(&weights.unsqueeze(2)?)?;
    let mut out = Tensor::zeros((t, h), xf.dtype(), xf.device())?;
    for j in 0..k {
        out = (out + y.narrow(1, j, 1)?.squeeze(1)?)?;
    }
    Ok(out)
}

/// One indexed projection in the dtype convention of the per-expert forward it replaces: the
/// activation in the format's input dtype (GGML: widened to f32, as `QuantizedLinear` does), the
/// result back in the activation's dtype.
fn project_indexed(
    bank: &IndexedExperts,
    x: &Tensor,
    ids: &Tensor,
    rows: MoeRows,
) -> Result<Tensor> {
    let io = bank.format().io_dtype(x.dtype()).ok_or_else(|| {
        Error::Msg(format!(
            "{:?} activations reached an indexed bank that refuses them",
            x.dtype()
        ))
    })?;
    let y = bank
        .forward(&x.to_dtype(io)?, ids, rows)
        .map_err(gemv_err)?;
    if bank.format() == IndexedFormat::Nvfp4 {
        // The NVFP4 path tally counts this launch as a decode-GEMV run (its kernel is the GEMV's).
        crate::primitives::nvfp4_path::note_gemv();
    }
    Ok(y.to_dtype(x.dtype())?)
}

/// The gathered dispatch: gather each (token, slot) pair's expert weights by its device index and
/// run one batched matmul per projection. Everything stays on the device and every shape is
/// fixed by `(t, k)`, so the step is graph-replayable — but each pair copies its expert's whole
/// matrices first. The dispatch a dense bank takes where the indexed kernels do not serve it.
fn dispatch_stacked(
    xf: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    [gate, up, down]: [&Tensor; 3],
) -> Result<Tensor> {
    let (t, h) = xf.dims2()?;
    let k = ids.dim(1)?;
    // Visit each token's experts in ascending id order: the grouped dispatch adds them in that
    // order, and matching it keeps the two paths bit-identical.
    // `ids` is `route`'s fresh `[t, k]` tensor: candle's CPU arg-sort reads from the start of the
    // storage whatever a view's offset, so a narrowed view would sort the wrong rows.
    let (ids, perm) = ids.sort_last_dim(true)?;
    let weights = weights.gather(&perm, 1)?.to_dtype(xf.dtype())?; // [t, k]
    let flat = ids.flatten_all()?; // [t·k]
    let xs = xf
        .unsqueeze(1)?
        .broadcast_as((t, k, h))?
        .contiguous()?
        .reshape((t * k, 1, h))?;
    let project = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
        Ok(x.matmul(&w.index_select(&flat, 0)?.transpose(1, 2)?)?)
    };
    let g = project(&xs, gate)?; // [t·k, 1, inter]
    let u = project(&xs, up)?;
    let y = project(&swiglu(&g, &u)?, down)?; // [t·k, 1, h]
    let y = y
        .reshape((t, k, h))?
        .broadcast_mul(&weights.unsqueeze(2)?)?;
    let mut out = Tensor::zeros((t, h), xf.dtype(), xf.device())?;
    for j in 0..k {
        out = (out + y.narrow(1, j, 1)?.squeeze(1)?)?;
    }
    Ok(out)
}

/// Read the routes back to the host — the grouped dispatch's one device→host transfer, counted
/// as one host sync. The ids ride along as f32 (exact below 2²⁴ experts) so ids and weights move
/// in a single copy.
fn read_routes(ids: &Tensor, weights: &Tensor) -> Result<Vec<Vec<(u32, f32)>>> {
    let k = ids.dim(1)?;
    let both = Tensor::cat(&[&ids.to_dtype(DType::F32)?, weights], 1)?;
    note_host_sync();
    let rows = both.to_vec2::<f32>()?;
    Ok(rows
        .into_iter()
        .map(|row| (0..k).map(|j| (row[j] as u32, row[k + j])).collect())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::host_sync::host_sync_count;
    use crate::primitives::projection::QuantSpec;
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    use candle_core::Device;

    fn randn(shape: &[usize], rng: &mut SplitMix64) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn block(
        e: usize,
        h: usize,
        inter: usize,
        routing: MoeRouting,
        quant: Option<QuantSpec>,
    ) -> SparseMoe {
        let mut rng = SplitMix64::new(0x2444_0100);
        let router = randn(&[e, h], &mut rng);
        let swiglu = |rng: &mut SplitMix64, quant: Option<QuantSpec>| SwiGlu {
            gate: Projection::load(randn(&[inter, h], rng), quant).unwrap(),
            up: Projection::load(randn(&[inter, h], rng), quant).unwrap(),
            down: Projection::load(randn(&[h, inter], rng), quant).unwrap(),
        };
        let experts = (0..e).map(|_| swiglu(&mut rng, quant)).collect();
        let shared = swiglu(&mut rng, None);
        let gate = randn(&[1, h], &mut rng);
        SparseMoe::new(router, experts, shared, Some(gate), routing).unwrap()
    }

    const NORM: MoeRouting = MoeRouting {
        experts_per_tok: 2,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    };
    const SCALED: MoeRouting = MoeRouting {
        experts_per_tok: 3,
        norm_topk_prob: false,
        routed_scaling_factor: 2.5,
    };
    /// Enough slots that the order the routed products are summed in shows in the bits.
    const WIDE_K: MoeRouting = MoeRouting {
        experts_per_tok: 6,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    };

    /// The device router picks the top-k and weights them exactly as the host router did.
    #[test]
    fn device_routes_match_a_host_top_k() {
        for routing in [NORM, SCALED] {
            let moe = block(8, 32, 16, routing, None);
            let mut rng = SplitMix64::new(9);
            let x = randn(&[5, 32], &mut rng);
            let (ids, weights) = moe.route(&x).unwrap();
            let probs =
                candle_nn::ops::softmax_last_dim(&x.matmul(&moe.router.t().unwrap()).unwrap())
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
            let ids = ids.to_vec2::<u32>().unwrap();
            let weights = weights.to_vec2::<f32>().unwrap();
            for (ti, row) in probs.iter().enumerate() {
                let mut idx: Vec<usize> = (0..8).collect();
                idx.sort_unstable_by(|&a, &b| row[b].total_cmp(&row[a]));
                let top = &idx[..routing.experts_per_tok];
                let denom: f32 = top.iter().map(|&e| row[e]).sum();
                for (j, &e) in top.iter().enumerate() {
                    assert_eq!(ids[ti][j] as usize, e, "token {ti} slot {j}");
                    let want = if routing.norm_topk_prob {
                        row[e] / denom
                    } else {
                        row[e] * routing.routed_scaling_factor
                    };
                    assert_eq!(
                        weights[ti][j].to_bits(),
                        want.to_bits(),
                        "token {ti} slot {j}"
                    );
                }
            }
        }
    }

    /// A single-token step through the stacked device dispatch is bit-identical to the grouped
    /// dispatch (each routed expert sees exactly one row either way). A multi-row device dispatch
    /// is row-for-row bit-identical to dispatching its tokens' routes one at a time — every
    /// (token, slot) pair is its own one-row product — and agrees with the grouped dispatch (whose
    /// per-expert GEMMs span several rows) to rounding. (The router's own `[t, hidden]` matmul is a
    /// plain GEMM whose rounding the backend may vary with `t`; the dispatch is compared on one
    /// set of routes.)
    #[test]
    fn stacked_device_dispatch_matches_the_grouped_dispatch() {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for routing in [NORM, SCALED, WIDE_K] {
            let moe = block(8, 32, 16, routing, None);
            let Layout::Stacked { gate, up, down } = &moe.bank.layout else {
                panic!("a dense bank stacks")
            };
            let device = |x: &Tensor, ids: &Tensor, weights: &Tensor| {
                host(&dispatch_stacked(x, ids, weights, [gate, up, down]).unwrap())
            };
            for t in 1..=DEVICE_DISPATCH_MAX_ROWS {
                let mut rng = SplitMix64::new(0x2444_0200 + t as u64);
                let x = randn(&[t, 32], &mut rng);
                let (ids, weights) = moe.route(&x).unwrap();
                let d = device(&x, &ids, &weights);
                let g = host(&moe.dispatch_grouped(&x, &ids, &weights).unwrap());
                if t == 1 {
                    assert_eq!(bits(&d), bits(&g), "{routing:?}: device vs grouped");
                } else {
                    let md = d
                        .iter()
                        .zip(&g)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(md < 1e-6, "t={t} {routing:?}: device vs grouped {md}");
                }
                let ids_host = ids.to_vec2::<u32>().unwrap();
                for (ti, row_ids) in ids_host.iter().enumerate() {
                    let one = |a: &Tensor| a.narrow(0, ti, 1).unwrap();
                    // Fresh storage, not a view: candle's CPU arg-sort ignores a view's start
                    // offset (it would sort row 0), and `route` never hands out a view.
                    let row_ids =
                        Tensor::from_slice(row_ids, (1, row_ids.len()), &Device::Cpu).unwrap();
                    let row = device(&one(&x), &row_ids, &one(&weights));
                    assert_eq!(
                        bits(&d[ti * 32..(ti + 1) * 32]),
                        bits(&row),
                        "t={t} {routing:?}: row {ti} depends on its batch"
                    );
                }
            }
        }
    }

    /// Syncs and device dispatches a `forward` of `x` issues on this thread.
    fn step_counts(moe: &SparseMoe, x: &Tensor) -> (u64, u64) {
        let (syncs, dispatches) = (host_sync_count(), moe_device_dispatch_count());
        moe.forward(x).unwrap();
        (
            host_sync_count() - syncs,
            moe_device_dispatch_count() - dispatches,
        )
    }

    /// A decode step through a dense bank, dispatched as on a GPU, reads nothing back; a quantized
    /// bank's grouped dispatch reads its routes once — and says so as its graph refusal. On the
    /// CPU the dense bank goes grouped too (a host read is free there).
    #[test]
    fn host_reads_follow_the_bank() {
        let x = randn(&[1, 1, 64], &mut SplitMix64::new(3));
        let dense = block(8, 64, 64, NORM, None);
        assert_eq!(with_device_dispatch(|| step_counts(&dense, &x)), (0, 1));
        assert_eq!(dense.graph_refusal(), None);
        assert_eq!(
            step_counts(&dense, &x),
            (1, 0),
            "the CPU dispatches grouped"
        );

        let q8 = block(8, 64, 64, NORM, Some(QuantSpec::q8()));
        assert_eq!(
            q8.expert_kind(0, ExpertPart::Gate),
            crate::primitives::projection::ProjectionKind::Ggml
        );
        assert_eq!(with_device_dispatch(|| step_counts(&q8, &x)), (1, 0));
        // The indexed tables exist on CUDA only: on the CPU a quantized bank is refused by name,
        // the cause composed into the graph refusal itself.
        assert_eq!(
            q8.graph_refusal(),
            Some("moe_expert_host_dispatch:not_cuda")
        );
        assert_eq!(q8.indexed_refusal(), Some("not_cuda"));
        // A dense bank the indexed kernels refuse still dispatches on the device (gathered).
        assert_eq!(dense.indexed_refusal(), Some("not_cuda"));
    }

    /// Every cause the indexed kernels can refuse a bank for composes into a named graph refusal
    /// — the bank's own refusals, the kernel crate's refusal and compile labels — and an unknown
    /// cause falls back to the bare reason.
    #[test]
    fn every_indexed_refusal_cause_composes_a_named_graph_refusal() {
        use candle_quant_kernels::{KernelCompileError, MoeGemvError, MoeGemvRefusal};
        let compile = |e: KernelCompileError| MoeGemvError::Compile(e).label();
        let name = "k";
        let causes = [
            "prism",
            "dtype",
            "nvfp4_gemv_disabled",
            MoeGemvRefusal::NotCuda.label(),
            MoeGemvRefusal::Empty.label(),
            MoeGemvRefusal::Mixed(String::new()).label(),
            MoeGemvRefusal::Format(String::new()).label(),
            MoeGemvRefusal::Input(String::new()).label(),
            MoeGemvError::Candle(candle_core::Error::Msg(String::new())).label(),
            compile(KernelCompileError::BelowComputeFloor {
                name,
                floor: (7, 0),
                found: (6, 0),
            }),
            compile(KernelCompileError::Nvrtc {
                name,
                message: String::new(),
            }),
            compile(KernelCompileError::Load {
                name,
                message: String::new(),
            }),
            compile(KernelCompileError::Device {
                name,
                message: String::new(),
            }),
            compile(KernelCompileError::MissingFunction {
                name,
                function: String::new(),
            }),
        ];
        for cause in causes {
            assert_eq!(
                host_dispatch_reason(Some(cause)),
                format!("{REASON_EXPERT_HOST_DISPATCH}:{cause}"),
                "{cause}"
            );
        }
        assert_eq!(
            host_dispatch_reason(Some("new")),
            REASON_EXPERT_HOST_DISPATCH
        );
        assert_eq!(host_dispatch_reason(None), REASON_EXPERT_HOST_DISPATCH);
    }

    /// Off CUDA a dense bank's device dispatch is the gathered one (it copies its experts — the
    /// copy counter says so), and [`with_grouped_dispatch`] overrides every device choice.
    #[test]
    fn the_gathered_dispatch_is_counted_and_grouped_can_be_forced() {
        let x = randn(&[1, 1, 64], &mut SplitMix64::new(6));
        let dense = block(8, 64, 64, NORM, None);
        let (gathers, indexed) = (moe_expert_gather_count(), moe_indexed_dispatch_count());
        assert_eq!(with_device_dispatch(|| step_counts(&dense, &x)), (0, 1));
        assert_eq!(moe_expert_gather_count() - gathers, 1);
        assert_eq!(moe_indexed_dispatch_count() - indexed, 0);
        assert_eq!(
            with_device_dispatch(|| with_grouped_dispatch(|| step_counts(&dense, &x))),
            (1, 0)
        );
        assert!(
            FORCE_GROUPED_DISPATCH.with(|c| !c.get()),
            "the flag is restored"
        );
    }

    /// Admission's per-token bytes cover the dispatch a step takes: on the CPU the grouped one
    /// (each routed expert's projections and SwiGLU on its row).
    #[test]
    fn step_bytes_cover_the_grouped_dispatch_on_the_cpu() {
        let (e, h, inter) = (8, 64, 32);
        let dense = block(e, h, inter, SCALED, None);
        let slots = SCALED.experts_per_tok;
        let grouped = (slots * (3 * inter + 2 * h) * 4) as u64;
        assert_eq!(dense.step_bytes_per_token(), grouped);
        let q8 = block(e, h, inter, SCALED, Some(QuantSpec::q8()));
        assert_eq!(q8.step_bytes_per_token(), grouped);
    }

    /// The graph refusal and `forward` make one decision: a full decode batch
    /// ([`DEVICE_DISPATCH_MAX_ROWS`] tokens) through 8 experts at k = 2 — more (token, slot) pairs
    /// than experts — is dispatched on the device and not refused; a prefill-sized step is not.
    #[test]
    fn the_graph_refusal_matches_the_dispatch_forward_takes() {
        let dense = block(8, 64, 64, NORM, None);
        let decode = randn(&[1, DEVICE_DISPATCH_MAX_ROWS, 64], &mut SplitMix64::new(4));
        assert_eq!(dense.graph_refusal(), None);
        assert_eq!(
            with_device_dispatch(|| step_counts(&dense, &decode)),
            (0, 1)
        );
        let prefill = randn(
            &[1, DEVICE_DISPATCH_MAX_ROWS + 1, 64],
            &mut SplitMix64::new(5),
        );
        assert_eq!(
            with_device_dispatch(|| step_counts(&dense, &prefill)),
            (1, 0)
        );
    }

    /// Timing, not a gate: the stacked device dispatch against the grouped dispatch on the CPU at
    /// a real-model-shaped layer (64 experts, k = 8, hidden 1024, expert FFN 512). Run by hand:
    ///
    /// ```text
    /// cargo test -p candle-llm --release --lib moe::tests::stacked_vs_grouped_cpu_timing \
    ///     -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "timing; run by hand with --release"]
    fn stacked_vs_grouped_cpu_timing() {
        let (e, h, inter, k) = (64, 1024, 512, 8);
        let routing = MoeRouting {
            experts_per_tok: k,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        };
        let moe = block(e, h, inter, routing, None);
        let Layout::Stacked { gate, up, down } = &moe.bank.layout else {
            panic!("a dense bank stacks")
        };
        let median_ms = |f: &dyn Fn()| {
            f();
            let mut v: Vec<f64> = (0..15)
                .map(|_| {
                    let at = std::time::Instant::now();
                    f();
                    at.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        for t in [1, 4, 8] {
            let x = randn(&[t, h], &mut SplitMix64::new(t as u64));
            let (ids, weights) = moe.route(&x).unwrap();
            let stacked = median_ms(&|| {
                dispatch_stacked(&x, &ids, &weights, [gate, up, down]).unwrap();
            });
            let grouped = median_ms(&|| {
                moe.dispatch_grouped(&x, &ids, &weights).unwrap();
            });
            eprintln!(
                "t={t} E={e} k={k} h={h} ffn={inter}: stacked {stacked:.3} ms, grouped \
                 {grouped:.3} ms ({:.2}x)",
                stacked / grouped
            );
        }
    }

    /// Every device→host read in this module goes through [`read_routes`], so the host-sync
    /// counter cannot miss one: no host-read spelling (method, UFCS or a move to the CPU) appears
    /// anywhere else. And the device path — [`SparseMoe::route`] and [`dispatch_stacked`] — calls
    /// only the device tensor ops named here, so a host read hidden in a helper (in this module or
    /// another) cannot enter it either. A new op on that path is added to `DEVICE_OPS` on purpose.
    #[test]
    fn host_reads_go_through_read_routes() {
        const RAW_READS: [&str; 9] = [
            "to_vec0",
            "to_vec1",
            "to_vec2",
            "to_vec3",
            "to_scalar",
            "to_device",
            "Device::Cpu",
            "storage_and_layout",
            "to_cpu_storage",
        ];
        const DEVICE_PATH: [&str; 4] = [
            "route",
            "dispatch_stacked",
            "dispatch_indexed",
            "project_indexed",
        ];
        // `bank.forward` is `IndexedExperts::forward` (candle-quant-kernels): kernel launches
        // over device tables, no host read — audited, and listed on purpose by its receiver.
        const DEVICE_OPS: [&str; 36] = [
            "Ok",
            "f64::from",
            "Tensor::zeros",
            "candle_nn::ops::softmax_last_dim",
            "swiglu",
            "project",
            "top_k",
            "matmul",
            "t",
            "to_dtype",
            "arg_sort_last_dim",
            "sort_last_dim",
            "narrow",
            "contiguous",
            "gather",
            "broadcast_div",
            "broadcast_mul",
            "broadcast_as",
            "dims2",
            "dim",
            "dtype",
            "device",
            "flatten_all",
            "unsqueeze",
            "squeeze",
            "reshape",
            "index_select",
            "transpose",
            "project_indexed",
            "format",
            "io_dtype",
            "ok_or_else",
            "Error::Msg",
            "bank.forward",
            "map_err",
            "crate::primitives::nvfp4_path::note_gemv",
        ];
        // The callee of every `name(` / `path::name(` / `name::<T>(` on a line — a method call
        // qualified by its receiver, `receiver.name`, when the receiver is a plain identifier on
        // this line or (a chained call starting the line) ends the previous code line.
        fn calls(line: &str, previous: &str) -> Vec<String> {
            let b = line.as_bytes();
            let mut out = Vec::new();
            for (i, _) in line.match_indices('(') {
                let mut end = i;
                if end > 0 && b[end - 1] == b'>' {
                    // A turbofish: step back over `::<…>`.
                    match line[..end].rfind("::<") {
                        Some(at) => end = at,
                        None => continue,
                    }
                }
                let start = line[..end]
                    .rfind(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
                    .map_or(0, |at| at + 1);
                let name = &line[start..end];
                if !name.is_empty() && !name.starts_with(':') {
                    let ident = |text: &str| {
                        let at = text
                            .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
                            .map_or(0, |at| at + 1);
                        text[at..].to_owned()
                    };
                    let receiver = match line[..start].strip_suffix('.') {
                        Some(before) if !before.trim().is_empty() => ident(before),
                        Some(_) => ident(previous.trim_end()),
                        None => String::new(),
                    };
                    out.push(match receiver.is_empty() {
                        true => name.to_owned(),
                        false => format!("{receiver}.{name}"),
                    });
                }
            }
            out
        }
        let source = include_str!("moe.rs");
        let code = &source[..source
            .find("#[cfg(test)]\nmod tests {")
            .expect("tests marker")];
        let mut current_fn = "";
        let mut raw = Vec::new();
        let mut previous = "";
        for (n, line) in code.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            let signature = trimmed.find("fn ").map(|at| {
                current_fn = trimmed[at + 3..]
                    .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                    .next()
                    .unwrap_or_default();
            });
            if RAW_READS.iter().any(|r| line.contains(r)) && current_fn != "read_routes" {
                raw.push(format!("line {}: {}", n + 1, line.trim()));
            }
            let code_part = line.split("//").next().unwrap_or_default();
            if DEVICE_PATH.contains(&current_fn) && signature.is_none() {
                for callee in calls(code_part, previous) {
                    // A method call passes by its bare name or, for a receiver-specific entry
                    // (`bank.forward`), only by its qualified one.
                    let bare = callee.rsplit('.').next().unwrap_or_default();
                    if !DEVICE_OPS.contains(&callee.as_str()) && !DEVICE_OPS.contains(&bare) {
                        raw.push(format!("line {}: `{callee}` on the device path", n + 1));
                    }
                }
            }
            previous = code_part;
        }
        assert!(raw.is_empty(), "uncounted host reads:\n{}", raw.join("\n"));
    }
}

/// The indexed dispatch on a CUDA device (sc-24440): for every expert format a bank can hold, a
/// decode step routes and runs every expert on the device — indexed, no host read, nothing
/// gathered — and agrees with the grouped per-expert dispatch: bit for bit at one token for GGML
/// and NVFP4 (the kernels are those projections' own decode forwards), to rounding otherwise.
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::primitives::host_sync::host_sync_count;
    use crate::primitives::projection::{ProjectionFormat, QuantSpec};
    use crate::primitives::quant::QuantizedLinear;
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    use candle_core::quantized::GgmlDType;

    #[derive(Clone, Copy, Debug)]
    enum Fmt {
        Dense,
        Ggml(GgmlDType),
        MlxQ8,
        Nvfp4,
    }

    fn randn(shape: &[usize], rng: &mut SplitMix64, dev: &Device, dtype: DType) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
        Tensor::from_vec(data, shape, dev)
            .unwrap()
            .to_dtype(dtype)
            .unwrap()
    }

    fn projection(
        rows: usize,
        cols: usize,
        rng: &mut SplitMix64,
        dev: &Device,
        dtype: DType,
        fmt: Fmt,
    ) -> Projection {
        let w = randn(&[rows, cols], rng, dev, dtype);
        match fmt {
            Fmt::Dense => Projection::load(w, None).unwrap(),
            Fmt::Ggml(d) => Projection::Quantized(QuantizedLinear::quantize(&w, d, None).unwrap()),
            Fmt::MlxQ8 => {
                let group = 32;
                let words: Vec<u32> = (0..rows * cols / 4)
                    .map(|_| (rng.next_f32() * u32::MAX as f32) as u32)
                    .collect();
                let scales = randn(&[rows, cols / group], rng, &Device::Cpu, DType::F32)
                    .affine(0.01, 0.0)
                    .unwrap();
                let biases = randn(&[rows, cols / group], rng, &Device::Cpu, DType::F32);
                Projection::load_mlx_affine_q8(
                    &Tensor::from_vec(words, (rows, cols / 4), &Device::Cpu).unwrap(),
                    &scales,
                    &biases,
                    None,
                    QuantSpec::from_bits_and_group_size(8, group).unwrap(),
                    dev,
                )
                .unwrap()
            }
            Fmt::Nvfp4 => {
                let format = ProjectionFormat::nvfp4(dev).unwrap();
                Projection::load_as(w, None, Some(&format)).unwrap()
            }
        }
    }

    fn block(dev: &Device, dtype: DType, h: usize, inter: usize, fmt: Fmt) -> SparseMoe {
        let e = 8;
        let mut rng = SplitMix64::new(0x2444_0300);
        let router = randn(&[e, h], &mut rng, dev, dtype);
        let swiglu = |rng: &mut SplitMix64, fmt: Fmt| SwiGlu {
            gate: projection(inter, h, rng, dev, dtype, fmt),
            up: projection(inter, h, rng, dev, dtype, fmt),
            down: projection(h, inter, rng, dev, dtype, fmt),
        };
        let experts = (0..e).map(|_| swiglu(&mut rng, fmt)).collect();
        let shared = swiglu(&mut rng, Fmt::Dense);
        let gate = randn(&[1, h], &mut rng, dev, dtype);
        SparseMoe::new(router, experts, shared, Some(gate), WIDE_K).unwrap()
    }

    const WIDE_K: MoeRouting = MoeRouting {
        experts_per_tok: 3,
        norm_topk_prob: true,
        routed_scaling_factor: 1.0,
    };

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    #[test]
    fn every_expert_format_is_dispatched_indexed_and_matches_the_grouped_dispatch() {
        let dev = crate::device::new_cuda_for_test().expect("a CUDA device");
        let sm120 = ProjectionFormat::nvfp4(&dev).is_ok();
        let cases: Vec<(Fmt, DType, usize, usize)> = vec![
            (Fmt::Dense, DType::F32, 64, 96),
            (Fmt::Dense, DType::BF16, 64, 96),
            (Fmt::Dense, DType::F16, 64, 96),
            (Fmt::Ggml(GgmlDType::Q8_0), DType::BF16, 64, 96),
            (Fmt::Ggml(GgmlDType::Q4_0), DType::BF16, 64, 96),
            (Fmt::Ggml(GgmlDType::Q4K), DType::BF16, 256, 256),
            (Fmt::Ggml(GgmlDType::Q6K), DType::F32, 256, 256),
            (Fmt::MlxQ8, DType::BF16, 64, 96),
            (Fmt::Nvfp4, DType::BF16, 64, 96),
        ];
        for (fmt, dtype, h, inter) in cases {
            if matches!(fmt, Fmt::Nvfp4) && !sm120 {
                candle_quant_kernels::skip_without_sm120("NVFP4 MoE experts");
                continue;
            }
            let moe = block(&dev, dtype, h, inter, fmt);
            assert_eq!(moe.graph_refusal(), None, "{fmt:?} {dtype:?}");
            assert_eq!(moe.indexed_refusal(), None, "{fmt:?} {dtype:?}");
            // The tables the load priced: within the per-layer bound the provider charges.
            let priced = 3 * moe.num_experts() * candle_quant_kernels::MAX_TABLE_BYTES_PER_EXPERT;
            let tables = moe.indexed_table_bytes();
            assert!(
                tables > 0 && tables <= priced,
                "{fmt:?} {dtype:?}: {tables} table bytes vs {priced} priced"
            );
            for t in [1usize, 3, DEVICE_DISPATCH_MAX_ROWS] {
                let x = randn(&[1, t, h], &mut SplitMix64::new(t as u64 + 9), &dev, dtype);
                let before = (
                    host_sync_count(),
                    moe_indexed_dispatch_count(),
                    moe_expert_gather_count(),
                );
                let indexed = moe.forward(&x).unwrap();
                assert_eq!(
                    (
                        host_sync_count() - before.0,
                        moe_indexed_dispatch_count() - before.1,
                        moe_expert_gather_count() - before.2,
                    ),
                    (0, 1, 0),
                    "{fmt:?} {dtype:?} t={t}: (host syncs, indexed dispatches, gathers)"
                );
                let again = moe.forward(&x).unwrap();
                assert_eq!(host(&indexed), host(&again), "deterministic");
                let grouped = with_grouped_dispatch(|| moe.forward(&x).unwrap());
                let (a, b) = (host(&indexed), host(&grouped));
                let exact = t == 1 && matches!(fmt, Fmt::Ggml(_) | Fmt::Nvfp4);
                if exact {
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&a), bits(&b), "{fmt:?} {dtype:?}: indexed vs grouped");
                } else {
                    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let tol = match dtype {
                        DType::F32 => 1e-5,
                        _ => 2e-2,
                    } * (1.0 + scale);
                    let diff = a
                        .iter()
                        .zip(&b)
                        .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
                    assert!(
                        diff <= tol,
                        "{fmt:?} {dtype:?} t={t}: indexed vs grouped differ by {diff} (tol {tol})"
                    );
                }
            }
        }
    }

    /// NVFP4's indexed kernel is the decode GEMV's core: with the GEMV switched off every NVFP4
    /// projection runs cuBLASLt, so the bank is dispatched grouped and the refusal names why.
    #[test]
    fn an_nvfp4_bank_follows_the_decode_gemv_switch() {
        let dev = crate::device::new_cuda_for_test().expect("a CUDA device");
        if ProjectionFormat::nvfp4(&dev).is_err() {
            candle_quant_kernels::skip_without_sm120("NVFP4 MoE experts");
            return;
        }
        let moe = block(&dev, DType::BF16, 64, 96, Fmt::Nvfp4);
        let _off = crate::primitives::nvfp4_path::nvfp4_gemv_policy_guard(Some(false));
        assert_eq!(
            moe.graph_refusal(),
            Some("moe_expert_host_dispatch:nvfp4_gemv_disabled")
        );
        assert_eq!(moe.indexed_refusal(), Some("nvfp4_gemv_disabled"));
    }

    /// Admission's per-token bytes on CUDA cover the indexed dispatch's workspace (GGML: the f32
    /// outputs and the Q8_1 activation copies) — more than the grouped estimate.
    #[test]
    fn step_bytes_cover_the_indexed_workspace() {
        let dev = crate::device::new_cuda_for_test().expect("a CUDA device");
        let moe = block(&dev, DType::BF16, 64, 96, Fmt::Ggml(GgmlDType::Q8_0));
        let Ok(banks) = &moe.bank.indexed else {
            panic!("a CUDA Q8_0 bank is indexed")
        };
        let slots = WIDE_K.experts_per_tok;
        let workspace: usize = banks
            .iter()
            .zip([MoeRows::PerToken, MoeRows::PerToken, MoeRows::PerSlot])
            .map(|(b, r)| b.workspace_bytes(DType::BF16, 1, slots, r))
            .sum();
        assert!(moe.step_bytes_per_token() as usize > workspace);
        // A stacked dense bank on the device may also run gathered (its indexed kernel refused):
        // admission prices those copies too.
        let (h, inter) = (64, 96);
        let dense = block(&dev, DType::BF16, h, inter, Fmt::Dense);
        let gathered = slots * 3 * inter * h * 2;
        assert!(dense.step_bytes_per_token() as usize >= gathered);
    }
}

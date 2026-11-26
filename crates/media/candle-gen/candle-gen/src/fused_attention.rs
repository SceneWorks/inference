//! Fused-attention dispatch (sc-26691, SW-6130): the one decision point that routes an eligible
//! attention through the **FlashAttention-2** CUDA kernel, and everything else through the
//! materialized, query-chunked path in [`crate::attention`].
//!
//! # One dispatch point for every model (SW-6130)
//!
//! The four shared helpers — [`crate::attention::sdpa_budgeted_bhsd`],
//! [`crate::attention::sdpa_planned_bhsd`], [`crate::attention::sdpa_budgeted_flat`] and
//! [`crate::attention::sdpa_planned_flat`] — consult [`route_attention_kernel`] on every call, so
//! every provider that routes its attention through them takes the fused kernel when it can,
//! without a per-model opt-in. The rule is, in order:
//!
//! 1. the caller did not ask for the materialized path (the `*_materialized` entry points, the
//!    explicit opt-out for non-standard softmax, oracle tests and training-only code);
//! 2. the build compiled the kernel (`candle-gen/flash-attn`);
//! 3. the process-wide toggle ([`set_flash_attention_enabled`], default on) is on;
//! 4. no gradient can flow through the call: no operand is on the autograd tape
//!    ([`Tensor::track_op`]), and the thread is not inside a training scope
//!    ([`with_materialized_attention`], entered by the shared training-job and gradient-checkpoint
//!    drivers, whose detached stash forward would otherwise hide the tape from rule 4's first half);
//! 5. [`select_attention_kernel`] accepts the operands (device, arch, dtype, mask, shape, ABI).
//!
//! Each call increments exactly one process-wide counter — the fused kernel, or the reason it was
//! refused — readable through [`attention_path_counts`] / [`take_attention_path_counts`].
//!
//! ## The softmax contract
//!
//! The fused kernel computes the **plain last-axis softmax** of `q·kᵀ·scale` (in f32, with the
//! probabilities rounded to the value dtype for `P·V`). When it is taken, the caller's `softmax`
//! closure is **not called**. So the flash-eligible helpers are only correct for a closure that is
//! `softmax_last_dim`, the composable `softmax(_, D::Minus1)`, or an f32-upcast-then-cast-back
//! variant of either. Any other closure — sinks, clamps, a temperature, masking or scaling inside
//! the closure, a counting or sabotaging test wrapper — must call a `*_materialized` entry point.
//!
//! # Why this exists
//!
//! [`crate::attention`]'s budgeted path materializes the `[B, H, block, Sk]` scores tensor. That is
//! the right guard for the i32 indexing hazard, but it is the wrong *kernel* for a very long joint
//! sequence: at MiniMax-H3's ~141k-row packed sequence (56 heads, `head_dim` 128) the 1e9 budget
//! splits every layer into ~1,100 query chunks, each re-reading all of K and V, which measured at
//! ≈6+ minutes per DiT forward. FlashAttention streams the scores through on-chip memory, so no
//! `[.., Sq, Sk]` tensor exists at all and the i32 hazard the budget guards never arises.
//!
//! # Selection is pure, and the refusals are typed
//!
//! [`flash_eligibility`] is a pure function of the build, the device kind, the dtype, the shapes and
//! whether a mask is present, so every refusal is unit-testable on the CPU lane without a CUDA
//! toolkit. The kernel is taken only when **all** of these hold:
//!
//! - the build enables `candle-gen/flash-attn` (it compiles CUTLASS kernels and needs nvcc);
//! - the tensors live on a CUDA device **whose compute capability the compiled fatbin serves**:
//!   at least sm_8.0 (FlashAttention-2's floor) and covered by the vendored
//!   `vendor/candle-flash-attn` `-gencode` ladder (SASS within one major, PTX forward), read back
//!   through `build.rs` ([`crate::cuda_arch::FatbinLadder::flash_from_build`]). A cubin-only
//!   kernel with no image for the device would not error — it would leave the output unwritten —
//!   so an uncovered device always takes the budgeted path. The capability is read once per GPU
//!   ordinal and cached;
//! - the dtype is bf16 or f16 (the kernel has no f32 path — f32 callers keep their numerics), and
//!   q, k and v share it and share one device (the kernel reads all three as one element type);
//! - there is **no additive mask** — the wrapper exposes only dense / causal, and an arbitrary
//!   additive bias cannot be expressed, so a masked call keeps the budgeted path;
//! - q/k/v are rank 4, share the batch, K and V share their shape, the query heads are a multiple of
//!   the KV heads, and the common `head_dim` is a multiple of 8 and at most
//!   [`FLASH_MAX_HEAD_DIM`];
//! - every stride and extent the kernel receives fits its `u32` parameter ABI: the batch / row /
//!   head strides of the `[B, S, H, D]` views `flash_bhsd` actually hands it (the operand's own
//!   strides when its last dim is unit-stride, else those of the contiguous copy), and the
//!   contiguous output's `Sq·H·D` batch stride. candle-flash-attn's dense path converts these
//!   checked too, but refusing here keeps an oversized call on the budgeted path instead of
//!   failing it.
//!
//! # Stream ordering
//!
//! candle-flash-attn launches on CUDA stream 0 (the legacy default stream). candle-gen's devices
//! are built by `Device::new_cuda`, which uses the context's default stream, so the kernel is
//! ordered against every other op the media lane issues. A device created on its *own*
//! non-blocking stream (`Device::new_cuda_with_stream`, which only candle-llm's graph runner uses)
//! would NOT be ordered against it; no media provider builds one.
//!
//! # Cancellation
//!
//! The fused kernel is one launch per call, so a cancel cannot land *inside* it. The planned
//! helpers check the plan's flag once before launching (returning [`crate::CandleError::Canceled`]);
//! the budgeted path keeps its between-chunk checks. Long sequences should additionally check
//! between layers, which is where a fused forward's cancel latency is bounded.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use candle_core::{DType, Tensor};

use crate::attention::{sdpa_budgeted_bhsd, sdpa_planned_bhsd, AttentionPlan};

/// Candidate-only seam for callers owning a separate replay/tape fallback.
/// Select outside the tape fold: `Some` carries the native custom backward;
/// `None` means reference mode, unsupported input, or cached compile refusal.
/// Allocation and execution errors propagate. The caller retains its existing
/// cancellation checks and must not wrap a successful candidate in another
/// autograd owner. Standard unmasked SDPA only, with no dropout or softcap.
pub fn try_training_sdpa_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
) -> candle_core::Result<Option<Tensor>> {
    crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())
}

/// First-order training SDPA. The experimental streaming candidate is explicitly
/// scoped by [`crate::training_attention::with_training_attention_mode`]. Callers
/// must supply ordinary last-axis softmax (possibly upcast to f32), with no
/// dropout, softcap or other score transform. Masks always retain the reference.
/// Cancellation is checked before launch; a native launch cannot be interrupted.
///
/// Several providers also run their *inference* attention through this entry point, so the
/// fallback is the flash-eligible [`sdpa_planned_bhsd`]: under autograd (an operand on the tape)
/// or inside a training scope ([`with_materialized_attention`]) the router refuses the fused
/// kernel, so a training step is exactly the materialized path it always was.
pub fn training_sdpa_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    plan: AttentionPlan<'_>,
) -> crate::Result<Tensor> {
    if plan.is_cancelled() {
        return Err(crate::CandleError::Canceled);
    }
    if let Some(out) = crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())? {
        return Ok(out);
    }
    sdpa_planned_bhsd(q, k, v, scale, mask, softmax, plan)
}

/// Legacy-budget overload of [`training_sdpa_bhsd`], retaining the exact
/// composable fallback. GQA inputs must be expanded by the caller when the
/// reference requires it; the candidate itself can sum grouped KV gradients.
/// The fallback routes like [`training_sdpa_bhsd`]'s: fused only when no gradient can flow.
pub fn training_sdpa_budgeted_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    softmax: impl Fn(&Tensor) -> candle_core::Result<Tensor>,
    budget: usize,
) -> candle_core::Result<Tensor> {
    if let Some(out) = crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())? {
        return Ok(out);
    }
    sdpa_budgeted_bhsd(q, k, v, scale, mask, softmax, budget)
}

/// The most score elements one [`training_sdpa_standard_bhsd`] call's reference owner holds at
/// once, in its forward and in its backward recompute (its query tiles never exceed it, whatever
/// the caller's budget): the bound a trainer's memory estimate prices training attention at.
pub const TRAINING_SCORE_TILE_ELEMENTS: u64 =
    crate::reference_training_attention::REFERENCE_SCORE_TILE_ELEMENTS;

/// Known standard composable-softmax training contract. Reference uses a single
/// dense autograd owner instead of retaining every score intermediate per layer.
/// Arbitrary softmax callbacks must use `training_sdpa_bhsd` above. Training-only, so the final
/// fallback is the always-materialized [`crate::attention::sdpa_planned_bhsd_materialized`].
pub fn training_sdpa_standard_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    mask: Option<&Tensor>,
    plan: AttentionPlan<'_>,
) -> crate::Result<Tensor> {
    if plan.is_cancelled() {
        return Err(crate::CandleError::Canceled);
    }
    if let Some(out) = crate::training_attention::try_streaming(q, k, v, scale, mask.is_some())? {
        return Ok(out);
    }
    if let Some(out) = crate::reference_training_attention::try_forward(q, k, v, scale, mask, plan)?
    {
        return Ok(out);
    }
    crate::attention::sdpa_planned_bhsd_materialized(
        q,
        k,
        v,
        scale,
        mask,
        |s| candle_nn::ops::softmax(s, candle_core::D::Minus1),
        plan,
    )
}

/// Largest `head_dim` the dispatcher hands to the fused kernel. candle-flash-attn compiles
/// kernels up to 512 at this rev, but the 256 ceiling is the long-standing FA2 envelope and covers
/// every media head dim in the tree (MiniMax-H3's DiT is 128, the Qwen3-VL tower 72).
pub const FLASH_MAX_HEAD_DIM: usize = 256;

/// `true` when this build compiled the fused kernel in (`candle-gen/flash-attn`).
pub const FLASH_ATTN_COMPILED: bool = cfg!(feature = "flash-attn");

/// Why one attention call did not take the fused kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashRefusal {
    /// The build does not enable `candle-gen/flash-attn`.
    NotCompiled,
    /// The tensors are not on a CUDA device.
    NotCuda,
    /// The CUDA device's compute capability (`None`: unreadable) is below sm_8.0 or not served by
    /// the compiled flash fatbin.
    UnsupportedArch(Option<u32>),
    /// The dtype has no fused kernel (only bf16 / f16 do).
    Dtype(DType),
    /// The call carries an additive mask, which the fused kernel cannot express.
    Masked,
    /// The `head_dim` is not a multiple of 8, exceeds [`FLASH_MAX_HEAD_DIM`], or differs across
    /// q / k / v.
    HeadDim(usize),
    /// Rank, batch, K/V shape or head-group mismatch.
    Shape,
    /// q, k and v do not share one dtype.
    MixedDtype,
    /// q, k and v do not live on one device.
    MixedDevice,
    /// A stride or extent the kernel would receive exceeds `u32::MAX` (its parameter ABI).
    IndexOverflow,
    /// The process-wide toggle ([`set_flash_attention_enabled`]) is off.
    Disabled,
    /// The caller asked for the materialized path: a `*_materialized` entry point, or a call
    /// inside a training scope ([`with_materialized_attention`]).
    Materialized,
    /// An operand is on the autograd tape, and the fused kernel has no backward.
    GradTracked,
}

/// One attention operand as [`flash_eligibility`] sees it: dtype, `[B, H, S, D]` dims and element
/// strides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashOperand<'a> {
    /// The element type.
    pub dtype: DType,
    /// The `[B, H, S, D]` dims.
    pub dims: &'a [usize],
    /// The element strides, one per dim.
    pub strides: &'a [usize],
}

impl<'a> FlashOperand<'a> {
    /// The operand view of a real tensor.
    pub fn of(t: &'a Tensor) -> Self {
        Self {
            dtype: t.dtype(),
            dims: t.dims(),
            strides: t.stride(),
        }
    }
}

/// The `(batch, row, head)` element strides `flash_bhsd` hands the kernel for one `[B, H, S, D]`
/// operand: the `[B, S, H, D]` transpose's own strides when the last dim is unit-stride, else
/// those of the contiguous copy it makes. `None` when the copy's strides overflow `usize`.
fn kernel_strides(dims: [usize; 4], strides: &[usize]) -> Option<[usize; 3]> {
    let [_, h, s, d] = dims;
    match *strides {
        [sb, sh, ss, 1] => Some([sb, ss, sh]),
        _ => {
            let hd = h.checked_mul(d)?;
            Some([s.checked_mul(hd)?, hd, d])
        }
    }
}

/// Whether every value the kernel's `u32` parameter ABI would carry for this call fits.
fn fits_kernel_abi(
    q: [usize; 4],
    k: [usize; 4],
    q_strides: &[usize],
    k_strides: &[usize],
    v_strides: &[usize],
) -> bool {
    let fits = |x: usize| u32::try_from(x).is_ok();
    let strides_fit =
        |dims, strides| kernel_strides(dims, strides).is_some_and(|s| s.into_iter().all(fits));
    let [b, h, sq, d] = q;
    let [_, kh, sk, _] = k;
    // The output is a fresh contiguous `[B, Sq, H, D]`: batch stride `Sq·H·D`, row `H·D`.
    let out_batch = h.checked_mul(d).and_then(|hd| hd.checked_mul(sq));
    // The kernel also takes the 128-rounded sequence lengths.
    let rounded = |x: usize| x.checked_next_multiple_of(128);
    strides_fit(q, q_strides)
        && strides_fit(k, k_strides)
        && strides_fit(k, v_strides)
        && out_batch.is_some_and(fits)
        && rounded(sq).is_some_and(fits)
        && rounded(sk).is_some_and(fits)
        && [b, h, kh, d].into_iter().all(fits)
}

/// The kernel one call takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKernel {
    /// candle-flash-attn's fused FA2 forward — scores never materialize.
    Flash,
    /// The materialized, query-chunked [`sdpa_planned_bhsd`], with the reason the fused kernel was
    /// refused.
    Budgeted(FlashRefusal),
}

impl AttentionKernel {
    /// A stable short name for logs.
    pub fn name(self) -> &'static str {
        match self {
            AttentionKernel::Flash => "flash_attn_v2",
            AttentionKernel::Budgeted(_) => "budgeted_sdpa",
        }
    }
}

/// The device side of [`flash_eligibility`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashDevice {
    /// CPU / Metal.
    NotCuda,
    /// A CUDA device with its packed compute capability (`120` = sm_12.0), `None` if unreadable,
    /// and whether [`flash_arch_supported`] accepted it.
    Cuda {
        compute_cap: Option<u32>,
        supported: bool,
    },
}

/// FlashAttention-2's architecture floor: sm_8.0 (Ampere).
pub const FLASH_MIN_COMPUTE_CAP: u32 = 80;

/// Whether the fused kernel may launch on a device of `compute_cap` given the compiled `ladder`:
/// the capability is known, at least [`FLASH_MIN_COMPUTE_CAP`], and covered by a non-empty ladder.
/// Pure — the unit tests drive it with synthetic ladders.
pub fn flash_arch_supported(
    compute_cap: Option<u32>,
    ladder: &crate::cuda_arch::FatbinLadder,
) -> bool {
    match compute_cap {
        Some(cap) => cap >= FLASH_MIN_COMPUTE_CAP && !ladder.is_empty() && ladder.covers(cap),
        None => false,
    }
}

/// The pure selection rule (see the module docs). `q`/`k`/`v` carry the `[B, H, S, D]` dims and
/// strides; `device` classifies q's device and `same_device` says whether k and v share it.
pub fn flash_eligibility(
    compiled: bool,
    device: FlashDevice,
    same_device: bool,
    q: FlashOperand<'_>,
    k: FlashOperand<'_>,
    v: FlashOperand<'_>,
    masked: bool,
) -> AttentionKernel {
    use AttentionKernel::Budgeted;
    if !compiled {
        return Budgeted(FlashRefusal::NotCompiled);
    }
    match device {
        FlashDevice::NotCuda => return Budgeted(FlashRefusal::NotCuda),
        FlashDevice::Cuda {
            compute_cap,
            supported: false,
        } => return Budgeted(FlashRefusal::UnsupportedArch(compute_cap)),
        FlashDevice::Cuda { .. } => {}
    }
    if !matches!(q.dtype, DType::BF16 | DType::F16) {
        return Budgeted(FlashRefusal::Dtype(q.dtype));
    }
    if k.dtype != q.dtype || v.dtype != q.dtype {
        return Budgeted(FlashRefusal::MixedDtype);
    }
    if !same_device {
        return Budgeted(FlashRefusal::MixedDevice);
    }
    if masked {
        return Budgeted(FlashRefusal::Masked);
    }
    let (qdims @ [qb, qh, qs, qd], kdims @ [kb, kh, ks, kd]) = match (q.dims, k.dims) {
        (&[a, b, c, d], &[e, f, g, h]) => ([a, b, c, d], [e, f, g, h]),
        _ => return Budgeted(FlashRefusal::Shape),
    };
    if v.dims.len() != 4
        || qb != kb
        || k.dims != v.dims
        || qs == 0
        || ks == 0
        || kh == 0
        || qh % kh != 0
        || [q.strides.len(), k.strides.len(), v.strides.len()] != [4; 3]
    {
        return Budgeted(FlashRefusal::Shape);
    }
    if qd != kd || qd % 8 != 0 || qd == 0 || qd > FLASH_MAX_HEAD_DIM {
        return Budgeted(FlashRefusal::HeadDim(qd));
    }
    if !fits_kernel_abi(qdims, kdims, q.strides, k.strides, v.strides) {
        return Budgeted(FlashRefusal::IndexOverflow);
    }
    AttentionKernel::Flash
}

/// [`flash_eligibility`] for real tensors in this build.
pub fn select_attention_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> AttentionKernel {
    flash_eligibility(
        FLASH_ATTN_COMPILED,
        flash_device(q.device()),
        q.device().same_device(k.device()) && q.device().same_device(v.device()),
        FlashOperand::of(q),
        FlashOperand::of(k),
        FlashOperand::of(v),
        mask.is_some(),
    )
}

/// Classify a real device for [`flash_eligibility`]. The compute capability is cached per GPU
/// ordinal ([`crate::cuda_arch::device_compute_cap`]) and the baked ladder parsed once, so this is
/// a couple of lookups per attention call. Skipped entirely when the kernel is not compiled.
pub fn flash_device(device: &candle_core::Device) -> FlashDevice {
    if !device.is_cuda() {
        return FlashDevice::NotCuda;
    }
    if !FLASH_ATTN_COMPILED {
        return FlashDevice::Cuda {
            compute_cap: None,
            supported: false,
        };
    }
    static LADDER: std::sync::OnceLock<crate::cuda_arch::FatbinLadder> = std::sync::OnceLock::new();
    let ladder = LADDER.get_or_init(crate::cuda_arch::FatbinLadder::flash_from_build);
    let compute_cap = crate::cuda_arch::device_compute_cap(device);
    FlashDevice::Cuda {
        compute_cap,
        supported: flash_arch_supported(compute_cap, ladder),
    }
}

// ---------------------------------------------------------------------------------------------
// The process-wide toggle (SW-6130 E3).
// ---------------------------------------------------------------------------------------------

static FLASH_ATTENTION_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable the fused attention kernel for every subsequent call in this process (the
/// one toggle; default **on**). Off, every shared helper takes the materialized path and counts
/// [`FlashRefusal::Disabled`]. It has no effect on a build without `candle-gen/flash-attn`.
///
/// Read per call, so flipping it between requests takes effect at the next attention call — no
/// pipeline reload. The SDXL `set_flash_attn` and Z-Image `set_accel_attn` knobs are thin wrappers
/// over this one switch.
pub fn set_flash_attention_enabled(on: bool) {
    FLASH_ATTENTION_ENABLED.store(on, Ordering::Relaxed);
}

/// Whether the fused attention kernel is currently allowed ([`set_flash_attention_enabled`]).
pub fn flash_attention_enabled() -> bool {
    FLASH_ATTENTION_ENABLED.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------------------------
// The training scope (SW-6130 E2/E5).
// ---------------------------------------------------------------------------------------------

thread_local! {
    static MATERIALIZED_SCOPE: Cell<u32> = const { Cell::new(0) };
}

/// Run `f` with every shared attention helper on this thread forced onto the materialized path
/// (counted as [`FlashRefusal::Materialized`]). Nests, and restores on unwind.
///
/// The autograd check ([`Tensor::track_op`]) alone cannot see a training forward that runs on
/// *detached* operands and recomputes later under the tape — the gradient-checkpoint stash forward,
/// a tape fold, a custom-backward reference owner. The shared training-job scope
/// (`train::policy::with_job_policy`) and the checkpointed backward driver enter this scope, so a
/// training step is the materialized path it always was, bit for bit.
pub fn with_materialized_attention<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(u32);
    impl Drop for Restore {
        fn drop(&mut self) {
            MATERIALIZED_SCOPE.set(self.0);
        }
    }
    let _restore = Restore(MATERIALIZED_SCOPE.replace(MATERIALIZED_SCOPE.get().saturating_add(1)));
    f()
}

/// Whether this thread is inside [`with_materialized_attention`].
pub fn in_materialized_attention_scope() -> bool {
    MATERIALIZED_SCOPE.get() > 0
}

// ---------------------------------------------------------------------------------------------
// The router (SW-6130 E1).
// ---------------------------------------------------------------------------------------------

/// The per-call inputs of the router that are not properties of the operands' shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashGate {
    /// The caller asked for the materialized path (a `*_materialized` entry point or a training
    /// scope).
    pub materialized: bool,
    /// The process-wide toggle ([`flash_attention_enabled`]).
    pub enabled: bool,
    /// Some operand is on the autograd tape.
    pub grad_tracked: bool,
}

/// The pure front half of the router: the refusal that applies before any operand inspection, or
/// `None` when the call may go on to [`flash_eligibility`]. The order is the module-level rule's.
pub fn flash_gate_refusal(gate: FlashGate, compiled: bool) -> Option<FlashRefusal> {
    if gate.materialized {
        Some(FlashRefusal::Materialized)
    } else if !compiled {
        Some(FlashRefusal::NotCompiled)
    } else if !gate.enabled {
        Some(FlashRefusal::Disabled)
    } else if gate.grad_tracked {
        Some(FlashRefusal::GradTracked)
    } else {
        None
    }
}

/// The full pure router: [`flash_gate_refusal`], then [`flash_eligibility`].
#[allow(clippy::too_many_arguments)]
pub fn flash_route(
    gate: FlashGate,
    compiled: bool,
    device: FlashDevice,
    same_device: bool,
    q: FlashOperand<'_>,
    k: FlashOperand<'_>,
    v: FlashOperand<'_>,
    masked: bool,
) -> AttentionKernel {
    match flash_gate_refusal(gate, compiled) {
        Some(refusal) => AttentionKernel::Budgeted(refusal),
        None => flash_eligibility(compiled, device, same_device, q, k, v, masked),
    }
}

/// The kernel a flash-eligible shared helper takes for these `[B, H, S, D]` operands right now:
/// [`flash_route`] for real tensors, the toggle, the training scope and the autograd state. Pure
/// with respect to the counters — the helpers count; callers that only need to *know* the route
/// (MiniMax-H3's execution receipt) may call this freely.
pub fn route_attention_kernel(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> AttentionKernel {
    let gate = FlashGate {
        materialized: in_materialized_attention_scope(),
        enabled: flash_attention_enabled(),
        grad_tracked: q.track_op() || k.track_op() || v.track_op(),
    };
    match flash_gate_refusal(gate, FLASH_ATTN_COMPILED) {
        Some(refusal) => AttentionKernel::Budgeted(refusal),
        // The device classification only runs once the cheap gate passed.
        None => select_attention_kernel(q, k, v, mask),
    }
}

/// Route one helper call and count it. `true` = take the fused kernel.
pub(crate) fn take_flash(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> bool {
    let kernel = route_attention_kernel(q, k, v, mask);
    record_attention_path(kernel);
    kernel == AttentionKernel::Flash
}

/// Count one explicitly materialized helper call.
pub(crate) fn record_materialized_call() {
    record_attention_path(AttentionKernel::Budgeted(FlashRefusal::Materialized));
}

// ---------------------------------------------------------------------------------------------
// The path counters (SW-6130 E4).
// ---------------------------------------------------------------------------------------------

/// Process-wide count of shared-helper attention calls by the path they took: the fused kernel,
/// or the budgeted path with the reason the fused kernel was refused. A cheap `Copy` snapshot —
/// read it with [`attention_path_counts`], or drain it per job with [`take_attention_path_counts`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AttentionPathCounts {
    /// Calls that ran the FlashAttention-2 kernel.
    pub flash: u64,
    /// Budgeted: the build does not compile the kernel.
    pub not_compiled: u64,
    /// Budgeted: the operands are not on a CUDA device.
    pub not_cuda: u64,
    /// Budgeted: the CUDA device's compute capability is not served.
    pub unsupported_arch: u64,
    /// Budgeted: the dtype has no fused kernel (only bf16 / f16 do).
    pub dtype: u64,
    /// Budgeted: q / k / v disagree on dtype or device.
    pub mixed_dtype_or_device: u64,
    /// Budgeted: the call carries an additive mask.
    pub masked: u64,
    /// Budgeted: rank, batch, K/V shape or head-group mismatch.
    pub shape: u64,
    /// Budgeted: the head dim is unsupported.
    pub head_dim: u64,
    /// Budgeted: a stride or extent exceeds the kernel's `u32` ABI.
    pub index_overflow: u64,
    /// Budgeted: the process-wide toggle is off.
    pub disabled: u64,
    /// Budgeted: an operand is on the autograd tape.
    pub grad_tracked: u64,
    /// Budgeted: the caller asked for the materialized path.
    pub materialized: u64,
}

impl AttentionPathCounts {
    /// Every budgeted (non-fused) call, whatever the reason.
    pub fn budgeted(&self) -> u64 {
        self.not_compiled
            + self.not_cuda
            + self.unsupported_arch
            + self.dtype
            + self.mixed_dtype_or_device
            + self.masked
            + self.shape
            + self.head_dim
            + self.index_overflow
            + self.disabled
            + self.grad_tracked
            + self.materialized
    }

    /// Every counted call.
    pub fn total(&self) -> u64 {
        self.flash + self.budgeted()
    }

    /// The field-wise difference `self - earlier` (saturating), for a window between two snapshots.
    pub fn since(&self, earlier: &Self) -> Self {
        let d = |a: u64, b: u64| a.saturating_sub(b);
        Self {
            flash: d(self.flash, earlier.flash),
            not_compiled: d(self.not_compiled, earlier.not_compiled),
            not_cuda: d(self.not_cuda, earlier.not_cuda),
            unsupported_arch: d(self.unsupported_arch, earlier.unsupported_arch),
            dtype: d(self.dtype, earlier.dtype),
            mixed_dtype_or_device: d(self.mixed_dtype_or_device, earlier.mixed_dtype_or_device),
            masked: d(self.masked, earlier.masked),
            shape: d(self.shape, earlier.shape),
            head_dim: d(self.head_dim, earlier.head_dim),
            index_overflow: d(self.index_overflow, earlier.index_overflow),
            disabled: d(self.disabled, earlier.disabled),
            grad_tracked: d(self.grad_tracked, earlier.grad_tracked),
            materialized: d(self.materialized, earlier.materialized),
        }
    }
}

/// One counter slot per [`AttentionPathCounts`] field, in field order.
const PATH_SLOTS: usize = 13;
static PATH_COUNTS: [AtomicU64; PATH_SLOTS] = [const { AtomicU64::new(0) }; PATH_SLOTS];

/// The counter slot a routed call lands in.
fn path_slot(kernel: AttentionKernel) -> usize {
    use FlashRefusal as R;
    match kernel {
        AttentionKernel::Flash => 0,
        AttentionKernel::Budgeted(reason) => match reason {
            R::NotCompiled => 1,
            R::NotCuda => 2,
            R::UnsupportedArch(_) => 3,
            R::Dtype(_) => 4,
            R::MixedDtype | R::MixedDevice => 5,
            R::Masked => 6,
            R::Shape => 7,
            R::HeadDim(_) => 8,
            R::IndexOverflow => 9,
            R::Disabled => 10,
            R::GradTracked => 11,
            R::Materialized => 12,
        },
    }
}

/// Count one routed call. A single relaxed increment: no logging, no allocation.
pub fn record_attention_path(kernel: AttentionKernel) {
    PATH_COUNTS[path_slot(kernel)].fetch_add(1, Ordering::Relaxed);
}

fn counts_from(read: impl Fn(&AtomicU64) -> u64) -> AttentionPathCounts {
    let c: [u64; PATH_SLOTS] = std::array::from_fn(|i| read(&PATH_COUNTS[i]));
    AttentionPathCounts {
        flash: c[0],
        not_compiled: c[1],
        not_cuda: c[2],
        unsupported_arch: c[3],
        dtype: c[4],
        mixed_dtype_or_device: c[5],
        masked: c[6],
        shape: c[7],
        head_dim: c[8],
        index_overflow: c[9],
        disabled: c[10],
        grad_tracked: c[11],
        materialized: c[12],
    }
}

/// A snapshot of the process-wide attention path counters.
pub fn attention_path_counts() -> AttentionPathCounts {
    counts_from(|a| a.load(Ordering::Relaxed))
}

/// Drain the process-wide attention path counters: return their values and reset them to zero
/// (each slot atomically; a call racing the drain lands in this snapshot or the next, never lost).
pub fn take_attention_path_counts() -> AttentionPathCounts {
    counts_from(|a| a.swap(0, Ordering::Relaxed))
}

// ---------------------------------------------------------------------------------------------
// The fused forward.
// ---------------------------------------------------------------------------------------------

/// `[N, S, D]` as the `[B = N, H = 1, S, D]` view the router and kernel see: a free `unsqueeze`,
/// no copy. The flat helpers fold heads into the batch (SDXL) or run single-head (a VAE mid
/// block), and every one of the `N` rows is its own independent attention, so `N` is the batch.
pub(crate) fn flat_as_bhsd(t: &Tensor) -> candle_core::Result<Tensor> {
    t.unsqueeze(1)
}

/// The pure dims/strides of [`flat_as_bhsd`]'s view, for the routing unit tests.
#[cfg(test)]
fn flat_as_bhsd_layout(dims: [usize; 3], strides: [usize; 3]) -> ([usize; 4], [usize; 4]) {
    let [n, s, d] = dims;
    let [sn, ss, sd] = strides;
    // candle's `unsqueeze(1)` gives the new unit dim the stride of the dim it displaces (`ss`); any
    // stride is valid for an extent-1 dim, and this one keeps a contiguous view contiguous.
    ([n, 1, s, d], [sn, ss, ss, sd])
}

/// The fused forward. The kernel wants `[B, S, H, D]`, which is a free transpose of a `[B, H, S, D]`
/// view (only the last dim must be unit-stride; the batch/row/head strides are passed through).
#[cfg(feature = "flash-attn")]
pub(crate) fn flash_bhsd(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
) -> candle_core::Result<Tensor> {
    let unit_last = |t: &Tensor| -> candle_core::Result<Tensor> {
        let t = t.transpose(1, 2)?;
        if t.stride().last() == Some(&1) {
            Ok(t)
        } else {
            t.contiguous()
        }
    };
    let out = candle_flash_attn::flash_attn(
        &unit_last(q)?,
        &unit_last(k)?,
        &unit_last(v)?,
        scale as f32,
        false,
    )?; // [B, Sq, H, D]
    out.transpose(1, 2)?.contiguous()
}

#[cfg(not(feature = "flash-attn"))]
pub(crate) fn flash_bhsd(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _scale: f64,
) -> candle_core::Result<Tensor> {
    // Unreachable: `flash_eligibility` refuses with `NotCompiled` first.
    candle_core::bail!("candle-gen: the fused attention kernel is not compiled into this build")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::{AttentionBudget, ATTN_SCORES_BUDGET};
    use candle_core::Device;
    use candle_nn::ops::softmax_last_dim;

    const H3: [usize; 4] = [1, 56, 4096, 128];

    const SM120: FlashDevice = FlashDevice::Cuda {
        compute_cap: Some(120),
        supported: true,
    };

    /// Row-major strides of `dims`.
    fn contiguous(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    /// The eligibility of contiguous operands of one dtype on one device.
    fn elig(
        compiled: bool,
        device: FlashDevice,
        dtype: DType,
        q: &[usize],
        k: &[usize],
        v: &[usize],
        masked: bool,
    ) -> AttentionKernel {
        let (qs, ks, vs) = (contiguous(q), contiguous(k), contiguous(v));
        let op = |dims, strides| FlashOperand {
            dtype,
            dims,
            strides,
        };
        flash_eligibility(
            compiled,
            device,
            true,
            op(q, &qs),
            op(k, &ks),
            op(v, &vs),
            masked,
        )
    }

    fn pick(dtype: DType, q: &[usize], k: &[usize], masked: bool) -> AttentionKernel {
        elig(true, SM120, dtype, q, k, k, masked)
    }

    fn shipped_ladder() -> crate::cuda_arch::FatbinLadder {
        // The vendored fatbin at the SceneWorks cap=80 baseline: sm_80/90/120 SASS + compute_120.
        crate::cuda_arch::FatbinLadder::explicit_plus_baseline("90,120", "120", "80")
    }

    #[test]
    fn the_arch_guard_admits_only_devices_the_fatbin_serves() {
        let ladder = shipped_ladder();
        for cap in [80, 86, 89, 90, 120, 121, 130] {
            assert!(flash_arch_supported(Some(cap), &ladder), "sm_{cap}");
        }
        // Below FA2's floor, the datacenter-Blackwell gap (major 10), and an unreadable device.
        for cap in [Some(61), Some(75), Some(100), Some(103), None] {
            assert!(!flash_arch_supported(cap, &ladder), "{cap:?}");
        }
        // A cap=120-only build (no sm_80 rung) must not launch on Ampere.
        let native = crate::cuda_arch::FatbinLadder::explicit_plus_baseline("120", "", "120");
        assert!(!flash_arch_supported(Some(86), &native));
        assert!(flash_arch_supported(Some(120), &native));
        // An empty ladder is "unknown", which for a launch means "no".
        assert!(!flash_arch_supported(
            Some(120),
            &crate::cuda_arch::FatbinLadder::default()
        ));
    }

    #[test]
    fn an_unsupported_arch_refuses_before_any_shape_check() {
        let old = FlashDevice::Cuda {
            compute_cap: Some(75),
            supported: false,
        };
        assert_eq!(
            elig(true, old, DType::BF16, &H3, &H3, &H3, false),
            AttentionKernel::Budgeted(FlashRefusal::UnsupportedArch(Some(75)))
        );
    }

    #[test]
    fn the_h3_dit_shape_takes_the_fused_kernel_in_bf16_and_f16() {
        assert_eq!(pick(DType::BF16, &H3, &H3, false), AttentionKernel::Flash);
        assert_eq!(pick(DType::F16, &H3, &H3, false), AttentionKernel::Flash);
        // Cross-attention: a different key length is fine.
        assert_eq!(
            pick(DType::BF16, &H3, &[1, 56, 300, 128], false),
            AttentionKernel::Flash
        );
        // Grouped-query: KV heads dividing the query heads is fine.
        assert_eq!(
            pick(DType::BF16, &H3, &[1, 8, 4096, 128], false),
            AttentionKernel::Flash
        );
        // The Qwen3-VL tower's 72 is a multiple of 8.
        let vit = [1, 16, 4032, 72];
        assert_eq!(pick(DType::BF16, &vit, &vit, false), AttentionKernel::Flash);
    }

    #[test]
    fn every_refusal_is_typed_and_ordered() {
        use AttentionKernel::Budgeted;
        let f = |compiled, cuda: bool, dtype| {
            let device = if cuda { SM120 } else { FlashDevice::NotCuda };
            elig(compiled, device, dtype, &H3, &H3, &H3, false)
        };
        assert_eq!(
            f(false, true, DType::BF16),
            Budgeted(FlashRefusal::NotCompiled)
        );
        assert_eq!(f(true, false, DType::BF16), Budgeted(FlashRefusal::NotCuda));
        assert_eq!(
            f(true, true, DType::F32),
            Budgeted(FlashRefusal::Dtype(DType::F32))
        );
        assert_eq!(
            pick(DType::BF16, &H3, &H3, true),
            Budgeted(FlashRefusal::Masked)
        );
    }

    #[test]
    fn shape_and_head_dim_guards_refuse_what_the_kernel_cannot_serve() {
        use AttentionKernel::Budgeted;
        let shape = Budgeted(FlashRefusal::Shape);
        // Rank 3 (the flat `[N, S, D]` shape) is not this dispatcher's.
        assert_eq!(pick(DType::BF16, &[56, 4096, 128], &H3, false), shape);
        // Batch mismatch.
        assert_eq!(pick(DType::BF16, &H3, &[2, 56, 4096, 128], false), shape);
        // KV heads that do not divide the query heads.
        assert_eq!(pick(DType::BF16, &H3, &[1, 5, 4096, 128], false), shape);
        // K and V disagreeing.
        assert_eq!(
            elig(
                true,
                SM120,
                DType::BF16,
                &H3,
                &H3,
                &[1, 56, 4095, 128],
                false
            ),
            shape
        );
        // Empty sequences.
        assert_eq!(pick(DType::BF16, &[1, 56, 0, 128], &H3, false), shape);
        // Head dims: not a multiple of 8, too wide, or mismatched q vs k.
        let hd = |d| Budgeted(FlashRefusal::HeadDim(d));
        assert_eq!(
            pick(DType::BF16, &[1, 56, 64, 100], &[1, 56, 64, 100], false),
            hd(100)
        );
        assert_eq!(
            pick(DType::BF16, &[1, 4, 64, 512], &[1, 4, 64, 512], false),
            hd(512)
        );
        assert_eq!(
            pick(DType::BF16, &[1, 56, 64, 128], &[1, 56, 64, 64], false),
            hd(128)
        );
    }

    #[test]
    fn q_k_v_must_share_one_dtype_and_one_device() {
        use AttentionKernel::Budgeted;
        let st = contiguous(&H3);
        let op = |dtype| FlashOperand {
            dtype,
            dims: &H3,
            strides: &st,
        };
        let (bf, f16) = (op(DType::BF16), op(DType::F16));
        assert_eq!(
            flash_eligibility(true, SM120, true, bf, bf, bf, false),
            AttentionKernel::Flash
        );
        for (k, v) in [(f16, bf), (bf, f16), (op(DType::F32), bf)] {
            assert_eq!(
                flash_eligibility(true, SM120, true, bf, k, v, false),
                Budgeted(FlashRefusal::MixedDtype)
            );
        }
        assert_eq!(
            flash_eligibility(true, SM120, false, bf, bf, bf, false),
            Budgeted(FlashRefusal::MixedDevice)
        );
    }

    #[test]
    fn strides_and_extents_beyond_the_kernels_u32_abi_refuse() {
        use AttentionKernel::Budgeted;
        let overflow = Budgeted(FlashRefusal::IndexOverflow);
        let max = u32::MAX as usize;
        let call = |q: &[usize], qs: &[usize], k: &[usize], ks: &[usize], vs: &[usize]| {
            let op = |dims, strides| FlashOperand {
                dtype: DType::BF16,
                dims,
                strides,
            };
            flash_eligibility(true, SM120, true, op(q, qs), op(k, ks), op(k, vs), false)
        };
        // The longest contiguous `[1, H, S, D]` whose `S·H·D` (the batch stride of q and of the
        // output) fits u32 takes the kernel; one more row does not.
        let (h, d) = (8usize, 128usize);
        let rows = max / (h * d);
        let fit = [1, h, rows, d];
        let st = contiguous(&fit);
        assert_eq!(call(&fit, &st, &fit, &st, &st), AttentionKernel::Flash);
        let over = [1, h, rows + 1, d];
        let so = contiguous(&over);
        assert_eq!(
            call(&over, &so, &fit, &st, &st),
            overflow,
            "q batch / output"
        );
        assert_eq!(
            call(&fit, &st, &over, &so, &so),
            overflow,
            "k/v batch stride"
        );
        // A small strided view: each oversized stride alone refuses — batch, head and sequence
        // (the `[B, S, H, D]` kernel view's row stride).
        let small = [2, 4, 16, 64];
        let ok = contiguous(&small);
        assert_eq!(call(&small, &ok, &small, &ok, &ok), AttentionKernel::Flash);
        for big in [
            [max + 1, 1024, 64, 1],
            [4096, max + 1, 64, 1],
            [4096, 1024, max + 1, 1],
        ] {
            assert_eq!(call(&small, &big, &small, &ok, &ok), overflow, "q {big:?}");
            assert_eq!(call(&small, &ok, &small, &big, &ok), overflow, "k {big:?}");
            assert_eq!(call(&small, &ok, &small, &ok, &big), overflow, "v {big:?}");
        }
        // A non-unit last stride is judged by the contiguous copy `flash_bhsd` makes, so huge
        // strides that the copy discards are fine...
        let gather = [1 << 40, 1 << 36, 1 << 33, 2];
        assert_eq!(
            call(&small, &gather, &small, &ok, &ok),
            AttentionKernel::Flash
        );
        // ...while the copy's own `S·H·D` still has to fit.
        assert_eq!(call(&over, &gather, &fit, &st, &st), overflow);
        // Strides whose rank disagrees with the dims are a shape error, not a launch.
        assert_eq!(
            call(&small, &ok[..3], &small, &ok, &ok),
            Budgeted(FlashRefusal::Shape)
        );
    }

    // -----------------------------------------------------------------------------------------
    // SW-6130 routing: pure decisions and counters. None of these runs attention (or any learned)
    // computation: they drive the pure router, record counts, or build zero-filled operands only to
    // read their layout / autograd state, and the entry-point probes fail on the operand rank before
    // a single matmul.
    // -----------------------------------------------------------------------------------------

    const OPEN: FlashGate = FlashGate {
        materialized: false,
        enabled: true,
        grad_tracked: false,
    };

    #[test]
    fn the_gate_refuses_in_rule_order() {
        let r = |materialized, enabled, grad_tracked, compiled| {
            flash_gate_refusal(
                FlashGate {
                    materialized,
                    enabled,
                    grad_tracked,
                },
                compiled,
            )
        };
        assert_eq!(r(false, true, false, true), None);
        // An explicit materialized request wins over everything, even a non-flash build.
        assert_eq!(
            r(true, false, true, false),
            Some(FlashRefusal::Materialized)
        );
        assert_eq!(
            r(false, false, true, false),
            Some(FlashRefusal::NotCompiled)
        );
        assert_eq!(r(false, false, true, true), Some(FlashRefusal::Disabled));
        assert_eq!(r(false, true, true, true), Some(FlashRefusal::GradTracked));
    }

    #[test]
    fn the_route_is_the_gate_then_the_eligibility_rule() {
        use AttentionKernel::Budgeted;
        let st = contiguous(&H3);
        let op = FlashOperand {
            dtype: DType::BF16,
            dims: &H3,
            strides: &st,
        };
        let route = |gate, masked| flash_route(gate, true, SM120, true, op, op, op, masked);
        assert_eq!(route(OPEN, false), AttentionKernel::Flash);
        // The toggle off, a materialized request and a taped operand each refuse a call that
        // would otherwise take the kernel.
        let off = FlashGate {
            enabled: false,
            ..OPEN
        };
        assert_eq!(route(off, false), Budgeted(FlashRefusal::Disabled));
        let mat = FlashGate {
            materialized: true,
            ..OPEN
        };
        assert_eq!(route(mat, false), Budgeted(FlashRefusal::Materialized));
        let taped = FlashGate {
            grad_tracked: true,
            ..OPEN
        };
        assert_eq!(route(taped, false), Budgeted(FlashRefusal::GradTracked));
        // An open gate defers to the eligibility rule: a mask still refuses.
        assert_eq!(route(OPEN, true), Budgeted(FlashRefusal::Masked));
        // A non-flash build refuses before any operand inspection.
        assert_eq!(
            flash_route(OPEN, false, SM120, true, op, op, op, false),
            Budgeted(FlashRefusal::NotCompiled)
        );
    }

    /// Every routed kernel lands in exactly one counter, the one its reason names.
    #[test]
    fn every_route_increments_exactly_its_own_counter() {
        use AttentionKernel::Budgeted;
        use FlashRefusal as R;
        type Field = fn(&AttentionPathCounts) -> u64;
        let cases: [(AttentionKernel, Field); 16] = [
            (AttentionKernel::Flash, |c| c.flash),
            (Budgeted(R::NotCompiled), |c| c.not_compiled),
            (Budgeted(R::NotCuda), |c| c.not_cuda),
            (Budgeted(R::UnsupportedArch(Some(75))), |c| {
                c.unsupported_arch
            }),
            (Budgeted(R::UnsupportedArch(None)), |c| c.unsupported_arch),
            (Budgeted(R::Dtype(DType::F32)), |c| c.dtype),
            (Budgeted(R::MixedDtype), |c| c.mixed_dtype_or_device),
            (Budgeted(R::MixedDevice), |c| c.mixed_dtype_or_device),
            (Budgeted(R::Masked), |c| c.masked),
            (Budgeted(R::Shape), |c| c.shape),
            (Budgeted(R::HeadDim(100)), |c| c.head_dim),
            (Budgeted(R::IndexOverflow), |c| c.index_overflow),
            (Budgeted(R::Disabled), |c| c.disabled),
            (Budgeted(R::GradTracked), |c| c.grad_tracked),
            (Budgeted(R::Materialized), |c| c.materialized),
            (Budgeted(R::Materialized), |c| c.materialized),
        ];
        for (kernel, field) in cases {
            let before = attention_path_counts();
            record_attention_path(kernel);
            let delta = attention_path_counts().since(&before);
            assert_eq!(field(&delta), 1, "{kernel:?} missed its counter");
            assert_eq!(
                delta.total(),
                1,
                "{kernel:?} landed in more than one counter"
            );
            let budgeted = u64::from(kernel != AttentionKernel::Flash);
            assert_eq!(delta.budgeted(), budgeted, "{kernel:?}");
        }
    }

    #[test]
    fn take_drains_the_counters() {
        record_attention_path(AttentionKernel::Flash);
        record_attention_path(AttentionKernel::Budgeted(FlashRefusal::Masked));
        let drained = take_attention_path_counts();
        assert!(drained.flash >= 1 && drained.masked >= 1, "{drained:?}");
        // Serial test threads (`.cargo/config.toml`): nothing ran in between.
        assert_eq!(attention_path_counts(), AttentionPathCounts::default());
    }

    /// Zero-filled `[1, 2, 16, 8]` operands: an allocation, never an attention computation.
    fn zero_operands(dtype: DType) -> candle_core::Result<(Tensor, Tensor, Tensor)> {
        let z = || Tensor::zeros((1, 2, 16, 8), dtype, &Device::Cpu);
        Ok((z()?, z()?, z()?))
    }

    /// What a CPU call reports once the gate is open: the build's first eligibility refusal.
    fn cpu_refusal() -> FlashRefusal {
        if FLASH_ATTN_COMPILED {
            FlashRefusal::NotCuda
        } else {
            FlashRefusal::NotCompiled
        }
    }

    #[test]
    fn the_toggle_round_trips_and_refuses_as_disabled() -> candle_core::Result<()> {
        let (q, k, v) = zero_operands(DType::BF16)?;
        assert!(flash_attention_enabled(), "the toggle defaults on");
        assert_eq!(
            route_attention_kernel(&q, &k, &v, None),
            AttentionKernel::Budgeted(cpu_refusal())
        );
        set_flash_attention_enabled(false);
        let off = route_attention_kernel(&q, &k, &v, None);
        set_flash_attention_enabled(true);
        // A non-flash build reports the missing kernel first; a flash build the switch.
        let want = if FLASH_ATTN_COMPILED {
            FlashRefusal::Disabled
        } else {
            FlashRefusal::NotCompiled
        };
        assert_eq!(off, AttentionKernel::Budgeted(want));
        assert!(flash_attention_enabled());
        Ok(())
    }

    #[test]
    fn a_training_scope_routes_materialized_and_nests() -> candle_core::Result<()> {
        let (q, k, v) = zero_operands(DType::BF16)?;
        let mat = AttentionKernel::Budgeted(FlashRefusal::Materialized);
        assert!(!in_materialized_attention_scope());
        let (outer, inner, after_inner) = with_materialized_attention(|| {
            let outer = route_attention_kernel(&q, &k, &v, None);
            let inner = with_materialized_attention(|| route_attention_kernel(&q, &k, &v, None));
            (outer, inner, in_materialized_attention_scope())
        });
        assert_eq!((outer, inner), (mat, mat));
        assert!(
            after_inner,
            "leaving the inner scope must keep the outer one"
        );
        assert!(!in_materialized_attention_scope());
        assert_eq!(
            route_attention_kernel(&q, &k, &v, None),
            AttentionKernel::Budgeted(cpu_refusal())
        );
        Ok(())
    }

    #[test]
    fn an_operand_on_the_tape_refuses_the_fused_kernel() -> candle_core::Result<()> {
        let (q, k, v) = zero_operands(DType::BF16)?;
        let taped = candle_core::Var::zeros((1, 2, 16, 8), DType::BF16, &Device::Cpu)?;
        let want = if FLASH_ATTN_COMPILED {
            FlashRefusal::GradTracked
        } else {
            FlashRefusal::NotCompiled
        };
        for (q, k, v) in [
            (taped.as_tensor(), &k, &v),
            (&q, taped.as_tensor(), &v),
            (&q, &k, taped.as_tensor()),
        ] {
            assert_eq!(
                route_attention_kernel(q, k, v, None),
                AttentionKernel::Budgeted(want)
            );
        }
        // A detached view of the same storage is off the tape again.
        assert_eq!(
            route_attention_kernel(&taped.as_tensor().detach(), &k, &v, None),
            AttentionKernel::Budgeted(cpu_refusal())
        );
        Ok(())
    }

    /// The `*_materialized` entry points count `Materialized` without consulting the router, and
    /// the flash-eligible ones count the router's reason. Rank-2 operands fail every helper at its
    /// first `dims4` / `dims3`, so no attention is computed.
    #[test]
    fn each_entry_point_counts_its_route() {
        use crate::attention::{
            sdpa_budgeted_bhsd_materialized, sdpa_budgeted_flat, sdpa_budgeted_flat_materialized,
            sdpa_planned_bhsd_materialized, sdpa_planned_flat, sdpa_planned_flat_materialized,
        };
        let bad = Tensor::zeros((4, 8), DType::F32, &Device::Cpu).unwrap();
        let sm = |s: &Tensor| softmax_last_dim(s);
        let plan = AttentionPlan::UNBOUNDED;
        let count = |f: &dyn Fn() -> bool| {
            let before = attention_path_counts();
            assert!(f(), "the probe call must fail on its operand rank");
            attention_path_counts().since(&before)
        };
        let materialized = [
            count(&|| sdpa_budgeted_bhsd_materialized(&bad, &bad, &bad, 1.0, None, sm, 8).is_err()),
            count(&|| {
                sdpa_planned_bhsd_materialized(&bad, &bad, &bad, 1.0, None, sm, plan).is_err()
            }),
            count(&|| sdpa_budgeted_flat_materialized(&bad, &bad, &bad, 1.0, sm, 8).is_err()),
            count(&|| sdpa_planned_flat_materialized(&bad, &bad, &bad, 1.0, sm, plan).is_err()),
        ];
        for delta in materialized {
            assert_eq!(delta.materialized, 1, "{delta:?}");
            assert_eq!(delta.total(), 1, "{delta:?}");
        }
        // The flash-eligible bhsd helpers route first: on the CPU the refusal is the build's.
        let routed = [
            count(&|| sdpa_budgeted_bhsd(&bad, &bad, &bad, 1.0, None, sm, 8).is_err()),
            count(&|| sdpa_planned_bhsd(&bad, &bad, &bad, 1.0, None, sm, plan).is_err()),
        ];
        for delta in routed {
            assert_eq!(delta.total(), 1, "{delta:?}");
            assert_eq!(delta.flash, 0, "{delta:?}");
            assert_eq!(delta.materialized, 0, "{delta:?}");
        }
        // The flat helpers refuse a non-rank-3 operand as a shape mismatch before any view.
        for delta in [
            count(&|| sdpa_budgeted_flat(&bad, &bad, &bad, 1.0, sm, 8).is_err()),
            count(&|| sdpa_planned_flat(&bad, &bad, &bad, 1.0, sm, plan).is_err()),
        ] {
            assert_eq!(delta.shape, 1, "{delta:?}");
            assert_eq!(delta.total(), 1, "{delta:?}");
        }
    }

    /// The flat `[N, S, D]` helpers route the free `[N, 1, S, D]` view: the pure layout map agrees
    /// with candle's own `unsqueeze` (so the router judges exactly what the kernel receives), and
    /// the kernel's view of it needs no copy.
    #[test]
    fn the_flat_layout_maps_to_n_single_head_rows_without_a_copy() -> candle_core::Result<()> {
        // SDXL's heads-into-batch self-attention at 1024² (N = B·H = 2·10, S = 4096, d = 64), and a
        // strided (sliced) operand.
        let base = Tensor::zeros((20, 4096, 64), DType::BF16, &Device::Cpu)?;
        let sliced = Tensor::zeros((20, 4096, 128), DType::BF16, &Device::Cpu)?.narrow(2, 0, 64)?;
        for t in [&base, &sliced] {
            let view = flat_as_bhsd(t)?;
            let dims: [usize; 3] = t.dims().try_into().unwrap();
            let strides: [usize; 3] = t.stride().try_into().unwrap();
            let (want_dims, want_strides) = flat_as_bhsd_layout(dims, strides);
            assert_eq!(view.dims(), want_dims);
            assert_eq!(view.stride(), want_strides);
            // The kernel's `[N, S, 1, D]` transpose keeps a unit last stride: passed through as-is.
            assert_eq!(view.transpose(1, 2)?.stride().last(), Some(&1));
        }
        let (dims, strides) = flat_as_bhsd_layout([20, 4096, 64], [4096 * 64, 64, 1]);
        let op = FlashOperand {
            dtype: DType::BF16,
            dims: &dims,
            strides: &strides,
        };
        assert_eq!(
            flash_eligibility(true, SM120, true, op, op, op, false),
            AttentionKernel::Flash
        );
        // A single-head 512-channel VAE mid block keeps the materialized path (head dim > 256).
        let (dims, strides) = flat_as_bhsd_layout([1, 16384, 512], [16384 * 512, 512, 1]);
        let vae = FlashOperand {
            dtype: DType::BF16,
            dims: &dims,
            strides: &strides,
        };
        assert_eq!(
            flash_eligibility(true, SM120, true, vae, vae, vae, false),
            AttentionKernel::Budgeted(FlashRefusal::HeadDim(512))
        );
        Ok(())
    }

    /// On the CPU every call refuses (`NotCuda` or `NotCompiled`), and the flash-eligible helper is
    /// then the materialized path verbatim — the same values, and the same typed cancel.
    #[test]
    fn the_cpu_path_is_the_planned_budgeted_path() -> crate::Result<()> {
        use crate::attention::sdpa_planned_bhsd_materialized;
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        let k = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        let v = Tensor::randn(0f32, 1.0, (1, 2, 16, 8), &dev)?;
        assert!(matches!(
            select_attention_kernel(&q, &k, &v, None),
            AttentionKernel::Budgeted(FlashRefusal::NotCompiled | FlashRefusal::NotCuda)
        ));
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let got = sdpa_planned_bhsd(&q, &k, &v, 0.35, None, softmax_last_dim, plan)?;
        let want = sdpa_planned_bhsd_materialized(&q, &k, &v, 0.35, None, softmax_last_dim, plan)?;
        let d = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(d, 0.0);

        // A tripped flag on a *bounded* plan that actually chunks surfaces as `Canceled`.
        let cancel = crate::gen_core::CancelFlag::default();
        cancel.cancel();
        let tiny = AttentionPlan::budgeted(AttentionBudget::from_score_elements(64, false))
            .with_cancel(&cancel);
        let err = sdpa_planned_bhsd(&q, &k, &v, 0.35, None, softmax_last_dim, tiny).unwrap_err();
        assert!(matches!(err, crate::CandleError::Canceled), "{err:?}");
        Ok(())
    }

    // -----------------------------------------------------------------------------------------
    // Inference compute — `#[ignore]`d, never part of normal CI. Run on a pinned validation GPU:
    //
    // CUDA_VISIBLE_DEVICES=<gpu> cargo test -p candle-gen --features flash-attn --lib \
    //   fused_attention::tests::cuda_ -- --ignored --nocapture --test-threads=1
    // -----------------------------------------------------------------------------------------

    /// `(max |a - b|, ‖a - b‖₂ / ‖b‖₂)` in f32.
    #[cfg(feature = "flash-attn")]
    fn errors(a: &Tensor, b: &Tensor) -> crate::Result<(f32, f32)> {
        let a = a.to_dtype(DType::F32)?;
        let b = b.to_dtype(DType::F32)?;
        let d = (&a - &b)?;
        let max = d.abs()?.max_all()?.to_scalar::<f32>()?;
        let num = d.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let den = b.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        Ok((max, num / den.max(f32::MIN_POSITIVE)))
    }

    /// Flash vs the materialized path, through the **shared helpers themselves**, at head dims 64,
    /// 128 and 256, with and without GQA, bf16, on the 4-D and the flat helpers.
    ///
    /// The reference is the materialized path entirely in f32 (GQA expanded with `repeat_kv`).
    /// Tolerance, declared: **max |Δ| ≤ 2e-2 and rel-L2 ≤ 1e-2** against that reference. The output
    /// is a convex combination of unit-variance bf16 values (|v| ≲ 5), so rounding the output alone
    /// to bf16 (8 significant bits, a half-ULP of 2^-9 relative) costs up to ~1e-2 absolute; the
    /// kernel adds its bf16 probabilities on top. The bf16 materialized path is printed beside it
    /// as the yardstick: flash must sit at the same order, not merely under the ceiling.
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_flash_matches_budgeted() -> crate::Result<()> {
        use crate::attention::{repeat_kv, sdpa_budgeted_flat, sdpa_planned_bhsd_materialized};
        let dev = Device::new_cuda(0)?;
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let f32_softmax = |s: &Tensor| -> candle_core::Result<Tensor> {
            softmax_last_dim(&s.to_dtype(DType::F32)?)?.to_dtype(s.dtype())
        };
        let (mut worst_max, mut worst_rel) = (0f32, 0f32);
        // (b, q heads, kv heads, sq, sk, d)
        for (b, h, hk, sq, sk, d) in [
            (1, 8, 8, 1024, 1024, 64),
            (2, 8, 2, 333, 517, 64),
            (1, 24, 24, 1024, 1024, 128),
            (1, 32, 8, 777, 777, 128),
            (1, 8, 8, 512, 512, 256),
            (1, 8, 4, 300, 600, 256),
        ] {
            let q = Tensor::randn(0f32, 1.0, (b, h, sq, d), &dev)?.to_dtype(DType::BF16)?;
            let k = Tensor::randn(0f32, 1.0, (b, hk, sk, d), &dev)?.to_dtype(DType::BF16)?;
            let v = Tensor::randn(0f32, 1.0, (b, hk, sk, d), &dev)?.to_dtype(DType::BF16)?;
            assert_eq!(
                route_attention_kernel(&q, &k, &v, None),
                AttentionKernel::Flash
            );
            let scale = 1.0 / (d as f64).sqrt();
            let before = attention_path_counts();
            let fused = sdpa_planned_bhsd(&q, &k, &v, scale, None, f32_softmax, plan)?;
            assert_eq!(attention_path_counts().since(&before).flash, 1);
            let (ke, ve) = (repeat_kv(&k, h / hk)?, repeat_kv(&v, h / hk)?);
            let want = sdpa_planned_bhsd_materialized(
                &q.to_dtype(DType::F32)?,
                &ke.to_dtype(DType::F32)?,
                &ve.to_dtype(DType::F32)?,
                scale,
                None,
                softmax_last_dim,
                plan,
            )?;
            let old = sdpa_planned_bhsd_materialized(&q, &ke, &ve, scale, None, f32_softmax, plan)?;
            let (fm, fr) = errors(&fused, &want)?;
            let (bm, br) = errors(&old, &want)?;
            println!(
                "flash parity bhsd b={b} h={h} hk={hk} sq={sq} sk={sk} d={d}: flash max|Δ| {fm:.3e} \
                 rel-L2 {fr:.3e}; bf16 budgeted max|Δ| {bm:.3e} rel-L2 {br:.3e}"
            );
            worst_max = worst_max.max(fm);
            worst_rel = worst_rel.max(fr);

            // The flat helper on the same rows (heads folded into the batch, no GQA there).
            if hk == h {
                let flat = |t: &Tensor| t.reshape((b * h, t.dim(2)?, d));
                let before = attention_path_counts();
                let fused_flat = sdpa_budgeted_flat(
                    &flat(&q)?,
                    &flat(&k)?,
                    &flat(&v)?,
                    scale,
                    f32_softmax,
                    ATTN_SCORES_BUDGET,
                )?;
                assert_eq!(attention_path_counts().since(&before).flash, 1);
                let (fm, fr) = errors(&fused_flat, &flat(&want)?)?;
                println!(
                    "flash parity flat n={} s={sq} d={d}: max|Δ| {fm:.3e} rel-L2 {fr:.3e}",
                    b * h
                );
                worst_max = worst_max.max(fm);
                worst_rel = worst_rel.max(fr);
            }
        }
        println!("flash parity worst: max|Δ| {worst_max:.3e}, rel-L2 {worst_rel:.3e}");
        assert!(
            worst_max <= 2e-2,
            "flash vs f32 reference max|Δ| = {worst_max}"
        );
        assert!(
            worst_rel <= 1e-2,
            "flash vs f32 reference rel-L2 = {worst_rel}"
        );
        Ok(())
    }

    /// A masked call through the flash-eligible helper must be the materialized path exactly —
    /// the router refuses the kernel, which cannot express an additive bias. (The SW-6130 mutation
    /// check forced the kernel on this call and watched the comparison fail.)
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_masked_calls_keep_the_materialized_path() -> crate::Result<()> {
        use crate::attention::sdpa_planned_bhsd_materialized;
        let dev = Device::new_cuda(0)?;
        let (b, h, s, d) = (1, 8, 512, 128);
        let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        // A key-padding mask hiding the last quarter of the keys.
        let keep: Vec<f32> = (0..s)
            .map(|i| if i < 3 * s / 4 { 0.0 } else { -1e9 })
            .collect();
        let mask = Tensor::from_vec(keep, (1, 1, 1, s), &dev)?.to_dtype(DType::BF16)?;
        let plan = AttentionPlan::UNBOUNDED;
        let scale = 1.0 / (d as f64).sqrt();
        let before = attention_path_counts();
        let got = sdpa_planned_bhsd(&q, &k, &v, scale, Some(&mask), softmax_last_dim, plan)?;
        let delta = attention_path_counts().since(&before);
        let want =
            sdpa_planned_bhsd_materialized(&q, &k, &v, scale, Some(&mask), softmax_last_dim, plan)?;
        let (max, rel) = errors(&got, &want)?;
        println!("masked parity: max|Δ| {max:.3e} rel-L2 {rel:.3e}, counts {delta:?}");
        assert_eq!(delta.masked, 1, "{delta:?}");
        assert_eq!(max, 0.0, "a masked call left the materialized path");
        Ok(())
    }

    /// The fused path never allocates the `[B, H, Sq, Sk]` scores. At S = 16384 (8 heads, d = 128,
    /// bf16) the materialized scores alone would be 4 GiB; the measured live-byte peak of the call
    /// (the stream-ordered pool's `USED_MEM_HIGH`, [`crate::cuda_mempool::MemPool`]) must stay
    /// under 1/16 of that — in practice it is the output and its transposed copy.
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute: run by the validation workflow, not CI"]
    fn cuda_flash_peak_memory_has_no_score_matrix() -> crate::Result<()> {
        use crate::cuda_mempool::MemPool;
        let dev = Device::new_cuda(0)?;
        let (b, h, s, d) = (1usize, 8usize, 16384usize, 128usize);
        let q = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (b, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let scale = 1.0 / (d as f64).sqrt();
        dev.synchronize()?;
        let pool = MemPool::device_current(0).expect("the CUDA device's current pool");
        let baseline = pool.used().expect("pool USED");
        assert!(pool.reset_used_high_water(), "reset USED_MEM_HIGH");
        let before = attention_path_counts();
        let out = sdpa_planned_bhsd(
            &q,
            &k,
            &v,
            scale,
            None,
            softmax_last_dim,
            AttentionPlan::UNBOUNDED,
        )?;
        dev.synchronize()?;
        assert_eq!(attention_path_counts().since(&before).flash, 1);
        let peak = pool
            .used_high()
            .expect("pool USED_MEM_HIGH")
            .saturating_sub(baseline);
        let scores = (b * h * s * s * DType::BF16.size_in_bytes()) as u64;
        let operand = (b * h * s * d * DType::BF16.size_in_bytes()) as u64;
        println!(
            "flash peak at S={s}: {peak} bytes above baseline ({:.1} MiB); one operand {:.1} MiB; \
             the materialized [B,H,S,S] bf16 scores would be {scores} bytes ({:.1} MiB)",
            peak as f64 / 1048576.0,
            operand as f64 / 1048576.0,
            scores as f64 / 1048576.0
        );
        assert!(peak < scores / 16, "peak {peak} vs scores {scores}");
        assert!(peak <= 4 * operand, "peak {peak} vs operand {operand}");
        drop(out);
        Ok(())
    }

    /// **Inference compute — `#[ignore]`d.** Wall time of one MiniMax-H3-shaped attention
    /// (B=1, 56 heads, d=128, bf16) through the fused kernel and the 1e9-budgeted path.
    /// `H3_ATTN_BENCH_SEQ` overrides the sequence length (default 40,000).
    #[cfg(feature = "flash-attn")]
    #[test]
    #[ignore = "CUDA compute benchmark: run by hand, not CI"]
    fn h3_shaped_attention_flash_vs_budgeted_timing() -> crate::Result<()> {
        use crate::attention::sdpa_planned_bhsd_materialized;
        let dev = Device::new_cuda(0)?;
        let s: usize = std::env::var("H3_ATTN_BENCH_SEQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(40_000);
        let (h, d) = (56, 128);
        let q = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1.0, (1, h, s, d), &dev)?.to_dtype(DType::BF16)?;
        let scale = 1.0 / (d as f64).sqrt();
        let plan = AttentionPlan::budgeted(crate::attention::attention_budget_from_usize(
            ATTN_SCORES_BUDGET,
        ));
        let time = |f: &dyn Fn() -> crate::Result<Tensor>| -> crate::Result<(f64, Tensor)> {
            let _warm = f()?;
            dev.synchronize()?;
            let t = std::time::Instant::now();
            let out = f()?;
            dev.synchronize()?;
            Ok((t.elapsed().as_secs_f64(), out))
        };
        let (tf, of) =
            time(&|| sdpa_planned_bhsd(&q, &k, &v, scale, None, softmax_last_dim, plan))?;
        let (tb, ob) = time(&|| {
            sdpa_planned_bhsd_materialized(&q, &k, &v, scale, None, softmax_last_dim, plan)
        })?;
        let diff = (of.to_dtype(DType::F32)? - ob.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        println!(
            "h3-shaped attention S={s} H={h} d={d} bf16: flash {tf:.3}s, budgeted {tb:.3}s \
             ({:.1}x), max|Δ| = {diff:.3e}",
            tb / tf
        );
        Ok(())
    }
}

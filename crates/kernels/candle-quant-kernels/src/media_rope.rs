//! The fused **strided RoPE** and **QK-RMSNorm + RoPE** kernels of the media lane (SW-6128,
//! feature SW-F287 S3), compiled once per device from [`MEDIA_ROPE_SRC`] through the nvrtc seam.
//!
//! The public entry points are candle-gen's `rope::apply_rope*` / `rope::rms_norm_rope*`: they
//! own the reference op chains (the parity oracle and every non-CUDA path, epic E1) and dispatch
//! here through [`ROPE_PATH`] / [`RMS_NORM_ROPE_PATH`] (the media fused-path seam,
//! [`crate::fused_path`]). This module holds what the kernel needs: the spec vocabulary
//! ([`RopeSpec`]), the pure plan ([`check_rope`], CPU-tested) with its typed refusals
//! ([`RopeRefusal`]), and the cuda-only launches.
//!
//! # What the kernels do that the op chain cannot
//!
//! - **Strided operands.** `x` is addressed through its own strides and start offset, so a
//!   permuted `[B, S, H, D] -> [B, H, S, D]` view, or q / k narrowed out of a fused QKV
//!   projection, is rotated without a `.contiguous()` copy. Only the lane (last) axis must have
//!   stride 1.
//! - **In place.** The `in_place` form writes the result into `x`'s own buffer (no q/k-sized
//!   allocation) when the view does not overlap itself and carries no gradient.
//! - **One launch for QK-norm + RoPE** (Krea's `norm_q` → RoPE), with no f32 intermediates.
//!
//! The arithmetic is the reference chain's, operation for operation (see the contract at the top
//! of `media_rope.cu`), so the result is bit-identical to candle-gen's reference on CUDA in every
//! served dtype and math mode — the GPU parity test holds it to that.
//!
//! # Why a media kernel file rather than `fused_decode::rms_norm_rope`
//!
//! The LLM decode primitive (`fused_decode::rms_norm_rope`) is a different contract on a
//! different seam: contiguous `[B, S, H, D]` only (it `.contiguous()`s every operand and always
//! allocates), tables in `x`'s dtype at full rotary width, native-dtype math only, f32 / bf16 only,
//! and bit-identity against candle-llm's own chain (separate `sum` / `mean` / `sqrt` ops, not
//! candle's fused `rmsnorm` kernel that the media lane's `candle_nn::ops::rms_norm` calls).
//! Extending it would put the media addressing and f32-math modes on the LLM lane's numerics and
//! telemetry (`candle_layers::fused`); a second source on the media seam keeps both contracts
//! exact.

use std::fmt;

use candle_core::{DType, Layout};

use crate::fused_path::{FusedKernelPath, PathReason};
use crate::nvrtc::KernelSource;

/// The fused media RoPE source; compiled once per device on first use.
pub const MEDIA_ROPE_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_media_rope_v1",
    src: include_str!("media_rope.cu"),
    // f32 arithmetic, `cvt.f32.f16` / `cvt.rn.f16.f32`, software bf16 rounding, `shfl.sync`
    // (sm_70+): sm_70 is the oldest this workspace targets.
    cc_floor: (7, 0),
};

/// The fused RoPE's path record (candle-gen `rope::apply_rope` / `apply_rope_in_place`).
pub static ROPE_PATH: FusedKernelPath = FusedKernelPath::new("rope", "CANDLE_FUSED_ROPE");

/// The fused QK-RMSNorm + RoPE's path record (candle-gen `rope::rms_norm_rope*`).
pub static RMS_NORM_ROPE_PATH: FusedKernelPath =
    FusedKernelPath::new("rms_norm_rope", "CANDLE_FUSED_RMS_NORM_ROPE");

/// Largest tensor either kernel launches (epic E6); larger ones take the reference.
pub const ROPE_MAX_ELEMS: usize = i32::MAX as usize;
/// The RMSNorm + RoPE kernel serves head dims below this (candle's `rmsnorm` switches to a
/// 1024-thread block, a different summation order, at 1024).
pub const RMS_NORM_ROPE_MAX_HEAD_DIM: usize = 1023;
/// Threads per block of the RoPE kernel.
pub const ROPE_THREADS: u32 = 256;
/// Most blocks the RoPE kernel launches (grid-stride beyond).
pub const ROPE_MAX_BLOCKS: u32 = 1 << 20;
/// Warps (rows) per block of the RMSNorm + RoPE kernel (`RMS_ROPE_WARPS` in the source).
pub const RMS_NORM_ROPE_WARPS: u32 = 4;

/// Which lanes form a rotated pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopePairing {
    /// GPT-J / complex pairs `(2i, 2i+1)` — Krea, FLUX, Lumina, candle `rope_i`.
    Interleaved,
    /// NeoX / HF rotate-half pairs `(i, i + rotary_dim/2)` — candle `rope`.
    SplitHalf,
}

/// Which axis of a rank-4 `x` is the sequence (the other non-lane axis is heads).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeLayout {
    /// `[batch, seq, heads, head_dim]`.
    Bshd,
    /// `[batch, heads, seq, head_dim]`.
    Bhsd,
}

impl RopeLayout {
    /// `x`'s sequence axis.
    pub fn seq_axis(self) -> usize {
        match self {
            Self::Bshd => 1,
            Self::Bhsd => 2,
        }
    }

    /// `x`'s head axis.
    pub fn head_axis(self) -> usize {
        match self {
            Self::Bshd => 2,
            Self::Bhsd => 1,
        }
    }
}

/// The precision the rotation is computed in — the two reference chains in the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RopeMath {
    /// Upcast `x` and the tables to f32, rotate in f32, round once to `x`'s dtype (Krea, the
    /// shared `apply_interleaved_rope`, FLUX.2 vision).
    F32,
    /// Cast the tables to `x`'s dtype and multiply / add in that dtype (the bf16-native HF copies:
    /// Anima, Ideogram). Identical to [`Self::F32`] for an f32 `x`.
    Native,
}

/// What to rotate and how: one value describes every RoPE call site's convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RopeSpec {
    /// Pair layout.
    pub pairing: RopePairing,
    /// Where the sequence axis is.
    pub layout: RopeLayout,
    /// Leading lanes rotated (`None` = the whole head; partial rotary passes the rest through).
    pub rotary_dim: Option<usize>,
    /// Compute precision.
    pub math: RopeMath,
}

impl RopeSpec {
    /// Full-head rotary in f32 math.
    pub const fn new(pairing: RopePairing, layout: RopeLayout) -> Self {
        Self {
            pairing,
            layout,
            rotary_dim: None,
            math: RopeMath::F32,
        }
    }

    /// Interleaved pairs, full head, f32 math.
    pub const fn interleaved(layout: RopeLayout) -> Self {
        Self::new(RopePairing::Interleaved, layout)
    }

    /// Rotate-half pairs, full head, f32 math.
    pub const fn split_half(layout: RopeLayout) -> Self {
        Self::new(RopePairing::SplitHalf, layout)
    }

    /// Rotate only the first `rotary_dim` lanes.
    pub const fn with_rotary_dim(mut self, rotary_dim: usize) -> Self {
        self.rotary_dim = Some(rotary_dim);
        self
    }

    /// Compute in `math`.
    pub const fn with_math(mut self, math: RopeMath) -> Self {
        self.math = math;
        self
    }

    /// The rotated lane count for a head of `head_dim` lanes.
    pub fn rotary(&self, head_dim: usize) -> usize {
        self.rotary_dim.unwrap_or(head_dim)
    }
}

/// Why the fused kernels do not serve a call. The reference serves it instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeRefusal {
    /// `x` (or a table / weight) is not on `x`'s CUDA device.
    NotCuda,
    /// Only `F32`, `BF16` and `F16` are served.
    Dtype(DType),
    /// A gradient could flow through the call (epic E9: training keeps the op chain).
    Autograd,
    /// `x` is not rank 4.
    Rank(usize),
    /// `x` has no elements.
    Empty,
    /// The rotary dim is odd, zero or wider than the head.
    RotaryDim {
        /// Requested rotated lanes.
        rotary: usize,
        /// Head lanes.
        head_dim: usize,
    },
    /// The tables are not `[1 | batch, seq, rotary/2 | rotary]`, or cos and sin differ in shape.
    TableShape,
    /// The norm weight is not `[head_dim]`.
    WeightShape,
    /// The RMSNorm + RoPE kernel serves head dims up to [`RMS_NORM_ROPE_MAX_HEAD_DIM`].
    HeadDimTooLarge(usize),
    /// `x`'s lane axis is not unit-stride.
    LaneStride,
    /// The in-place form was asked of a view whose elements overlap (a broadcast / stride-0 view).
    Overlapping,
    /// More than [`ROPE_MAX_ELEMS`] elements.
    TooLarge(usize),
}

impl RopeRefusal {
    /// The fused-path reason this refusal is counted under.
    pub fn reason(&self) -> PathReason {
        match self {
            Self::NotCuda => PathReason::NotCuda,
            Self::Dtype(d) => PathReason::Dtype(*d),
            Self::Autograd => PathReason::Autograd,
            Self::Rank(_)
            | Self::Empty
            | Self::RotaryDim { .. }
            | Self::TableShape
            | Self::WeightShape
            | Self::HeadDimTooLarge(_) => PathReason::Shape,
            Self::LaneStride | Self::Overlapping => PathReason::Layout,
            Self::TooLarge(_) => PathReason::IndexOverflow,
        }
    }
}

impl fmt::Display for RopeRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "operands are not on one CUDA device"),
            Self::Dtype(d) => write!(f, "dtype {d:?} is not served (F32, BF16 or F16)"),
            Self::Autograd => write!(f, "an operand tracks gradients"),
            Self::Rank(r) => write!(f, "x has rank {r}, not 4"),
            Self::Empty => write!(f, "x is empty"),
            Self::RotaryDim { rotary, head_dim } => {
                write!(f, "rotary dim {rotary} is not even and in 2..={head_dim}")
            }
            Self::TableShape => write!(f, "cos/sin are not [1|batch, seq, rotary/2|rotary]"),
            Self::WeightShape => write!(f, "the norm weight is not [head_dim]"),
            Self::HeadDimTooLarge(d) => {
                write!(f, "head dim {d} exceeds {RMS_NORM_ROPE_MAX_HEAD_DIM}")
            }
            Self::LaneStride => write!(f, "x's last axis is not unit-stride"),
            Self::Overlapping => write!(f, "x's view overlaps itself (cannot write in place)"),
            Self::TooLarge(n) => write!(f, "{n} elements exceed {ROPE_MAX_ELEMS}"),
        }
    }
}

/// One operand's view: logical dims, element strides and start offset (a candle [`Layout`]).
#[derive(Clone, Copy, Debug)]
pub struct RopeOperand<'a> {
    /// Logical dims.
    pub dims: &'a [usize],
    /// Element strides, one per dim.
    pub stride: &'a [usize],
    /// Start offset in elements.
    pub offset: usize,
}

impl<'a> From<&'a Layout> for RopeOperand<'a> {
    fn from(l: &'a Layout) -> Self {
        Self {
            dims: l.dims(),
            stride: l.stride(),
            offset: l.start_offset(),
        }
    }
}

/// Everything [`check_rope`] decides on (pure data, so every lane tests it).
#[derive(Clone, Copy, Debug)]
pub struct RopeRequest<'a> {
    /// `x` and every table / weight live on one CUDA device.
    pub on_cuda: bool,
    /// `x`'s dtype.
    pub dtype: DType,
    /// Any operand tracks gradients.
    pub tracks_grad: bool,
    /// `x`, rank 4 in [`RopeSpec::layout`].
    pub x: RopeOperand<'a>,
    /// `cos`, as `[table_batch, seq, width]`.
    pub cos: RopeOperand<'a>,
    /// `sin`'s dims (must equal `cos`'s).
    pub sin_dims: &'a [usize],
    /// The convention.
    pub spec: RopeSpec,
    /// Write into `x`'s buffer.
    pub in_place: bool,
    /// `Some(len)` for the RMSNorm + RoPE kernel: the norm weight's length.
    pub rms_weight_len: Option<usize>,
}

/// A served call's launch geometry. Row axes are `x`'s three non-lane axes ordered by descending
/// `x` stride (the kernel walks the last fastest), each with its `x`, output and table stride.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RopePlan {
    /// `x`'s dtype.
    pub dtype: DType,
    /// Rows (`batch · seq · heads`).
    pub rows: usize,
    /// Row-axis extents, slowest first.
    pub n: [usize; 3],
    /// `x` strides of the row axes.
    pub x_stride: [usize; 3],
    /// Output strides of the row axes (`x`'s for in place; contiguous otherwise).
    pub y_stride: [usize; 3],
    /// Table strides of the row axes (0 broadcasts).
    pub t_stride: [usize; 3],
    /// Table lane stride.
    pub t_lane: usize,
    /// `x`'s start offset (also the output's in place; 0 otherwise).
    pub x_offset: usize,
    /// Head lanes.
    pub head_dim: usize,
    /// Rotated lanes.
    pub rotary_dim: usize,
    /// Work items per row of the RoPE kernel.
    pub items: usize,
    /// One table entry per lane (`true`) or per pair.
    pub full_table: bool,
    /// Rotate-half pairs (else interleaved).
    pub split: bool,
    /// Round every product and sum to the dtype ([`RopeMath::Native`] on a 16-bit `x`).
    pub native: bool,
    /// Output is `x`'s buffer.
    pub in_place: bool,
    /// Blocks of the launch.
    pub blocks: u32,
}

impl RopePlan {
    /// The output's start offset.
    pub fn y_offset(&self) -> usize {
        if self.in_place {
            self.x_offset
        } else {
            0
        }
    }
}

/// Whether a strided view addresses every element at a distinct offset (sorted by stride, each
/// stride exceeds the span of the faster axes).
pub fn view_is_non_overlapping(dims: &[usize], stride: &[usize]) -> bool {
    let mut axes: Vec<(usize, usize)> = dims
        .iter()
        .zip(stride)
        .filter(|(n, _)| **n > 1)
        .map(|(n, s)| (*s, *n))
        .collect();
    axes.sort_unstable();
    let mut span = 0usize; // max offset reachable by the faster axes
    for (s, n) in axes {
        if s <= span {
            return false;
        }
        span = match (n - 1).checked_mul(s).and_then(|e| e.checked_add(span)) {
            Some(v) => v,
            None => return false,
        };
    }
    true
}

/// The fused kernels' input check and launch geometry (pure; every lane runs it).
pub fn check_rope(req: &RopeRequest<'_>) -> Result<RopePlan, RopeRefusal> {
    if !req.on_cuda {
        return Err(RopeRefusal::NotCuda);
    }
    if !matches!(req.dtype, DType::F32 | DType::BF16 | DType::F16) {
        return Err(RopeRefusal::Dtype(req.dtype));
    }
    if req.tracks_grad {
        return Err(RopeRefusal::Autograd);
    }
    let dims = req.x.dims;
    if dims.len() != 4 || req.x.stride.len() != 4 {
        return Err(RopeRefusal::Rank(dims.len()));
    }
    let elems: usize = dims.iter().product();
    if elems == 0 {
        return Err(RopeRefusal::Empty);
    }
    let (seq_axis, head_axis) = (req.spec.layout.seq_axis(), req.spec.layout.head_axis());
    let (batch, seq, head_dim) = (dims[0], dims[seq_axis], dims[3]);
    let rotary = req.spec.rotary(head_dim);
    if rotary == 0 || !rotary.is_multiple_of(2) || rotary > head_dim {
        return Err(RopeRefusal::RotaryDim { rotary, head_dim });
    }
    let t = req.cos.dims;
    if t.len() != 3
        || req.cos.stride.len() != 3
        || req.sin_dims != t
        || !(t[0] == 1 || t[0] == batch)
        || t[1] != seq
        || !(t[2] == rotary / 2 || t[2] == rotary)
    {
        return Err(RopeRefusal::TableShape);
    }
    if let Some(len) = req.rms_weight_len {
        if len != head_dim {
            return Err(RopeRefusal::WeightShape);
        }
        if head_dim > RMS_NORM_ROPE_MAX_HEAD_DIM {
            return Err(RopeRefusal::HeadDimTooLarge(head_dim));
        }
    }
    if head_dim > 1 && req.x.stride[3] != 1 {
        return Err(RopeRefusal::LaneStride);
    }
    if elems > ROPE_MAX_ELEMS {
        return Err(RopeRefusal::TooLarge(elems));
    }
    if req.in_place && !view_is_non_overlapping(dims, req.x.stride) {
        return Err(RopeRefusal::Overlapping);
    }

    // Per x axis 0..3: table stride (batch broadcasts when the table has one row; heads always).
    let table_stride = |axis: usize| -> usize {
        if axis == 0 {
            if t[0] == 1 {
                0
            } else {
                req.cos.stride[0]
            }
        } else if axis == seq_axis {
            req.cos.stride[1]
        } else {
            debug_assert_eq!(axis, head_axis);
            0
        }
    };
    let contiguous = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3]];
    let mut order = [0usize, 1, 2];
    // Slowest first: descending x stride (ties keep logical order).
    order.sort_by(|a, b| req.x.stride[*b].cmp(&req.x.stride[*a]));
    let pick = |f: &dyn Fn(usize) -> usize| -> [usize; 3] { order.map(f) };
    let x_stride = pick(&|a| req.x.stride[a]);
    let y_stride = if req.in_place {
        x_stride
    } else {
        pick(&|a| contiguous[a])
    };
    let rows = elems / head_dim;
    let items = if req.in_place && req.rms_weight_len.is_none() {
        rotary / 2 // pass-through lanes already hold their values
    } else {
        head_dim.div_ceil(2)
    };
    // The kernels index rows and items in 32 bits (offsets in 64).
    if rows * items > ROPE_MAX_ELEMS {
        return Err(RopeRefusal::TooLarge(rows * items));
    }
    let blocks = match req.rms_weight_len {
        Some(_) => rows.div_ceil(RMS_NORM_ROPE_WARPS as usize),
        None => (rows * items)
            .div_ceil(ROPE_THREADS as usize)
            .min(ROPE_MAX_BLOCKS as usize),
    } as u32;
    Ok(RopePlan {
        dtype: req.dtype,
        rows,
        n: order.map(|a| dims[a]),
        x_stride,
        y_stride,
        t_stride: pick(&table_stride),
        t_lane: req.cos.stride[2],
        x_offset: req.x.offset,
        head_dim,
        rotary_dim: rotary,
        items,
        full_table: t[2] == rotary,
        split: req.spec.pairing == RopePairing::SplitHalf,
        native: req.spec.math == RopeMath::Native && req.dtype != DType::F32,
        in_place: req.in_place,
        blocks,
    })
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use crate::fused_path::Attempt;
    use crate::nvrtc::KernelCompileError;
    use candle_core::cuda_backend::cudarc;
    use candle_core::cuda_backend::CudaStorageSlice;
    use candle_core::op::BackpropOp;
    use candle_core::{
        CpuStorage, CudaDevice, CudaStorage, Device, InplaceOp1, Result, Storage, Tensor,
    };
    use cudarc::driver::{CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};

    /// The six entry points of one device's module, resolved once.
    struct Functions {
        rope: [CudaFunction; 3],
        rms: [CudaFunction; 3],
    }

    impl Functions {
        fn resolve(dev: &CudaDevice) -> std::result::Result<Self, KernelCompileError> {
            let m = MEDIA_ROPE_SRC.compiled(dev)?;
            Ok(Self {
                rope: [
                    m.function("media_rope_f32")?,
                    m.function("media_rope_bf16")?,
                    m.function("media_rope_f16")?,
                ],
                rms: [
                    m.function("media_rms_norm_rope_f32")?,
                    m.function("media_rms_norm_rope_bf16")?,
                    m.function("media_rms_norm_rope_f16")?,
                ],
            })
        }

        fn pick(&self, rms: bool, dtype: DType) -> &CudaFunction {
            let table = if rms { &self.rms } else { &self.rope };
            match dtype {
                DType::BF16 => &table[1],
                DType::F16 => &table[2],
                _ => &table[0],
            }
        }
    }

    /// Per-ordinal function tables (E8: one module per device, shared by every layer); only
    /// successes are held — a compile failure stays cached in the nvrtc seam.
    fn functions(dev: &CudaDevice) -> std::result::Result<&'static Functions, KernelCompileError> {
        static TABLES: OnceLock<RwLock<HashMap<usize, &'static Functions>>> = OnceLock::new();
        let tables = TABLES.get_or_init(Default::default);
        let ordinal = dev.cuda_stream().context().ordinal();
        if let Some(t) = tables
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .get(&ordinal)
        {
            return Ok(t);
        }
        let resolved = Functions::resolve(dev)?;
        let mut tables = tables.write().unwrap_or_else(|p| p.into_inner());
        Ok(*tables
            .entry(ordinal)
            .or_insert_with(|| Box::leak(Box::new(resolved))))
    }

    fn drv(e: cudarc::driver::DriverError) -> candle_core::Error {
        candle_core::Error::Cuda(format!("fused media rope: {e:?}").into())
    }

    fn i64s(v: [usize; 3]) -> [i64; 3] {
        v.map(|s| s as i64)
    }

    /// The f32 operands the kernel reads besides `x`: the tables (each from its own offset, with
    /// the plan's table strides) and, for the RMSNorm kernel, the contiguous weight and eps.
    struct Aux {
        cos: Tensor,
        sin: Tensor,
        weight: Option<(Tensor, f32)>,
    }

    fn f32_slice<'a>(storage: &'a Storage, what: &str) -> Result<&'a CudaSlice<f32>> {
        match storage {
            Storage::Cuda(cs) => cs.as_cuda_slice::<f32>(),
            _ => candle_core::bail!("fused media rope: {what} is not on CUDA"),
        }
    }

    /// Launch the planned kernel reading `x` and writing `y` (the same slice in place: every
    /// thread writes only lanes it alone reads, see the source).
    fn launch<T>(
        dev: &CudaDevice,
        func: &CudaFunction,
        plan: &RopePlan,
        x: &CudaSlice<T>,
        y: &CudaSlice<T>,
        aux: &Aux,
    ) -> Result<()> {
        let (cos_g, cos_l) = aux.cos.storage_and_layout();
        let (sin_g, sin_l) = aux.sin.storage_and_layout();
        let cos = f32_slice(&cos_g, "cos")?.slice(cos_l.start_offset()..);
        let sin = f32_slice(&sin_g, "sin")?.slice(sin_l.start_offset()..);
        let weight_guard = aux.weight.as_ref().map(|(w, eps)| {
            let (g, l) = w.storage_and_layout();
            (g, l.start_offset(), *eps)
        });
        let n1 = plan.n[1] as i32;
        let n2 = plan.n[2] as i32;
        let (xo, yo) = (plan.x_offset as i64, plan.y_offset() as i64);
        let [xs0, xs1, xs2] = i64s(plan.x_stride);
        let [ys0, ys1, ys2] = i64s(plan.y_stride);
        let [ts0, ts1, ts2] = i64s(plan.t_stride);
        let (to, tl) = (0i64, plan.t_lane as i64);
        let (d, rd) = (plan.head_dim as i32, plan.rotary_dim as i32);
        let split = i32::from(plan.split);
        let full = i32::from(plan.full_table);
        let native = i32::from(plan.native);
        let rows = plan.rows as i64;
        let n_items = (plan.rows * plan.items) as i64;
        let items = plan.items as i32;
        let stream = dev.cuda_stream();
        let weight = match &weight_guard {
            Some((g, offset, eps)) => Some((f32_slice(g, "weight")?.slice(*offset..), *eps)),
            None => None,
        };
        let mut b = stream.launch_builder(func);
        b.arg(x).arg(y);
        let block_dim = match &weight {
            Some((w, _)) => {
                b.arg(w).arg(&cos).arg(&sin).arg(&rows);
                (32, RMS_NORM_ROPE_WARPS, 1)
            }
            None => {
                b.arg(&cos).arg(&sin).arg(&n_items).arg(&items);
                (ROPE_THREADS, 1, 1)
            }
        };
        b.arg(&n1).arg(&n2);
        b.arg(&xo).arg(&xs0).arg(&xs1).arg(&xs2);
        b.arg(&yo).arg(&ys0).arg(&ys1).arg(&ys2);
        b.arg(&to).arg(&ts0).arg(&ts1).arg(&ts2).arg(&tl);
        b.arg(&d).arg(&rd).arg(&split).arg(&full).arg(&native);
        if let Some((_, eps)) = &weight {
            b.arg(eps);
        }
        let cfg = LaunchConfig {
            grid_dim: (plan.blocks, 1, 1),
            block_dim,
            shared_mem_bytes: 0,
        };
        unsafe { b.launch(cfg) }.map_err(drv)?;
        Ok(())
    }

    /// The in-place form: candle hands `x`'s storage over for writing (`Tensor::inplace_op1`).
    struct InPlace<'a> {
        dev: &'a CudaDevice,
        func: &'a CudaFunction,
        plan: RopePlan,
        aux: &'a Aux,
    }

    impl InplaceOp1 for InPlace<'_> {
        fn name(&self) -> &'static str {
            "fused-media-rope-in-place"
        }

        fn cpu_fwd(&self, _: &mut CpuStorage, _: &Layout) -> Result<()> {
            candle_core::bail!("the fused media rope is CUDA-only")
        }

        fn cuda_fwd(&self, storage: &mut CudaStorage, _: &Layout) -> Result<()> {
            let (dev, func, plan, aux) = (self.dev, self.func, &self.plan, self.aux);
            match &storage.slice {
                CudaStorageSlice::F32(s) => launch(dev, func, plan, s, s, aux),
                CudaStorageSlice::BF16(s) => launch(dev, func, plan, s, s, aux),
                CudaStorageSlice::F16(s) => launch(dev, func, plan, s, s, aux),
                _ => candle_core::bail!("fused media rope: unserved dtype {:?}", plan.dtype),
            }
        }
    }

    fn wrap<T: candle_core::cuda_backend::CudaDType>(
        slice: CudaSlice<T>,
        dev: &CudaDevice,
        dims: &[usize],
    ) -> Tensor {
        Tensor::from_storage(
            Storage::Cuda(CudaStorage::wrap_cuda_slice(slice, dev.clone())),
            dims,
            BackpropOp::none(),
            false,
        )
    }

    /// The pure request for `x` with these tables.
    struct Call<'a> {
        x: &'a Tensor,
        tracks_grad: bool,
        spec: RopeSpec,
        in_place: bool,
        rms_weight_len: Option<usize>,
    }

    impl Call<'_> {
        fn plan(&self, cos: &Tensor, sin: &Tensor) -> std::result::Result<RopePlan, RopeRefusal> {
            check_rope(&RopeRequest {
                on_cuda: true,
                dtype: self.x.dtype(),
                tracks_grad: self.tracks_grad,
                x: self.x.layout().into(),
                cos: cos.layout().into(),
                sin_dims: sin.dims(),
                spec: self.spec,
                in_place: self.in_place,
                rms_weight_len: self.rms_weight_len,
            })
        }
    }

    /// Plan, prepare the auxiliary operands and run one kernel; `Attempt::Reference` for an
    /// unserved input or a cached compile failure.
    fn run(
        x: &Tensor,
        weight: Option<(&Tensor, f64)>,
        cos: &Tensor,
        sin: &Tensor,
        spec: &RopeSpec,
        in_place: bool,
    ) -> Result<Attempt<Tensor>> {
        let Device::Cuda(dev) = x.device() else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let same = |t: &Tensor| t.device().same_device(x.device());
        if !same(cos) || !same(sin) || weight.is_some_and(|(w, _)| !same(w)) {
            return Ok(Attempt::Reference(RopeRefusal::NotCuda.reason()));
        }
        if weight.is_some_and(|(w, _)| w.rank() != 1) {
            return Ok(Attempt::Reference(RopeRefusal::WeightShape.reason()));
        }
        let call = Call {
            x,
            tracks_grad: x.track_op()
                || cos.track_op()
                || sin.track_op()
                || weight.is_some_and(|(w, _)| w.track_op()),
            spec: *spec,
            in_place,
            rms_weight_len: weight.map(|(w, _)| w.elem_count()),
        };
        if let Err(refusal) = call.plan(cos, sin) {
            return Ok(Attempt::Reference(refusal.reason()));
        }
        let f = match functions(dev) {
            Ok(f) => f,
            Err(e) => return Ok(Attempt::Reference(e.into())),
        };
        // The kernel reads f32 tables sharing one stride set; anything else is made so (the
        // tables are [1|B, S, W], small next to x).
        let (cos, sin) = if cos.dtype() == DType::F32
            && sin.dtype() == DType::F32
            && cos.stride() == sin.stride()
        {
            (cos.clone(), sin.clone())
        } else {
            (
                cos.to_dtype(DType::F32)?.contiguous()?,
                sin.to_dtype(DType::F32)?.contiguous()?,
            )
        };
        let plan = match call.plan(&cos, &sin) {
            Ok(plan) => plan,
            Err(refusal) => return Ok(Attempt::Reference(refusal.reason())),
        };
        let weight = match weight {
            Some((w, eps)) => Some((w.to_dtype(DType::F32)?.contiguous()?, eps as f32)),
            None => None,
        };
        let func = f.pick(weight.is_some(), plan.dtype);
        let aux = Aux { cos, sin, weight };
        if in_place {
            x.inplace_op1(&InPlace {
                dev,
                func,
                plan,
                aux: &aux,
            })?;
            return Ok(Attempt::Fused(x.clone()));
        }
        let (storage, _) = x.storage_and_layout();
        let Storage::Cuda(cs) = &*storage else {
            return Ok(Attempt::Reference(PathReason::NotCuda));
        };
        let elems = x.elem_count();
        let out = match &cs.slice {
            CudaStorageSlice::F32(s) => {
                let y = unsafe { dev.alloc::<f32>(elems) }?;
                launch(dev, func, &plan, s, &y, &aux)?;
                wrap(y, dev, x.dims())
            }
            CudaStorageSlice::BF16(s) => {
                let y = unsafe { dev.alloc::<half::bf16>(elems) }?;
                launch(dev, func, &plan, s, &y, &aux)?;
                wrap(y, dev, x.dims())
            }
            CudaStorageSlice::F16(s) => {
                let y = unsafe { dev.alloc::<half::f16>(elems) }?;
                launch(dev, func, &plan, s, &y, &aux)?;
                wrap(y, dev, x.dims())
            }
            _ => return Ok(Attempt::Reference(PathReason::Dtype(plan.dtype))),
        };
        Ok(Attempt::Fused(out))
    }

    /// The fused RoPE of `x` (rank 4 in `spec.layout`) with `cos` / `sin` as
    /// `[1 | batch, seq, rotary/2 | rotary]`: `Attempt::Fused` with the rotated tensor, or
    /// `Attempt::Reference(reason)` when it does not serve the call. `in_place` writes the result
    /// into `x`'s buffer and returns `x` (the caller must own every element of that view). It
    /// neither consults the switch nor records the outcome — [`ROPE_PATH`]'s `dispatch` does both.
    pub fn fused_rope(
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        spec: &RopeSpec,
        in_place: bool,
    ) -> Result<Attempt<Tensor>> {
        run(x, None, cos, sin, spec, in_place)
    }

    /// The fused per-head RMSNorm (`weight` `[head_dim]`, `eps`) followed by the RoPE of
    /// [`fused_rope`], with its contract ([`RMS_NORM_ROPE_PATH`] dispatches it).
    pub fn fused_rms_norm_rope(
        x: &Tensor,
        weight: &Tensor,
        eps: f64,
        cos: &Tensor,
        sin: &Tensor,
        spec: &RopeSpec,
        in_place: bool,
    ) -> Result<Attempt<Tensor>> {
        run(x, Some((weight, eps)), cos, sin, spec, in_place)
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::{fused_rms_norm_rope, fused_rope};

#[cfg(test)]
mod tests {
    use super::*;

    fn contiguous_stride(dims: &[usize]) -> Vec<usize> {
        let mut s = vec![1usize; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
        s
    }

    struct Case {
        x_dims: Vec<usize>,
        x_stride: Vec<usize>,
        x_offset: usize,
        t_dims: Vec<usize>,
        t_stride: Vec<usize>,
    }

    impl Case {
        fn bshd(b: usize, s: usize, h: usize, d: usize, w: usize) -> Self {
            let x_dims = vec![b, s, h, d];
            let t_dims = vec![1, s, w];
            Self {
                x_stride: contiguous_stride(&x_dims),
                t_stride: contiguous_stride(&t_dims),
                x_dims,
                x_offset: 0,
                t_dims,
            }
        }

        fn req(&self, spec: RopeSpec, in_place: bool, rms: Option<usize>) -> RopeRequest<'_> {
            RopeRequest {
                on_cuda: true,
                dtype: DType::BF16,
                tracks_grad: false,
                x: RopeOperand {
                    dims: &self.x_dims,
                    stride: &self.x_stride,
                    offset: self.x_offset,
                },
                cos: RopeOperand {
                    dims: &self.t_dims,
                    stride: &self.t_stride,
                    offset: 0,
                },
                sin_dims: &self.t_dims,
                spec,
                in_place,
                rms_weight_len: rms,
            }
        }
    }

    const BSHD: RopeSpec = RopeSpec::interleaved(RopeLayout::Bshd);

    #[test]
    fn the_plan_refuses_what_the_kernels_do_not_serve() {
        let c = Case::bshd(1, 6, 3, 128, 64);
        let ok = c.req(BSHD, false, None);
        assert!(check_rope(&ok).is_ok());
        let mut r = ok;
        r.on_cuda = false;
        assert_eq!(check_rope(&r), Err(RopeRefusal::NotCuda));
        for dtype in [DType::U8, DType::F64, DType::I64, DType::F8E4M3] {
            let mut r = ok;
            r.dtype = dtype;
            assert_eq!(check_rope(&r), Err(RopeRefusal::Dtype(dtype)));
        }
        let mut r = ok;
        r.tracks_grad = true;
        assert_eq!(check_rope(&r), Err(RopeRefusal::Autograd));
        // Rotary dim: odd, zero, wider than the head.
        for rd in [0, 7, 130] {
            let r = c.req(BSHD.with_rotary_dim(rd), false, None);
            assert!(
                matches!(check_rope(&r), Err(RopeRefusal::RotaryDim { .. })),
                "{rd}"
            );
        }
        // Table width must be rotary/2 or rotary, batch 1 or B, seq S.
        let bad_w = Case::bshd(1, 6, 3, 128, 48);
        assert_eq!(
            check_rope(&bad_w.req(BSHD, false, None)),
            Err(RopeRefusal::TableShape)
        );
        let mut bad_s = Case::bshd(1, 6, 3, 128, 64);
        bad_s.t_dims = vec![1, 5, 64];
        assert_eq!(
            check_rope(&bad_s.req(BSHD, false, None)),
            Err(RopeRefusal::TableShape)
        );
        let mut sin_differs = ok;
        let other = [1usize, 6, 128];
        sin_differs.sin_dims = &other;
        assert_eq!(check_rope(&sin_differs), Err(RopeRefusal::TableShape));
        // RMS weight and head-dim limit.
        assert_eq!(
            check_rope(&c.req(BSHD, false, Some(64))),
            Err(RopeRefusal::WeightShape)
        );
        let wide = Case::bshd(1, 2, 1, 1024, 512);
        assert_eq!(
            check_rope(&wide.req(BSHD, false, Some(1024))),
            Err(RopeRefusal::HeadDimTooLarge(1024))
        );
        assert!(
            check_rope(&wide.req(BSHD, false, None)).is_ok(),
            "rope alone serves it"
        );
        // Lane stride.
        let mut lanes = Case::bshd(1, 6, 3, 128, 64);
        lanes.x_stride = vec![6 * 3 * 256, 3 * 256, 256, 2];
        assert_eq!(
            check_rope(&lanes.req(BSHD, false, None)),
            Err(RopeRefusal::LaneStride)
        );
        // Rank and emptiness.
        let mut r3 = Case::bshd(1, 6, 3, 128, 64);
        r3.x_dims = vec![6, 3, 128];
        r3.x_stride = contiguous_stride(&r3.x_dims);
        assert_eq!(
            check_rope(&r3.req(BSHD, false, None)),
            Err(RopeRefusal::Rank(3))
        );
        let empty = Case::bshd(1, 0, 3, 128, 64);
        assert_eq!(
            check_rope(&empty.req(BSHD, false, None)),
            Err(RopeRefusal::Empty)
        );
        // E6.
        let huge = Case::bshd(1, 1 << 16, 256, 128, 64);
        assert!(matches!(
            check_rope(&huge.req(BSHD, false, None)),
            Err(RopeRefusal::TooLarge(_))
        ));
        // A broadcast (stride-0) view cannot be written in place; out of place it is served.
        let mut bcast = Case::bshd(1, 6, 3, 128, 64);
        bcast.x_stride = vec![6 * 128, 128, 0, 1];
        assert_eq!(
            check_rope(&bcast.req(BSHD, true, None)),
            Err(RopeRefusal::Overlapping)
        );
        assert!(check_rope(&bcast.req(BSHD, false, None)).is_ok());
    }

    #[test]
    fn refusals_map_to_typed_path_reasons() {
        assert_eq!(RopeRefusal::NotCuda.reason().label(), "not_cuda");
        assert_eq!(RopeRefusal::Dtype(DType::U8).reason().label(), "dtype");
        assert_eq!(RopeRefusal::Autograd.reason().label(), "autograd");
        for shape in [
            RopeRefusal::Rank(3),
            RopeRefusal::Empty,
            RopeRefusal::RotaryDim {
                rotary: 3,
                head_dim: 8,
            },
            RopeRefusal::TableShape,
            RopeRefusal::WeightShape,
            RopeRefusal::HeadDimTooLarge(2048),
        ] {
            assert_eq!(shape.reason().label(), "shape", "{shape}");
        }
        assert_eq!(RopeRefusal::LaneStride.reason().label(), "layout");
        assert_eq!(RopeRefusal::Overlapping.reason().label(), "layout");
        assert_eq!(
            RopeRefusal::TooLarge(1 << 40).reason().label(),
            "index_overflow"
        );
    }

    /// A permuted BSHD -> BHSD view walks rows in memory order (seq slowest, heads fastest), with
    /// the table broadcast over heads and batch, and a contiguous output in x's logical order.
    #[test]
    fn a_permuted_view_is_planned_in_memory_order() {
        let (b, s, h, d) = (2usize, 5, 3, 16);
        // BSHD memory, logical BHSD dims [b, h, s, d] with strides [s*h*d, d, h*d, 1].
        let c = Case {
            x_dims: vec![b, h, s, d],
            x_stride: vec![s * h * d, d, h * d, 1],
            x_offset: 7,
            t_dims: vec![1, s, d / 2],
            t_stride: vec![s * d / 2, d / 2, 1],
        };
        let spec = RopeSpec::split_half(RopeLayout::Bhsd);
        let p = check_rope(&c.req(spec, false, None)).unwrap();
        assert_eq!(
            p.n,
            [b, s, h],
            "batch, then seq (stride h*d), then heads (stride d)"
        );
        assert_eq!(p.x_stride, [s * h * d, h * d, d]);
        // Output contiguous in logical [b, h, s, d]: batch h*s*d, seq d, heads s*d.
        assert_eq!(p.y_stride, [h * s * d, d, s * d]);
        assert_eq!(p.t_stride, [0, d / 2, 0]);
        assert_eq!((p.x_offset, p.y_offset()), (7, 0));
        assert!(p.split && !p.full_table && !p.native);
        assert_eq!(p.items, d / 2);
        let inplace = check_rope(&c.req(spec, true, None)).unwrap();
        assert_eq!(inplace.y_stride, inplace.x_stride);
        assert_eq!(inplace.y_offset(), 7);
    }

    #[test]
    fn items_cover_every_lane_and_in_place_skips_the_pass_through() {
        // head 10, rotary 6: 3 rotated pairs + 4 pass-through lanes (2 items) out of place.
        let mut c = Case::bshd(1, 4, 2, 10, 3);
        let spec = BSHD.with_rotary_dim(6);
        assert_eq!(check_rope(&c.req(spec, false, None)).unwrap().items, 5);
        assert_eq!(check_rope(&c.req(spec, true, None)).unwrap().items, 3);
        // RMS: the norm touches every lane, in place too.
        assert_eq!(check_rope(&c.req(spec, true, Some(10))).unwrap().items, 5);
        // An odd head: 4 rotated pairs + 1 lane, with a full-width (per-lane) table.
        c = Case::bshd(1, 4, 2, 9, 8);
        let spec = BSHD.with_rotary_dim(8);
        let p = check_rope(&c.req(spec, false, None)).unwrap();
        assert_eq!(p.items, 5);
        assert!(p.full_table, "width == rotary reads one entry per lane");
        assert!(!p.native, "f32 math by default");
        let native = c.req(spec.with_math(RopeMath::Native), false, None);
        assert!(check_rope(&native).unwrap().native);
    }

    #[test]
    fn the_grid_is_capped_and_rms_rows_are_warp_blocked() {
        let c = Case::bshd(1, 4608, 48, 128, 64);
        let p = check_rope(&c.req(BSHD, true, None)).unwrap();
        assert_eq!(p.rows, 4608 * 48);
        assert_eq!(
            p.blocks as usize,
            (4608 * 48 * 64usize).div_ceil(ROPE_THREADS as usize)
        );
        let r = check_rope(&c.req(BSHD, true, Some(128))).unwrap();
        assert_eq!(
            r.blocks as usize,
            (4608 * 48usize).div_ceil(RMS_NORM_ROPE_WARPS as usize)
        );
        let big = Case::bshd(1, 1 << 15, 256, 128, 64);
        assert_eq!(
            check_rope(&big.req(BSHD, false, None)).unwrap().blocks,
            ROPE_MAX_BLOCKS
        );
    }

    #[test]
    fn overlap_detection() {
        assert!(view_is_non_overlapping(&[2, 3, 4], &[12, 4, 1]));
        assert!(
            view_is_non_overlapping(&[2, 3, 4], &[1, 8, 2]),
            "permuted, gapped"
        );
        assert!(
            !view_is_non_overlapping(&[2, 3, 4], &[12, 0, 1]),
            "broadcast"
        );
        assert!(
            !view_is_non_overlapping(&[4, 4], &[2, 1]),
            "overlapping rows"
        );
        assert!(
            view_is_non_overlapping(&[1, 3, 1], &[0, 1, 0]),
            "size-1 axes ignored"
        );
        // q of a fused qkv projection [b, s, 3, h, d] narrowed to one part: gapped, disjoint.
        assert!(view_is_non_overlapping(&[2, 5, 4, 8], &[5 * 96, 96, 8, 1]));
    }
}

//! The indexed (gathered) MoE GEMV (sc-24440, epic sc-24432): one projection of a decode step's
//! routed experts — every (token, slot) pair's expert applied to one activation row — as **one
//! launch**, with the routes read on the device and every expert weight read in place.
//!
//! A Mixture-of-Experts decode step routes each token to `slots` experts. Run as candle ops, that
//! is either a host read of the routes (to pick each expert's weight) or a gather of whole expert
//! matrices by a device index (`index_select` copies `slots` matrices per projection per token).
//! [`IndexedExperts`] instead holds, per projection of a bank, a **device table** of each
//! expert's weight address (built once at load) and launches a kernel that reads `ids[task]` on
//! the device and the weight at `table[ids[task]]` — no host read, no copy, and launch shapes that
//! depend only on the step shape, so a CUDA graph of the step replays at any routing.
//!
//! # Formats ([`IndexedFormat`])
//!
//! * **GGML** blocks — every type candle's fast MMVQ serves (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
//!   Q2_K..Q6_K): candle's own mat-vec code (`mmvq_gguf.cu`, copied verbatim into
//!   [`MOE_GEMV_SRC`]; a test re-derives it from the vendored file) on a Q8_1-quantized f32
//!   activation, f32 out. A pair's output is **bit-identical** to `QMatMul::forward` of that
//!   expert on that row, which is what the per-expert dispatch runs at decode.
//! * **Q8_0 dequant** — the MLX-affine Q8 tier, whose per-expert forward dequantizes the weight to
//!   the activation dtype and runs a dense matmul: the same rounding of each weight, f32
//!   accumulation in the kernel's fixed order.
//! * **Dense** f32 / bf16 / f16 — a stacked `[experts, n, k]` bank: f32 accumulation in a fixed
//!   order, one output rounding.
//! * **NVFP4** — the fused decode GEMV's own core (`nvfp4_gemv.cu`) per task, with the packed
//!   weight, block scales and per-tensor scale from device tables: bit-identical to
//!   `Nvfp4Weight::forward_gemv` of that expert on that row. bf16 activations only (the GEMV's
//!   contract).
//!
//! Every kernel is deterministic: a fixed reduction order, no atomics, so an eager launch and a
//! graph replay agree bit for bit.
//!
//! # Workspace
//!
//! A forward allocates its output (`tasks × n` of the output dtype) and, for GGML, the Q8_1 copy
//! of the activation ([`indexed_workspace_bytes`]) — the bytes a caller's admission prices.

use std::fmt;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Tensor};

use crate::nvfp4_weight::Nvfp4Weight;
use crate::nvrtc::{KernelCompileError, KernelSource};

/// The indexed MoE GEMV source (GGML, dense and Q8_0-dequant kernels); compiled once per device.
/// The NVFP4 kernel lives with the fused NVFP4 GEMV ([`NVFP4_GEMV_SRC`](crate::NVFP4_GEMV_SRC)),
/// whose core it shares.
pub const MOE_GEMV_SRC: KernelSource = KernelSource {
    name: "candle_quant_kernels_moe_gemv_v1",
    src: include_str!("moe_gemv.cu"),
    // `__dp4a` (sm_61) with candle's own pre-sm_61 fallback; shuffles; `cvt` f16 conversions.
    cc_floor: (7, 0),
};

/// Output rows per block of the dense and Q8_0-dequant kernels (`MOE_GEMV_WARPS`, one per warp).
pub const MOE_GEMV_ROWS_PER_BLOCK: usize = 4;

/// candle's `MATRIX_ROW_PADDING`: a Q8_1 activation row is padded to a multiple of this.
const MATRIX_ROW_PADDING: usize = 512;
/// Elements per Q8_1 block.
const Q8_1_BLOCK: usize = 32;
/// Bytes per Q8_1 block (an f16 scale, an f16 sum, 32 int8).
const Q8_1_BYTES: usize = 36;

/// The largest device-table footprint of one expert in one projection's [`IndexedExperts`]: an
/// NVFP4 bank's packed and block-scale addresses (8 bytes each) and its 4-byte per-tensor scale;
/// every other format keeps one 8-byte address. What a load prices per expert projection.
pub const MAX_TABLE_BYTES_PER_EXPERT: usize = 8 + 8 + 4;

/// Which activation row each task reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRows {
    /// One row per token: every slot of token `t` reads row `t` (the gate / up projections).
    PerToken,
    /// One row per (token, slot) pair, in task order (the down projection).
    PerSlot,
}

/// How an [`IndexedExperts`] bank stores its weights — which kernel serves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedFormat {
    /// Dense weights of this dtype (`F32`, `BF16` or `F16`).
    Dense(DType),
    /// GGML blocks of this type, multiplied by a Q8_1 activation (candle's decode MMVQ).
    Ggml(GgmlDType),
    /// Q8_0 blocks dequantized to the activation dtype (the MLX-affine Q8 tier's forward).
    Q8Dequant,
    /// NVFP4 (the fused decode GEMV's core).
    Nvfp4,
}

impl IndexedFormat {
    /// Stable lower-case label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::Ggml(_) => "ggml",
            Self::Q8Dequant => "q8_dequant",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// The activation dtype a forward takes when the caller's activations are `act` — and the
    /// dtype it returns — or `None` when this format cannot serve `act`: dense banks take their own
    /// dtype; GGML takes f32 (the caller widens, as `QuantizedLinear` does); Q8_0-dequant takes
    /// any of f32 / bf16 / f16; NVFP4 takes bf16 only.
    pub fn io_dtype(&self, act: DType) -> Option<DType> {
        match self {
            Self::Dense(d) => (*d == act).then_some(act),
            Self::Ggml(_) => Some(DType::F32),
            Self::Q8Dequant => matches!(act, DType::F32 | DType::BF16 | DType::F16).then_some(act),
            Self::Nvfp4 => (act == DType::BF16).then_some(act),
        }
    }
}

/// The GGML kernel for `dtype`, or `None` for a block type with no indexed kernel: the ten types
/// candle's fast MMVQ serves.
pub fn ggml_kernel(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::Q4_0 => "moe_mmvq_q4_0",
        GgmlDType::Q4_1 => "moe_mmvq_q4_1",
        GgmlDType::Q5_0 => "moe_mmvq_q5_0",
        GgmlDType::Q5_1 => "moe_mmvq_q5_1",
        GgmlDType::Q8_0 => "moe_mmvq_q8_0",
        GgmlDType::Q2K => "moe_mmvq_q2_k",
        GgmlDType::Q3K => "moe_mmvq_q3_k",
        GgmlDType::Q4K => "moe_mmvq_q4_k",
        GgmlDType::Q5K => "moe_mmvq_q5_k",
        GgmlDType::Q6K => "moe_mmvq_q6_k",
        _ => return None,
    })
}

/// Device bytes one [`IndexedExperts::forward`] allocates: the `tokens · slots × n` output in the
/// format's output dtype for activations of `act` (f32 for GGML), plus — GGML only — the Q8_1
/// copy of the activation (`rows × ⌈k⌉₅₁₂ / 32 × 36` bytes). `act` is the caller's activation
/// dtype; an unserved combination prices as f32 (the widest).
pub fn indexed_workspace_bytes(
    format: IndexedFormat,
    act: DType,
    tokens: usize,
    slots: usize,
    rows: MoeRows,
    n: usize,
    k: usize,
) -> usize {
    let tasks = tokens.saturating_mul(slots);
    let out_elem = format
        .io_dtype(act)
        .map_or(4, |d| d.size_in_bytes())
        .max(match format {
            IndexedFormat::Ggml(_) => 4,
            _ => 0,
        });
    let out = tasks.saturating_mul(n).saturating_mul(out_elem);
    let q8 = match format {
        IndexedFormat::Ggml(_) => {
            let x_rows = match rows {
                MoeRows::PerToken => tokens,
                MoeRows::PerSlot => tasks,
            };
            let padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
            x_rows.saturating_mul(padded / Q8_1_BLOCK * Q8_1_BYTES)
        }
        _ => 0,
    };
    out.saturating_add(q8)
}

/// Why a bank cannot be served by an indexed kernel, or why an input is refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MoeGemvRefusal {
    /// The bank (or the activation) is not on a CUDA device, or this is not a `cuda` build.
    NotCuda,
    /// A bank of zero experts.
    Empty,
    /// The experts do not share one format, shape and device.
    Mixed(String),
    /// A weight format with no indexed kernel (named).
    Format(String),
    /// An input that does not match the bank (shape, dtype, routes).
    Input(String),
}

impl MoeGemvRefusal {
    /// Stable lower-case label for telemetry.
    pub fn label(&self) -> &'static str {
        match self {
            Self::NotCuda => "not_cuda",
            Self::Empty => "empty",
            Self::Mixed(_) => "mixed",
            Self::Format(_) => "format",
            Self::Input(_) => "input",
        }
    }
}

impl fmt::Display for MoeGemvRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCuda => write!(f, "the indexed MoE GEMV serves CUDA banks only"),
            Self::Empty => write!(f, "an expert bank needs at least one expert"),
            Self::Mixed(why) => write!(f, "the experts are not one bank: {why}"),
            Self::Format(why) => write!(f, "no indexed MoE kernel for {why}"),
            Self::Input(why) => write!(f, "indexed MoE GEMV input refused: {why}"),
        }
    }
}

/// A failed indexed MoE GEMV: a typed refusal, the kernel's cached compile error, or a candle /
/// driver error.
#[derive(Debug)]
pub enum MoeGemvError {
    /// Bank or input outside the kernels' declared constraints.
    Refused(MoeGemvRefusal),
    /// The kernel source does not compile / load on this device (cached by the nvrtc seam).
    Compile(KernelCompileError),
    /// A candle / driver error (allocation, upload, launch).
    Candle(candle_core::Error),
}

impl MoeGemvError {
    /// Stable telemetry label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Refused(r) => r.label(),
            Self::Compile(e) => e.label(),
            Self::Candle(_) => "candle",
        }
    }
}

impl fmt::Display for MoeGemvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Refused(r) => write!(f, "{r}"),
            Self::Compile(e) => write!(f, "indexed MoE GEMV unavailable: {e}"),
            Self::Candle(e) => write!(f, "indexed MoE GEMV failed: {e}"),
        }
    }
}

impl std::error::Error for MoeGemvError {}

impl From<MoeGemvRefusal> for MoeGemvError {
    fn from(r: MoeGemvRefusal) -> Self {
        Self::Refused(r)
    }
}

impl From<KernelCompileError> for MoeGemvError {
    fn from(e: KernelCompileError) -> Self {
        Self::Compile(e)
    }
}

impl From<candle_core::Error> for MoeGemvError {
    fn from(e: candle_core::Error) -> Self {
        Self::Candle(e)
    }
}

impl From<MoeGemvError> for candle_core::Error {
    fn from(e: MoeGemvError) -> Self {
        match e {
            MoeGemvError::Candle(e) => e,
            other => candle_core::Error::Msg(other.to_string()),
        }
    }
}

/// One projection of a routed-expert bank, held so the indexed kernels read expert `e`'s weight
/// by a device id: the per-expert weight addresses as a device table (built once, at load) plus
/// the weights themselves, kept alive for as long as the table is.
pub struct IndexedExperts {
    format: IndexedFormat,
    experts: usize,
    n: usize,
    k: usize,
    #[cfg(feature = "cuda")]
    tables: cuda_impl::Tables,
}

impl fmt::Debug for IndexedExperts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexedExperts")
            .field("format", &self.format)
            .field("experts", &self.experts)
            .field("n", &self.n)
            .field("k", &self.k)
            .finish_non_exhaustive()
    }
}

/// The `(n, k)` every expert of a bank shares, or the [`MoeGemvRefusal::Mixed`] naming the
/// first that differs.
fn common_shape(
    shapes: impl Iterator<Item = (usize, usize)>,
) -> Result<(usize, usize), MoeGemvError> {
    let mut first = None;
    for (e, shape) in shapes.enumerate() {
        match first {
            None => first = Some(shape),
            Some(want) if want != shape => {
                return Err(MoeGemvRefusal::Mixed(format!(
                    "expert {e} is {shape:?}, expert 0 is {want:?}"
                ))
                .into())
            }
            Some(_) => {}
        }
    }
    first.ok_or_else(|| MoeGemvRefusal::Empty.into())
}

impl IndexedExperts {
    /// A dense bank stacked `[experts, n, k]` (`F32`, `BF16` or `F16`) on a CUDA device.
    pub fn dense(stacked: &Tensor) -> Result<Self, MoeGemvError> {
        let (experts, n, k) = stacked.dims3()?;
        if experts == 0 {
            return Err(MoeGemvRefusal::Empty.into());
        }
        let dtype = stacked.dtype();
        if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(MoeGemvRefusal::Format(format!("dense {dtype:?} experts")).into());
        }
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                format: IndexedFormat::Dense(dtype),
                experts,
                n,
                k,
                tables: cuda_impl::Tables::dense(stacked)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (n, k);
            Err(MoeGemvRefusal::NotCuda.into())
        }
    }

    /// A bank of GGML block-quantized experts (`[n, k]` each, one shared type with a kernel —
    /// [`ggml_kernel`]) on one CUDA device. `dequant` selects the [`IndexedFormat::Q8Dequant`]
    /// form (the MLX-affine Q8 tier: Q8_0 only) instead of the Q8_1-activation MMVQ.
    pub fn ggml(experts: &[std::sync::Arc<QTensor>], dequant: bool) -> Result<Self, MoeGemvError> {
        let dtype = experts.first().ok_or(MoeGemvRefusal::Empty)?.dtype();
        if let Some(e) = experts.iter().position(|q| q.dtype() != dtype) {
            return Err(MoeGemvRefusal::Mixed(format!(
                "expert {e} is {:?}, expert 0 is {dtype:?}",
                experts[e].dtype()
            ))
            .into());
        }
        let format = if dequant {
            if dtype != GgmlDType::Q8_0 {
                return Err(
                    MoeGemvRefusal::Format(format!("dequantized {dtype:?} experts")).into(),
                );
            }
            IndexedFormat::Q8Dequant
        } else {
            if ggml_kernel(dtype).is_none() {
                return Err(MoeGemvRefusal::Format(format!("{dtype:?} experts")).into());
            }
            IndexedFormat::Ggml(dtype)
        };
        let shapes = experts
            .iter()
            .map(|q| q.shape().dims2())
            .collect::<candle_core::Result<Vec<_>>>()?;
        let (n, k) = common_shape(shapes.into_iter())?;
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                format,
                experts: experts.len(),
                n,
                k,
                tables: cuda_impl::Tables::ggml(experts)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (format, n, k);
            Err(MoeGemvRefusal::NotCuda.into())
        }
    }

    /// A bank of NVFP4 experts (one shape, one CUDA device, no bias).
    pub fn nvfp4(experts: &[std::sync::Arc<Nvfp4Weight>]) -> Result<Self, MoeGemvError> {
        if experts.is_empty() {
            return Err(MoeGemvRefusal::Empty.into());
        }
        if experts.iter().any(|w| w.bias().is_some()) {
            return Err(MoeGemvRefusal::Format("NVFP4 experts with a bias".into()).into());
        }
        let (n, k) = common_shape(experts.iter().map(|w| w.shape()))?;
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                format: IndexedFormat::Nvfp4,
                experts: experts.len(),
                n,
                k,
                tables: cuda_impl::Tables::nvfp4(experts)?,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (n, k);
            Err(MoeGemvRefusal::NotCuda.into())
        }
    }

    /// The bank's storage format.
    pub fn format(&self) -> IndexedFormat {
        self.format
    }

    /// The number of experts.
    pub fn experts(&self) -> usize {
        self.experts
    }

    /// The `(n, k)` (output, input features) every expert shares.
    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Bytes of the device tables this bank uploaded at load (an 8-byte address per expert; NVFP4
    /// adds a block-scale address and a 4-byte per-tensor scale) — its whole resident cost beyond
    /// the weights it addresses.
    pub fn table_bytes(&self) -> usize {
        match self.format {
            IndexedFormat::Nvfp4 => self.experts * MAX_TABLE_BYTES_PER_EXPERT,
            _ => self.experts * 8,
        }
    }

    /// Device bytes one [`forward`](Self::forward) of `tokens × slots` tasks allocates
    /// ([`indexed_workspace_bytes`]) for activations of `act`.
    pub fn workspace_bytes(&self, act: DType, tokens: usize, slots: usize, rows: MoeRows) -> usize {
        indexed_workspace_bytes(self.format, act, tokens, slots, rows, self.n, self.k)
    }

    /// Whether the kernel serving this bank compiles and loads on its device (the nvrtc seam's
    /// cached outcome; compiled on the first call).
    pub fn available(&self) -> Result<(), MoeGemvError> {
        #[cfg(feature = "cuda")]
        {
            self.tables.functions(self.format).map(|_| ())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(MoeGemvRefusal::NotCuda.into())
        }
    }

    /// `y[task] = x[row(task)] · W[ids[task]]ᵀ` for every task `task = token · slots + slot`:
    /// `ids` is the `[tokens, slots]` u32 route table (each id `< experts`, as a top-k over the
    /// bank's router guarantees — the kernel reads `table[id]` unchecked), `x` is `[tokens, k]`
    /// ([`MoeRows::PerToken`]) or `[tokens · slots, k]` ([`MoeRows::PerSlot`]) in the format's
    /// [`io_dtype`](IndexedFormat::io_dtype). Returns `[tokens · slots, n]` in that dtype.
    pub fn forward(&self, x: &Tensor, ids: &Tensor, rows: MoeRows) -> Result<Tensor, MoeGemvError> {
        let (tokens, slots) = ids.dims2()?;
        if ids.dtype() != DType::U32 {
            return Err(
                MoeGemvRefusal::Input(format!("routes are {:?}, not U32", ids.dtype())).into(),
            );
        }
        let want_rows = match rows {
            MoeRows::PerToken => tokens,
            MoeRows::PerSlot => tokens * slots,
        };
        let (x_rows, k) = x.dims2()?;
        if x_rows != want_rows || k != self.k || tokens == 0 || slots == 0 {
            return Err(MoeGemvRefusal::Input(format!(
                "activation {:?} with routes {:?} ({rows:?}) for experts of {} inputs",
                x.dims(),
                ids.dims(),
                self.k
            ))
            .into());
        }
        if self.format.io_dtype(x.dtype()) != Some(x.dtype()) {
            return Err(MoeGemvRefusal::Input(format!(
                "{:?} activations for {} experts",
                x.dtype(),
                self.format.label()
            ))
            .into());
        }
        #[cfg(feature = "cuda")]
        {
            self.tables.forward(self, x, ids, rows, tokens, slots)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(MoeGemvRefusal::NotCuda.into())
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use candle_core::backend::BackendDevice;
    use candle_core::cuda_backend::cudarc;
    use candle_core::op::BackpropOp;
    use candle_core::{CudaDevice, CudaStorage, Device, Shape, Storage};
    use cudarc::driver::{CudaFunction, DevicePtr, LaunchConfig, PushKernelArg};
    use std::sync::{Arc, OnceLock};

    use crate::nvfp4::{NVFP4_BLOCK, SF_ATOM_COLS};
    use crate::{NVFP4_GEMV_ROWS_PER_BLOCK, NVFP4_GEMV_SRC, NVFP4_GEMV_THREADS};

    fn drv(e: impl fmt::Debug) -> MoeGemvError {
        MoeGemvError::Candle(candle_core::Error::Cuda(
            format!("indexed MoE GEMV kernel: {e:?}").into(),
        ))
    }

    /// What keeps the weights the tables address alive.
    #[allow(dead_code)] // held for their `Drop`, never read
    enum Owners {
        Dense(Tensor),
        Ggml(Vec<Arc<QTensor>>),
        Nvfp4(Vec<Arc<Nvfp4Weight>>),
    }

    /// The kernels one bank launches, resolved once per bank.
    pub(crate) struct Functions {
        /// The main kernel for each served activation dtype.
        kernels: Vec<(DType, CudaFunction)>,
        /// GGML only: the Q8_1 activation quantizer.
        quantize: Option<CudaFunction>,
    }

    impl Functions {
        fn kernel(&self, dtype: DType) -> Result<&CudaFunction, MoeGemvError> {
            self.kernels
                .iter()
                .find(|(d, _)| *d == dtype)
                .map(|(_, f)| f)
                .ok_or_else(|| MoeGemvRefusal::Input(format!("{dtype:?} activations")).into())
        }
    }

    pub(crate) struct Tables {
        dev: CudaDevice,
        /// I64 `[experts]`: each expert's weight address (NVFP4: its packed nibbles).
        weights: Tensor,
        /// NVFP4 only: I64 `[experts]` block-scale addresses, F32 `[experts]` per-tensor scales,
        /// the padded column count and the scale-atom count every expert shares.
        nvfp4: Option<(Tensor, Tensor, usize, usize)>,
        /// Dense only: 16-byte loads (a row is a whole number of 16-byte units and every expert
        /// starts 16-byte aligned) — fixed per bank, so a result never depends on a call.
        vec: bool,
        functions: OnceLock<Result<Functions, KernelCompileError>>,
        _owners: Owners,
    }

    fn cuda_dev(device: &Device) -> Result<CudaDevice, MoeGemvError> {
        match device {
            Device::Cuda(d) => Ok(d.clone()),
            _ => Err(MoeGemvRefusal::NotCuda.into()),
        }
    }

    fn upload_addresses(addresses: Vec<u64>, dev: &CudaDevice) -> Result<Tensor, MoeGemvError> {
        let n = addresses.len();
        let words = addresses.into_iter().map(|a| a as i64).collect::<Vec<_>>();
        Ok(Tensor::from_vec(words, n, &Device::Cuda(dev.clone()))?)
    }

    fn same_device(a: &Device, dev: &CudaDevice) -> bool {
        matches!(a, Device::Cuda(d) if d.same_device(dev))
    }

    fn cuda_storage(storage: &Storage) -> Result<&CudaStorage, MoeGemvError> {
        match storage {
            Storage::Cuda(c) => Ok(c),
            _ => Err(MoeGemvRefusal::NotCuda.into()),
        }
    }

    impl Tables {
        pub(crate) fn dense(stacked: &Tensor) -> Result<Self, MoeGemvError> {
            let dev = cuda_dev(stacked.device())?;
            let stacked = stacked.contiguous()?;
            let (experts, n, k) = stacked.dims3()?;
            let elem = stacked.dtype().size_in_bytes();
            let base = {
                let (s, l) = stacked.storage_and_layout();
                let c = cuda_storage(&s)?;
                let stream = dev.cuda_stream();
                let offset = l.start_offset();
                macro_rules! addr {
                    ($t:ty) => {
                        c.as_cuda_slice::<$t>()?
                            .slice(offset..)
                            .device_ptr(&stream)
                            .0
                    };
                }
                match stacked.dtype() {
                    DType::F32 => addr!(f32),
                    DType::BF16 => addr!(half::bf16),
                    DType::F16 => addr!(half::f16),
                    other => {
                        return Err(
                            MoeGemvRefusal::Format(format!("dense {other:?} experts")).into()
                        )
                    }
                }
            };
            let stride = (n * k * elem) as u64;
            let addresses = (0..experts as u64)
                .map(|e| base + e * stride)
                .collect::<Vec<_>>();
            let vec = (k * elem).is_multiple_of(16) && addresses.iter().all(|a| a % 16 == 0);
            Ok(Self {
                weights: upload_addresses(addresses, &dev)?,
                dev,
                nvfp4: None,
                vec,
                functions: OnceLock::new(),
                _owners: Owners::Dense(stacked),
            })
        }

        pub(crate) fn ggml(experts: &[Arc<QTensor>]) -> Result<Self, MoeGemvError> {
            let dev = cuda_dev(&experts[0].device())?;
            let mut addresses = Vec::with_capacity(experts.len());
            for (e, q) in experts.iter().enumerate() {
                if !same_device(&q.device(), &dev) {
                    return Err(
                        MoeGemvRefusal::Mixed(format!("expert {e} is on another device")).into(),
                    );
                }
                addresses.push(q.device_ptr()? as u64);
            }
            Ok(Self {
                weights: upload_addresses(addresses, &dev)?,
                dev,
                nvfp4: None,
                vec: false,
                functions: OnceLock::new(),
                _owners: Owners::Ggml(experts.to_vec()),
            })
        }

        pub(crate) fn nvfp4(experts: &[Arc<Nvfp4Weight>]) -> Result<Self, MoeGemvError> {
            let dev = cuda_dev(experts[0].device())?;
            let stream = dev.cuda_stream();
            let (mut packed, mut scales, mut gs) = (Vec::new(), Vec::new(), Vec::new());
            let cols_padded = experts[0].staged().shape_padded().1;
            for (e, w) in experts.iter().enumerate() {
                if !same_device(w.device(), &dev) {
                    return Err(
                        MoeGemvRefusal::Mixed(format!("expert {e} is on another device")).into(),
                    );
                }
                let staged = w.staged();
                if staged.shape_padded().1 != cols_padded {
                    return Err(MoeGemvRefusal::Mixed(format!(
                        "expert {e} is padded to {} columns, expert 0 to {cols_padded}",
                        staged.shape_padded().1
                    ))
                    .into());
                }
                packed.push(staged.packed_slice().device_ptr(&stream).0);
                scales.push(staged.scales_slice().device_ptr(&stream).0);
                gs.push(staged.global_scale());
            }
            let num_k_atoms = (cols_padded / NVFP4_BLOCK).div_ceil(SF_ATOM_COLS);
            let count = gs.len();
            let gs = Tensor::from_vec(gs, count, &Device::Cuda(dev.clone()))?;
            Ok(Self {
                weights: upload_addresses(packed, &dev)?,
                nvfp4: Some((
                    upload_addresses(scales, &dev)?,
                    gs,
                    cols_padded,
                    num_k_atoms,
                )),
                dev,
                vec: false,
                functions: OnceLock::new(),
                _owners: Owners::Nvfp4(experts.to_vec()),
            })
        }

        /// The kernels serving `format` (by activation dtype) and, for GGML, the activation
        /// quantizer — resolved once; a compile failure is the seam's cached error.
        pub(crate) fn functions(&self, format: IndexedFormat) -> Result<&Functions, MoeGemvError> {
            self.functions
                .get_or_init(|| {
                    let source = match format {
                        IndexedFormat::Nvfp4 => NVFP4_GEMV_SRC,
                        _ => MOE_GEMV_SRC,
                    };
                    let module = source.compiled(&self.dev)?;
                    let typed = |kind: &str, dtypes: &[DType]| {
                        dtypes
                            .iter()
                            .map(|&d| {
                                let tag = match d {
                                    DType::F32 => "f32",
                                    DType::BF16 => "bf16",
                                    _ => "f16",
                                };
                                Ok((d, module.function(&format!("moe_gemv_{kind}_{tag}"))?))
                            })
                            .collect::<Result<Vec<_>, KernelCompileError>>()
                    };
                    Ok(match format {
                        IndexedFormat::Ggml(d) => Functions {
                            kernels: vec![(
                                DType::F32,
                                module.function(ggml_kernel(d).expect("constructor checked"))?,
                            )],
                            quantize: Some(module.function("mmvq_gguf_quantize_q8_1_f32")?),
                        },
                        IndexedFormat::Nvfp4 => Functions {
                            kernels: vec![(
                                DType::BF16,
                                module.function("nvfp4_gemv_indexed_bf16")?,
                            )],
                            quantize: None,
                        },
                        IndexedFormat::Dense(d) => Functions {
                            kernels: typed("dense", &[d])?,
                            quantize: None,
                        },
                        IndexedFormat::Q8Dequant => Functions {
                            kernels: typed("q8_0", &[DType::F32, DType::BF16, DType::F16])?,
                            quantize: None,
                        },
                    })
                })
                .as_ref()
                .map_err(|e| MoeGemvError::Compile(e.clone()))
        }

        pub(crate) fn forward(
            &self,
            bank: &IndexedExperts,
            x: &Tensor,
            ids: &Tensor,
            rows: MoeRows,
            tokens: usize,
            slots: usize,
        ) -> Result<Tensor, MoeGemvError> {
            if !same_device(x.device(), &self.dev) || !same_device(ids.device(), &self.dev) {
                return Err(MoeGemvRefusal::NotCuda.into());
            }
            let functions = self.functions(bank.format)?;
            let main = functions.kernel(x.dtype())?;
            let dev = &self.dev;
            let stream = dev.cuda_stream();
            let (n, k) = (bank.n, bank.k);
            let tasks = tokens * slots;
            let ids = ids.contiguous()?;
            // A dense, 16-byte-aligned activation: the kernels read it as packed rows (and the
            // dense / NVFP4 kernels with 16-byte loads).
            let x = x.contiguous()?;
            let x = if !(x.layout().start_offset() * x.dtype().size_in_bytes()).is_multiple_of(16) {
                x.force_contiguous()?
            } else {
                x
            };
            let (n_i, k_i, slots_i) = (n as i32, k as i32, slots as i32);
            let per_slot = i32::from(rows == MoeRows::PerSlot);
            let (is, il) = ids.storage_and_layout();
            let ids_view = cuda_storage(&is)?
                .as_cuda_slice::<u32>()?
                .slice(il.start_offset()..);
            let (ws, wl) = self.weights.storage_and_layout();
            let table = cuda_storage(&ws)?
                .as_cuda_slice::<i64>()?
                .slice(wl.start_offset()..);
            let (xs, xl) = x.storage_and_layout();
            let xc = cuda_storage(&xs)?;
            let x_off = xl.start_offset();
            let wrap = |out: CudaStorage| {
                Tensor::from_storage(
                    Storage::Cuda(out),
                    Shape::from((tasks, n)),
                    BackpropOp::none(),
                    false,
                )
            };
            match bank.format {
                IndexedFormat::Ggml(_) => {
                    let quantize = functions
                        .quantize
                        .as_ref()
                        .expect("GGML resolves its quantizer");
                    let x_rows = match rows {
                        MoeRows::PerToken => tokens,
                        MoeRows::PerSlot => tasks,
                    };
                    // candle's fast-MMVQ geometry: rows padded to MATRIX_ROW_PADDING.
                    let k_padded = k.div_ceil(MATRIX_ROW_PADDING) * MATRIX_ROW_PADDING;
                    let stride_col_y = k_padded / Q8_1_BLOCK;
                    // SAFETY: the quantizer writes every block of every row, padding included.
                    let vy = unsafe { dev.alloc::<u8>(x_rows * stride_col_y * Q8_1_BYTES) }?;
                    let xv = xc.as_cuda_slice::<f32>()?.slice(x_off..);
                    let cfg = LaunchConfig {
                        grid_dim: (k_padded.div_ceil(256) as u32, x_rows as u32, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let kp_i = k_padded as i32;
                    let mut b = stream.launch_builder(quantize);
                    b.arg(&xv).arg(&vy).arg(&k_i).arg(&kp_i);
                    // SAFETY: matches `mmvq_gguf_quantize_q8_1_f32` in `moe_gemv.cu`.
                    unsafe { b.launch(cfg) }.map_err(drv)?;
                    // SAFETY: the kernel writes every output element.
                    let out = unsafe { dev.alloc::<f32>(tasks * n) }?;
                    let cfg = LaunchConfig {
                        grid_dim: (n as u32, tasks as u32, 1),
                        block_dim: (32, 4, 1),
                        shared_mem_bytes: 0,
                    };
                    let stride_i = stride_col_y as i32;
                    let mut b = stream.launch_builder(main);
                    b.arg(&table)
                        .arg(&vy)
                        .arg(&ids_view)
                        .arg(&out)
                        .arg(&k_i)
                        .arg(&n_i)
                        .arg(&stride_i)
                        .arg(&slots_i)
                        .arg(&per_slot);
                    // SAFETY: matches `moe_mmvq_*` in `moe_gemv.cu`.
                    unsafe { b.launch(cfg) }.map_err(drv)?;
                    drop(vy);
                    Ok(wrap(CudaStorage::wrap_cuda_slice(out, dev.clone())))
                }
                IndexedFormat::Dense(_) | IndexedFormat::Q8Dequant => {
                    let cfg = LaunchConfig {
                        grid_dim: (n.div_ceil(MOE_GEMV_ROWS_PER_BLOCK) as u32, tasks as u32, 1),
                        block_dim: (MOE_GEMV_ROWS_PER_BLOCK as u32 * 32, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let vec = i32::from(self.vec);
                    let dense = matches!(bank.format, IndexedFormat::Dense(_));
                    macro_rules! run {
                        ($t:ty) => {{
                            let xv = xc.as_cuda_slice::<$t>()?.slice(x_off..);
                            // SAFETY: the kernel writes every output element.
                            let out = unsafe { dev.alloc::<$t>(tasks * n) }?;
                            let mut b = stream.launch_builder(main);
                            b.arg(&table)
                                .arg(&xv)
                                .arg(&ids_view)
                                .arg(&out)
                                .arg(&n_i)
                                .arg(&k_i)
                                .arg(&slots_i)
                                .arg(&per_slot);
                            if dense {
                                b.arg(&vec);
                            }
                            // SAFETY: matches `moe_gemv_{dense,q8_0}_*` in `moe_gemv.cu`.
                            unsafe { b.launch(cfg) }.map_err(drv)?;
                            Ok(wrap(CudaStorage::wrap_cuda_slice(out, dev.clone())))
                        }};
                    }
                    match x.dtype() {
                        DType::F32 => run!(f32),
                        DType::BF16 => run!(half::bf16),
                        DType::F16 => run!(half::f16),
                        other => {
                            Err(MoeGemvRefusal::Input(format!("{other:?} activations")).into())
                        }
                    }
                }
                IndexedFormat::Nvfp4 => {
                    let (scales, gs, cols_padded, num_k_atoms) =
                        self.nvfp4.as_ref().expect("an NVFP4 bank has its tables");
                    let (ss, sl) = scales.storage_and_layout();
                    let scales = cuda_storage(&ss)?
                        .as_cuda_slice::<i64>()?
                        .slice(sl.start_offset()..);
                    let (gss, gsl) = gs.storage_and_layout();
                    let gs = cuda_storage(&gss)?
                        .as_cuda_slice::<f32>()?
                        .slice(gsl.start_offset()..);
                    let xv = xc.as_cuda_slice::<half::bf16>()?.slice(x_off..);
                    // SAFETY: the kernel writes every output element (N % 16 == 0 for NVFP4).
                    let out = unsafe { dev.alloc::<half::bf16>(tasks * n) }?;
                    let cfg = LaunchConfig {
                        grid_dim: ((n / NVFP4_GEMV_ROWS_PER_BLOCK) as u32, tasks as u32, 1),
                        block_dim: (NVFP4_GEMV_THREADS, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let (cp, nka) = (*cols_padded as i32, *num_k_atoms as i32);
                    let vec_x = i32::from(k.is_multiple_of(8));
                    let mut b = stream.launch_builder(main);
                    b.arg(&table)
                        .arg(&scales)
                        .arg(&gs)
                        .arg(&xv)
                        .arg(&ids_view)
                        .arg(&out)
                        .arg(&n_i)
                        .arg(&k_i)
                        .arg(&cp)
                        .arg(&nka)
                        .arg(&vec_x)
                        .arg(&slots_i)
                        .arg(&per_slot);
                    // SAFETY: matches `nvfp4_gemv_indexed_bf16` in `nvfp4_gemv.cu`.
                    unsafe { b.launch(cfg) }.map_err(drv)?;
                    Ok(wrap(CudaStorage::wrap_cuda_slice(out, dev.clone())))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// The identifier renames that turn candle's `mmvq_gguf.cu` text into the header-free text of
    /// [`MOE_GEMV_SRC`]: whole identifiers only.
    fn rename(text: &str) -> String {
        const MAP: [(&str, &str); 6] = [
            ("half", "gg_half"),
            ("half2", "gg_half2"),
            ("__half22float2", "gg_half22float2"),
            ("__low2half", "gg_low2half"),
            ("__low2float", "gg_low2float"),
            ("__half2float", "gg_half2float"),
        ];
        let mut out = String::with_capacity(text.len());
        let mut ident = String::new();
        let flush = |ident: &mut String, out: &mut String| {
            let mapped = MAP
                .iter()
                .find(|(from, _)| *from == ident.as_str())
                .map_or(ident.as_str(), |(_, to)| to);
            out.push_str(mapped);
            ident.clear();
        };
        for c in text.chars() {
            if c.is_ascii_alphanumeric() || c == '_' {
                ident.push(c);
            } else {
                flush(&mut ident, &mut out);
                out.push(c);
            }
        }
        flush(&mut ident, &mut out);
        out
    }

    fn between<'a>(text: &'a str, start: &str, end: &str) -> &'a str {
        let a = text
            .find(start)
            .unwrap_or_else(|| panic!("marker {start:?}"));
        let b = a + text[a..]
            .find(end)
            .unwrap_or_else(|| panic!("marker {end:?}"));
        &text[a..b]
    }

    /// The GGML mat-vec code in [`MOE_GEMV_SRC`] is candle's own (the vendored candle-kernels
    /// `mmvq_gguf.cu` that `QMatMul` runs at decode), not a re-implementation: re-derived here from
    /// the vendored file with the documented fp16 renames, both derived segments must appear in
    /// the source verbatim. A drift on either side (a candle re-vendor, an edit here) fails.
    #[test]
    fn the_ggml_code_is_candles_mmvq_verbatim() {
        let vendored = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../media/candle-gen/vendor/candle-kernels/src/mmvq_gguf.cu");
        // Line endings are the checkout's (CRLF on a Windows runner): compare without them.
        let vendored = std::fs::read_to_string(&vendored)
            .unwrap_or_else(|e| panic!("{}: {e}", vendored.display()))
            .replace('\r', "");
        let source = MOE_GEMV_SRC.src.replace('\r', "");
        let core = between(
            &vendored,
            "// Constants, types, and helpers shared with the indexed MoE kernels.",
            "\n// ---------------------------------------------------------------------------\n// Extern-C kernel entry points",
        );
        let quantize = between(
            &vendored,
            "extern \"C\" __global__ void\nmmvq_gguf_quantize_q8_1_f32(",
            "\n\n// Host-side launchers",
        );
        for (name, segment) in [("core", core), ("quantize", quantize)] {
            let derived = rename(segment);
            assert!(
                source.contains(&derived),
                "the {name} segment of moe_gemv.cu is not candle's mmvq_gguf.cu text"
            );
        }
        // Every type candle's fast MMVQ serves has an indexed kernel, and nothing else claims one.
        for d in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            let kernel = ggml_kernel(d).unwrap();
            assert!(
                source.contains(&format!("MOE_MMVQ_ENTRY({}", &kernel[9..])),
                "{d:?}"
            );
        }
        for d in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q8_1,
            GgmlDType::Q8K,
        ] {
            assert_eq!(ggml_kernel(d), None, "{d:?}");
        }
    }

    #[test]
    fn io_dtypes_follow_each_formats_forward() {
        use IndexedFormat::*;
        assert_eq!(Dense(DType::BF16).io_dtype(DType::BF16), Some(DType::BF16));
        assert_eq!(Dense(DType::BF16).io_dtype(DType::F32), None);
        assert_eq!(Ggml(GgmlDType::Q4K).io_dtype(DType::BF16), Some(DType::F32));
        assert_eq!(Q8Dequant.io_dtype(DType::F16), Some(DType::F16));
        assert_eq!(Q8Dequant.io_dtype(DType::U32), None);
        assert_eq!(Nvfp4.io_dtype(DType::BF16), Some(DType::BF16));
        assert_eq!(Nvfp4.io_dtype(DType::F32), None);
        assert_eq!(
            [Dense(DType::F32), Ggml(GgmlDType::Q8_0), Q8Dequant, Nvfp4].map(|f| f.label()),
            ["dense", "ggml", "q8_dequant", "nvfp4"]
        );
    }

    /// The workspace a forward allocates: its output, plus GGML's Q8_1 activation rows padded to
    /// candle's 512.
    #[test]
    fn workspace_bytes_cover_the_output_and_the_q8_1_activation() {
        let (tokens, slots, n, k) = (2, 8, 512, 2048);
        assert_eq!(
            indexed_workspace_bytes(
                IndexedFormat::Dense(DType::BF16),
                DType::BF16,
                tokens,
                slots,
                MoeRows::PerToken,
                n,
                k
            ),
            tokens * slots * n * 2
        );
        let q8_rows = |rows: usize, k: usize| rows * k.div_ceil(512) * 512 / 32 * 36;
        assert_eq!(
            indexed_workspace_bytes(
                IndexedFormat::Ggml(GgmlDType::Q4K),
                DType::BF16,
                tokens,
                slots,
                MoeRows::PerToken,
                n,
                k
            ),
            tokens * slots * n * 4 + q8_rows(tokens, k)
        );
        assert_eq!(
            indexed_workspace_bytes(
                IndexedFormat::Ggml(GgmlDType::Q8_0),
                DType::BF16,
                tokens,
                slots,
                MoeRows::PerSlot,
                n,
                1408
            ),
            tokens * slots * n * 4 + q8_rows(tokens * slots, 1408)
        );
        assert_eq!(
            indexed_workspace_bytes(
                IndexedFormat::Nvfp4,
                DType::BF16,
                tokens,
                slots,
                MoeRows::PerSlot,
                n,
                k
            ),
            tokens * slots * n * 2
        );
    }

    /// No bank is built off CUDA, and a malformed bank is refused before any device work.
    #[test]
    fn banks_refuse_the_cpu_and_malformed_experts() {
        let dev = Device::Cpu;
        let stacked = Tensor::zeros((4, 32, 64), DType::F32, &dev).unwrap();
        assert_eq!(
            IndexedExperts::dense(&stacked).unwrap_err().label(),
            "not_cuda"
        );
        let u8s = Tensor::zeros((4, 32, 64), DType::U8, &dev).unwrap();
        assert_eq!(IndexedExperts::dense(&u8s).unwrap_err().label(), "format");
        let q = |rows: usize, dtype| {
            std::sync::Arc::new(
                QTensor::quantize(
                    &Tensor::zeros((rows, 256), DType::F32, &dev).unwrap(),
                    dtype,
                )
                .unwrap(),
            )
        };
        assert_eq!(
            IndexedExperts::ggml(&[], false).unwrap_err().label(),
            "empty"
        );
        assert_eq!(
            IndexedExperts::ggml(&[q(8, GgmlDType::Q8_0), q(8, GgmlDType::Q4K)], false)
                .unwrap_err()
                .label(),
            "mixed"
        );
        assert_eq!(
            IndexedExperts::ggml(&[q(8, GgmlDType::Q8_0), q(16, GgmlDType::Q8_0)], false)
                .unwrap_err()
                .label(),
            "mixed"
        );
        assert_eq!(
            IndexedExperts::ggml(&[q(8, GgmlDType::Q4K)], true)
                .unwrap_err()
                .label(),
            "format"
        );
        assert_eq!(
            IndexedExperts::ggml(&[q(8, GgmlDType::Q8_0)], false)
                .unwrap_err()
                .label(),
            "not_cuda"
        );
        assert_eq!(IndexedExperts::nvfp4(&[]).unwrap_err().label(), "empty");
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use candle_core::Device;
    use candle_core::Module;
    use std::sync::Arc;

    fn device() -> Option<Device> {
        match Device::new_cuda(0) {
            Ok(d) => Some(d),
            Err(_) => {
                eprintln!("skipping: no CUDA device");
                None
            }
        }
    }

    fn ramp(dims: &[usize], seed: u32, dev: &Device) -> Tensor {
        let n: usize = dims.iter().product();
        let mut x = seed.max(1);
        let data: Vec<f32> = (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 2001) as f32 / 1000.0 - 1.0
            })
            .collect();
        Tensor::from_vec(data, dims, dev).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    fn host(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// Routes `[tokens, slots]` over `experts`, distinct per token, deterministic.
    fn routes(tokens: usize, slots: usize, experts: usize, dev: &Device) -> (Tensor, Vec<u32>) {
        let ids: Vec<u32> = (0..tokens)
            .flat_map(|t| (0..slots).map(move |s| ((t * 5 + s * 3 + 1) % experts) as u32))
            .collect();
        (
            Tensor::from_vec(ids.clone(), (tokens, slots), dev).unwrap(),
            ids,
        )
    }

    /// The activation row task `task` reads.
    fn row(x: &Tensor, rows: MoeRows, task: usize, slots: usize) -> Tensor {
        let r = match rows {
            MoeRows::PerToken => task / slots,
            MoeRows::PerSlot => task,
        };
        x.narrow(0, r, 1).unwrap().contiguous().unwrap()
    }

    /// Every GGML type: each task's output is bit-identical to `QMatMul::forward` of its expert
    /// on its row (candle's decode MMVQ — the per-expert dispatch), for both row layouts and a K
    /// that needs candle's 512 padding; and two launches agree bit for bit.
    #[test]
    fn ggml_tasks_are_bit_identical_to_qmatmul_per_expert() {
        let Some(dev) = device() else { return };
        let (experts, n, slots, tokens) = (6usize, 48usize, 3usize, 2usize);
        for k in [512usize, 768] {
            for dtype in [
                GgmlDType::Q4_0,
                GgmlDType::Q4_1,
                GgmlDType::Q5_0,
                GgmlDType::Q5_1,
                GgmlDType::Q8_0,
                GgmlDType::Q2K,
                GgmlDType::Q3K,
                GgmlDType::Q4K,
                GgmlDType::Q5K,
                GgmlDType::Q6K,
            ] {
                if !k.is_multiple_of(dtype.block_size()) {
                    continue;
                }
                let qs: Vec<Arc<QTensor>> = (0..experts)
                    .map(|e| {
                        let w = ramp(&[n, k], 7 + e as u32, &Device::Cpu);
                        Arc::new(QTensor::quantize_onto(&w, dtype, &dev).unwrap())
                    })
                    .collect();
                let bank = IndexedExperts::ggml(&qs, false).unwrap();
                bank.available().unwrap();
                let (ids, host_ids) = routes(tokens, slots, experts, &dev);
                for rows in [MoeRows::PerToken, MoeRows::PerSlot] {
                    let x_rows = if rows == MoeRows::PerToken {
                        tokens
                    } else {
                        tokens * slots
                    };
                    let x = ramp(&[x_rows, k], 99, &dev);
                    let y = bank.forward(&x, &ids, rows).unwrap();
                    assert_eq!(y.dims(), &[tokens * slots, n]);
                    assert_eq!(bits(&y), bits(&bank.forward(&x, &ids, rows).unwrap()));
                    for (task, &e) in host_ids.iter().enumerate() {
                        let want = QMatMul::from_arc(qs[e as usize].clone())
                            .unwrap()
                            .forward(&row(&x, rows, task, slots))
                            .unwrap();
                        assert_eq!(
                            bits(&y.narrow(0, task, 1).unwrap()),
                            bits(&want),
                            "{dtype:?} k={k} {rows:?} task {task} (expert {e})"
                        );
                    }
                }
            }
        }
    }

    /// Dense and Q8_0-dequant banks against the per-expert matmul (the dequantize-then-matmul
    /// forward for Q8_0): within the output dtype's rounding plus the f32 accumulation order; and
    /// deterministic across launches.
    #[test]
    fn dense_and_q8_dequant_tasks_match_the_per_expert_matmul() {
        let Some(dev) = device() else { return };
        let (experts, n, slots, tokens) = (5usize, 40usize, 4usize, 3usize);
        for k in [256usize, 200] {
            for dtype in [DType::F32, DType::BF16, DType::F16] {
                let stacked = ramp(&[experts, n, k], 3, &dev)
                    .affine(0.05, 0.0)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                let dense = IndexedExperts::dense(&stacked).unwrap();
                let q8 = (k % 32 == 0).then(|| {
                    let qs: Vec<Arc<QTensor>> = (0..experts)
                        .map(|e| {
                            let w = stacked.get(e).unwrap().to_dtype(DType::F32).unwrap();
                            Arc::new(
                                QTensor::quantize_onto(
                                    &w.to_device(&Device::Cpu).unwrap(),
                                    GgmlDType::Q8_0,
                                    &dev,
                                )
                                .unwrap(),
                            )
                        })
                        .collect();
                    (IndexedExperts::ggml(&qs, true).unwrap(), qs)
                });
                let (ids, host_ids) = routes(tokens, slots, experts, &dev);
                for rows in [MoeRows::PerToken, MoeRows::PerSlot] {
                    let x_rows = if rows == MoeRows::PerToken {
                        tokens
                    } else {
                        tokens * slots
                    };
                    let x = ramp(&[x_rows, k], 41, &dev).to_dtype(dtype).unwrap();
                    let tol = match dtype {
                        DType::F32 => 1e-5,
                        DType::BF16 => 1.6e-2,
                        _ => 2e-3,
                    };
                    let check = |y: &Tensor, weight: &dyn Fn(usize) -> Tensor, what: &str| {
                        for (task, &e) in host_ids.iter().enumerate() {
                            let want = row(&x, rows, task, slots)
                                .matmul(&weight(e as usize).t().unwrap())
                                .unwrap();
                            let (got, want) = (host(&y.narrow(0, task, 1).unwrap()), host(&want));
                            for (g, w) in got.iter().zip(&want) {
                                assert!(
                                    (g - w).abs() <= tol * (1.0 + w.abs()),
                                    "{what} {dtype:?} k={k} {rows:?} task {task}: {g} vs {w}"
                                );
                            }
                        }
                    };
                    let y = dense.forward(&x, &ids, rows).unwrap();
                    assert_eq!(bits(&y), bits(&dense.forward(&x, &ids, rows).unwrap()));
                    check(&y, &|e| stacked.get(e).unwrap(), "dense");
                    if let Some((bank, qs)) = &q8 {
                        let y = bank.forward(&x, &ids, rows).unwrap();
                        assert_eq!(bits(&y), bits(&bank.forward(&x, &ids, rows).unwrap()));
                        check(
                            &y,
                            &|e| qs[e].dequantize(&dev).unwrap().to_dtype(dtype).unwrap(),
                            "q8_dequant",
                        );
                    }
                }
            }
        }
    }

    /// NVFP4: each task's output is bit-identical to the fused decode GEMV of its expert on its
    /// row — the same kernel core, the weight picked from the device tables.
    #[test]
    fn nvfp4_tasks_are_bit_identical_to_the_decode_gemv() {
        let Some(dev) = device() else { return };
        let Ok(ctx) = crate::Nvfp4Context::require(&dev) else {
            crate::skip_without_sm120("nvfp4 indexed MoE GEMV");
            return;
        };
        let (experts, n, k, slots, tokens) = (4usize, 64usize, 192usize, 2usize, 3usize);
        let ws: Vec<Arc<Nvfp4Weight>> = (0..experts)
            .map(|e| {
                let w = ramp(&[n, k], 11 + e as u32, &dev)
                    .to_dtype(DType::BF16)
                    .unwrap();
                Arc::new(Nvfp4Weight::quantize(&w, None, &ctx).unwrap())
            })
            .collect();
        let bank = IndexedExperts::nvfp4(&ws).unwrap();
        let (ids, host_ids) = routes(tokens, slots, experts, &dev);
        for rows in [MoeRows::PerToken, MoeRows::PerSlot] {
            let x_rows = if rows == MoeRows::PerToken {
                tokens
            } else {
                tokens * slots
            };
            let x = ramp(&[x_rows, k], 5, &dev).to_dtype(DType::BF16).unwrap();
            let y = bank.forward(&x, &ids, rows).unwrap();
            for (task, &e) in host_ids.iter().enumerate() {
                let want = ws[e as usize]
                    .forward_gemv(&row(&x, rows, task, slots))
                    .unwrap();
                assert_eq!(
                    bits(&y.narrow(0, task, 1).unwrap()),
                    bits(&want),
                    "{rows:?} task {task}"
                );
            }
        }
    }
}

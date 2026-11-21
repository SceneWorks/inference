//! LLM-specific projection extensions on the shared Candle layer spine (`SW-6957` / `SW-F335`).
//!
//! Unifies `Projection`, `KvProjection`, `QuantSpec`, `ProjectionFormat`, `ProjectionKind`,
//! `ProjectionTally`, `ProjectionCensus`, `WeightCensus`, `QuantizedLinear`, `Q8RowEmbedding`,
//! `to_ggml_block_tensor`, and `from_ggml_block_tensor` below both media and LLM runtimes while
//! preserving `Projection::{Dense, Quantized, Prism, Nvfp4}`'s exact operation order
//! (`AP13 class (a)`).

use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module};
use candle_quant_kernels::{nvfp4_shape_refusal, Nvfp4Context, Nvfp4Weight};
use thiserror::Error;

use crate::prism::PrismPackedWeight;
use crate::{
    AdaptLinear, Base, CandleError, DenseLinear, QLinear, QuantForward, QuantWeight, ScaleContext,
};

/// Errors surfaced by the Candle LLM projection and tensor primitives.
#[derive(Debug, Error)]
pub enum Error {
    /// A Candle tensor operation failed (shape mismatch, allocation, device error, …).
    #[error("candle op failed: {0}")]
    Candle(#[from] candle_core::Error),

    /// A required weight tensor was absent from the loaded checkpoint.
    #[error("missing tensor: {0}")]
    MissingTensor(String),

    /// Filesystem / checkpoint IO failure.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A model config (`config.json`) was malformed or missing a required field.
    #[error("invalid config: {0}")]
    Config(String),

    /// A requested capability is genuinely not supported (kept typed so callers can distinguish a
    /// capability gap from a generic failure — do not stringify into [`Error::Msg`]).
    #[error("unsupported: {0}")]
    Unsupported(String),

    /// A requested hardware capability is unavailable on the load device — kept typed so the
    /// refusal carries *which* capability and why (sc-24135: NVFP4 below sm_120 or on CPU). Maps to
    /// the contract's `Unsupported`, never to a silent fallback.
    #[error("unsupported: {0}")]
    Nvfp4Refused(#[from] candle_quant_kernels::Nvfp4Refusal),

    /// Generation was cancelled before it could run. Kept typed so the conformance suite and any
    /// consumer can tell cancellation apart from a real error.
    #[error("cancelled")]
    Canceled,

    /// A cache was asked to roll back to position `n` but holds no checkpoint there (a recurrent
    /// state cannot be inverted, so it refuses rather than approximate). `have` lists the positions
    /// it could return to besides `0` and its current length. Typed so a speculative engine can
    /// tell "no checkpoint" apart from a real failure and fall back (e.g. restore a clone).
    #[error(
        "no checkpoint at position {n} (have {have:?}); the recurrent state cannot be rolled back \
         without one"
    )]
    RollbackUnavailable { n: i32, have: Vec<i32> },

    /// A preallocated (static) KV cache was asked for more positions than it can hold — either at
    /// construction (`requested` capacity past the model's `max_position_embeddings`, reported as
    /// `capacity`) or at a step (`requested` = the position the step would end at, past the
    /// buffer's `capacity`). Raised **before** any device write, so the cache is untouched and the
    /// request fails closed instead of tripping an allocator OOM mid-decode (E6). Typed so admission
    /// and the decode drivers can tell a capacity bound from a real failure.
    #[error(
        "static KV cache capacity exceeded: {requested} positions requested, capacity {capacity}"
    )]
    KvCapacityExceeded { requested: usize, capacity: usize },

    /// Anything else, with a human-readable message.
    #[error("{0}")]
    Msg(String),
}

/// Result alias for Candle LLM projection primitives.
pub type Result<T> = std::result::Result<T, Error>;

/// The shared tensor-free config parser's two refusals keep their typed variants (sc-27255).
impl From<core_llm::model_config::ModelConfigError> for Error {
    fn from(e: core_llm::model_config::ModelConfigError) -> Self {
        match e {
            core_llm::model_config::ModelConfigError::Config(m) => Error::Config(m),
            core_llm::model_config::ModelConfigError::Unsupported(m) => Error::Unsupported(m),
        }
    }
}

/// A host image-preprocessing refusal (sc-27255) keeps its message.
impl From<core_llm::vision_preprocess::PreprocessError> for Error {
    fn from(e: core_llm::vision_preprocess::PreprocessError) -> Self {
        Error::Msg(e.0)
    }
}

/// A shared block-draft tree builder failure (the lattice and the sampler disagree) keeps its
/// message.
impl From<core_llm::LatticeTreeError> for Error {
    fn from(e: core_llm::LatticeTreeError) -> Self {
        Error::Msg(e.to_string())
    }
}

impl From<String> for Error {
    fn from(s: String) -> Self {
        Error::Msg(s)
    }
}

impl From<&str> for Error {
    fn from(s: &str) -> Self {
        Error::Msg(s.to_string())
    }
}

impl From<CandleError> for Error {
    fn from(e: CandleError) -> Self {
        match e {
            CandleError::Candle(c) => Error::Candle(c),
            CandleError::Unsupported(s) => Error::Unsupported(s),
            CandleError::Canceled => Error::Canceled,
            other => Error::Msg(other.to_string()),
        }
    }
}

impl From<Error> for CandleError {
    fn from(e: Error) -> Self {
        match e {
            Error::Candle(c) => CandleError::Candle(c),
            Error::Unsupported(s) => CandleError::Unsupported(s),
            Error::Nvfp4Refused(r) => CandleError::Unsupported(r.to_string()),
            Error::Canceled => CandleError::Canceled,
            other => CandleError::Msg(other.to_string()),
        }
    }
}

/// Group-wise quantization spec, mapped to a Candle GGML dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// The target GGML block-quant dtype.
    pub dtype: GgmlDType,
    /// Source group size when the checkpoint stores an MLX affine packed triple.
    /// Dense load-time quantization ignores this value.
    group_size: usize,
    /// Dense load-time quantization serves the weight by dequantizing it per forward
    /// ([`QuantSpec::with_dequantized_forward`]) instead of through `QMatMul`.
    dequantized_forward: bool,
}

impl QuantSpec {
    /// 4-bit (GGML Q4_K).
    pub const fn q4() -> Self {
        Self {
            dtype: GgmlDType::Q4K,
            group_size: 64,
            dequantized_forward: false,
        }
    }

    /// 8-bit (GGML Q8_0).
    pub const fn q8() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
            group_size: 64,
            dequantized_forward: false,
        }
    }

    /// The same blocks, but a dense weight quantized on load is served **weight-only**: the
    /// resident GGML tensor is dequantized to the activation dtype per forward and multiplied
    /// densely, so the activation is never quantized. The default (`QMatMul`) path quantizes the
    /// activation to per-32-element `q8_1` on CUDA, where a single outlier channel sets a block's
    /// scale and flushes its neighbours — harmless for next-token logits, but a conditioning
    /// encoder whose hidden states ARE the output (Iris-3B's Qwen3-VL tower, sc-26266) needs them
    /// intact; the same reason the media providers' packed tiers dequantize (sc-7702). The
    /// resident bytes are identical; only the per-forward dequant is added.
    pub fn with_dequantized_forward(mut self) -> Self {
        self.dequantized_forward = true;
        self
    }

    /// Whether a dense weight quantized under this spec is served weight-only
    /// ([`Self::with_dequantized_forward`]).
    pub fn dequantizes_forward(&self) -> bool {
        self.dequantized_forward
    }

    /// Map a persisted `quantization.bits` value to a spec: `4 → Q4_K`, `8 → Q8_0`. Any other width
    /// is unrecognized (`None`) — the snapshot writer only ever emits 4 or 8.
    pub fn from_bits(bits: u32) -> Option<Self> {
        Self::from_bits_and_group_size(bits, 64)
    }

    /// Map a persisted affine quantization block to its bit width and source group size.
    pub fn from_bits_and_group_size(bits: u32, group_size: usize) -> Option<Self> {
        if group_size == 0 {
            return None;
        }
        match bits {
            4 => Some(Self {
                dtype: GgmlDType::Q4K,
                group_size,
                dequantized_forward: false,
            }),
            8 => Some(Self {
                dtype: GgmlDType::Q8_0,
                group_size,
                dequantized_forward: false,
            }),
            _ => None,
        }
    }

    /// The bit width to persist in a snapshot's `quantization` block (`Q4_K → 4`, `Q8_0 → 8`).
    pub fn bits(&self) -> u32 {
        match self.dtype {
            GgmlDType::Q8_0 => 8,
            _ => 4,
        }
    }

    /// Group size declared by an MLX affine packed source.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// The GGML dtype a projection with input dimension `in_dim` is quantized to under this spec.
    ///
    /// Q4_K's 256-element super-block does not divide every model's MLP width (YuE stage-2's
    /// `intermediate_size` 5504 = 21.5 × 256, so its `down_proj` has a 5504-wide input). A 4-bit
    /// spec therefore falls back to Q4_0 — 32-element blocks, the same 4.5 bits/weight — for an
    /// `in_dim` that is 32- but not 256-aligned; every other case keeps the spec's own dtype (a
    /// dimension aligned to neither still fails loudly in the quantizer). The snapshot preparer and
    /// quantize-on-load both resolve through here, so a prepared Q4 snapshot re-quantizes on load
    /// to exactly the representation its persisted rounding came from.
    pub fn dtype_for_in_dim(&self, in_dim: usize) -> GgmlDType {
        let q4k_block = GgmlDType::Q4K.block_size();
        let q4_0_block = GgmlDType::Q4_0.block_size();
        if self.dtype == GgmlDType::Q4K
            && !in_dim.is_multiple_of(q4k_block)
            && in_dim.is_multiple_of(q4_0_block)
        {
            GgmlDType::Q4_0
        } else {
            self.dtype
        }
    }
}

/// The load-time storage format for a decoder's large projections — the weight-format selector
/// (`bf16 | Q4 | Q8 | NVFP4`; dense bf16 is `None` at the call sites).
///
/// The NVFP4 arm carries the shared per-device cuBLASLt context, which is why this is not `Copy`:
/// constructing it ([`ProjectionFormat::nvfp4`]) *is* the capability check, so a loader holding one
/// already knows the device can serve every NVFP4 projection it builds.
#[derive(Clone)]
pub enum ProjectionFormat {
    /// GGML block quantization (Q4_K / Q8_0) via Candle's quant.
    Ggml(QuantSpec),
    /// NVFP4, quantized on-device at load and served by the cuBLASLt W4A4 FP4 GEMM.
    Nvfp4(Nvfp4Context),
}

impl ProjectionFormat {
    /// The NVFP4 format for `device`, or the typed [`Nvfp4Refused`](Error::Nvfp4Refused) naming
    /// why the device cannot serve it (not CUDA, below sm_120, no fused quantizer). This is the
    /// whole capability floor, so a loader calls it before reading any weights.
    pub fn nvfp4(device: &Device) -> Result<Self> {
        Ok(Self::Nvfp4(Nvfp4Context::require(device)?))
    }

    /// [`Self::nvfp4`] with the compute-capability probe injected
    /// ([`Nvfp4Context::require_with`]), so the gate can be exercised with a mocked capability.
    #[cfg(feature = "cuda")]
    pub fn nvfp4_with_cap_probe(
        device: &Device,
        cap_probe: impl FnOnce(&candle_quant_kernels::CublasLt) -> candle_core::Result<(i32, i32)>,
    ) -> Result<Self> {
        Ok(Self::Nvfp4(Nvfp4Context::require_with(device, cap_probe)?))
    }

    /// The GGML spec, when this is a GGML format.
    pub fn ggml(&self) -> Option<QuantSpec> {
        match self {
            Self::Ggml(q) => Some(*q),
            Self::Nvfp4(_) => None,
        }
    }

    /// Whether this is NVFP4.
    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4(_))
    }
}

impl From<QuantSpec> for ProjectionFormat {
    fn from(q: QuantSpec) -> Self {
        Self::Ggml(q)
    }
}

impl std::fmt::Debug for ProjectionFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ggml(q) => f.debug_tuple("Ggml").field(q).finish(),
            Self::Nvfp4(_) => f.write_str("Nvfp4"),
        }
    }
}

/// Candle's CUDA `MATRIX_ROW_PADDING` (elements): every GGML tensor candle quantizes onto a CUDA
/// device is allocated this many elements' worth of blocks past its payload.
const GGML_CUDA_ROW_PADDING: u64 = 512;

/// Device bytes of an `elems`-element weight quantized to `dtype` by
/// [`QuantizedLinear::quantize`]: the GGML blocks (what the load census counts) plus the CUDA row
/// padding candle allocates past them. `None` on overflow. Load admission prices a GGML copy
/// with it (sc-24140).
pub fn ggml_device_bytes(elems: u64, dtype: GgmlDType) -> Option<u64> {
    let (block, size) = (dtype.block_size() as u64, dtype.type_size() as u64);
    elems
        .div_ceil(block)
        .checked_mul(size)?
        .checked_add(GGML_CUDA_ROW_PADDING * size / block)
}

/// Device bytes of a `[rows, cols]` weight quantized to NVFP4 by [`Nvfp4Weight::quantize`] — the
/// E2M1 nibbles over the `K`-padded columns plus the UE4M3 block scales laid out in cuBLASLt's
/// 128 × 4 scale-factor atoms — or `None` when the FP4 GEMM cannot serve the shape
/// ([`nvfp4_shape_refusal`], the test [`Projection::load_eligible`] keeps such a weight dense by)
/// or on overflow. Load admission prices an NVFP4 copy with it (sc-24140).
pub fn nvfp4_device_bytes(rows: usize, cols: usize) -> Option<u64> {
    use candle_quant_kernels::nvfp4::{SF_ATOM_COLS, SF_ATOM_ROWS};
    use candle_quant_kernels::{NVFP4_BLOCK, NVFP4_K_ALIGN};
    nvfp4_shape_refusal(rows, cols).ok()?;
    let cols_padded = cols.checked_next_multiple_of(NVFP4_K_ALIGN)?;
    let nibbles = rows.checked_mul(cols_padded / 2)?;
    let scales = rows
        .checked_next_multiple_of(SF_ATOM_ROWS)?
        .checked_mul((cols_padded / NVFP4_BLOCK).checked_next_multiple_of(SF_ATOM_COLS)?)?;
    Some(nibbles.checked_add(scales)? as u64)
}

/// The input width `in` of an MLX affine **8-bit** triple in the layout
/// [`QuantizedLinear::from_mlx_affine_q8`] accepts — a `U32` code matrix `[out, in / 4]` (four
/// 8-bit codes per word) beside `[out, in / group_size]` scales and biases — or `None` for any
/// other layout, which it refuses (a 4-bit pack, eight codes per word, among them). Load admission
/// prices the Q8_0 repack by this same test (sc-24140), so it prices no copy for a triple the
/// loader refuses.
pub fn mlx_affine_q8_in_dim(
    weight_is_u32: bool,
    weight: (usize, usize),
    scales: (usize, usize),
    biases: (usize, usize),
    group_size: usize,
) -> Option<usize> {
    mlx_affine_in_dim(8, weight_is_u32, weight, scales, biases, group_size)
}

/// [`mlx_affine_q8_in_dim`] for a `bits`-wide MLX affine triple (4 or 8: `32 / bits` codes per
/// `U32` word) — the layouts [`QuantizedLinear::from_mlx_affine`] accepts. `None` for any other.
pub fn mlx_affine_in_dim(
    bits: usize,
    weight_is_u32: bool,
    weight: (usize, usize),
    scales: (usize, usize),
    biases: (usize, usize),
    group_size: usize,
) -> Option<usize> {
    let (out_dim, packed_cols) = weight;
    let in_dim = scales.1.checked_mul(group_size)?;
    (matches!(bits, 4 | 8)
        && weight_is_u32
        && group_size != 0
        && scales.0 == out_dim
        && biases == scales
        && packed_cols.checked_mul(32 / bits) == Some(in_dim))
    .then_some(in_dim)
}

/// The GGML block types a prepared snapshot persists (sc-19375). Their block byte sizes are
/// pairwise distinct — Q4_0 18 B / 32 weights, Q8_0 34 B / 32, Q4_K 144 B / 256 — which is what
/// makes a stored block tensor self-describing.
pub const STORED_GGML: [GgmlDType; 3] = [GgmlDType::Q4_0, GgmlDType::Q8_0, GgmlDType::Q4K];

/// The GGML block type and logical `(rows, cols)` of a **stored GGML block tensor** — a prepared
/// Q4 / Q8 snapshot's on-disk projection form (sc-19375): a `U8` tensor `[rows, blocks_per_row,
/// block_bytes]` holding the raw GGML blocks of a `[rows, cols]` weight, row-major (GGML blocks run
/// along the input dimension, so each row is `blocks_per_row` whole blocks). A stacked weight (the
/// qwen3_5 MoE experts, `[experts, rows, cols]`) is stored `[experts, rows, blocks_per_row,
/// block_bytes]`, and its `rows` here count every expert's. The block type is the one of
/// [`STORED_GGML`] whose block size is `block_bytes`. `None` for any other tensor. The loader,
/// load admission, the weight reader and the NVFP4 gate classify a tensor by this one rule, from
/// its dtype and shape alone.
pub fn ggml_block_storage(is_u8: bool, shape: &[usize]) -> Option<(GgmlDType, usize, usize)> {
    let [lead @ .., rows, blocks, block_bytes] = shape else {
        return None;
    };
    if !is_u8 {
        return None;
    }
    let dtype = STORED_GGML
        .into_iter()
        .find(|d| d.type_size() == *block_bytes)?;
    let rows = lead.iter().try_fold(*rows, |n, &d| n.checked_mul(d))?;
    Some((dtype, rows, blocks.checked_mul(dtype.block_size())?))
}

/// Whether `t` is a stored GGML block tensor ([`ggml_block_storage`]).
pub fn is_ggml_block_tensor(t: &Tensor) -> bool {
    ggml_block_storage(t.dtype() == DType::U8, t.dims()).is_some()
}

/// Serialize a 2-D [`QTensor`] of a persisted block type (Q4_0 / Q8_0 / Q4_K) into its stored
/// GGML block tensor on the CPU: a `U8` `[rows, blocks_per_row, block_bytes]` tensor of the raw
/// GGML blocks, row-major, which [`from_ggml_block_tensor`] reads back. GGML quantizes each block
/// on its own, so the blocks of any row slice are byte-for-byte those of quantizing that slice
/// alone — which is what lets a loader carve fused parts and stacked experts out of one stored
/// tensor.
pub fn to_ggml_block_tensor(qt: &QTensor) -> Result<Tensor> {
    let dtype = qt.dtype();
    let (rows, cols) = qt.shape().dims2()?;
    if !STORED_GGML.contains(&dtype) || !cols.is_multiple_of(dtype.block_size()) {
        return Err(Error::Unsupported(format!(
            "ggml block storage: cannot persist a {dtype:?} [{rows}, {cols}] tensor"
        )));
    }
    let bytes = qt.data()?.into_owned();
    let blocks = cols / dtype.block_size();
    Ok(Tensor::from_vec(
        bytes,
        (rows, blocks, dtype.type_size()),
        &Device::Cpu,
    )?)
}

/// Rebuild the [`QTensor`] a rank-3 stored GGML block tensor holds, directly on `device`: the
/// blocks are used exactly as stored — never dequantized and re-quantized. A stacked (rank-4)
/// tensor is sliced to one expert first.
pub fn from_ggml_block_tensor(stored: &Tensor, device: &Device) -> Result<QTensor> {
    let storage = ggml_block_storage(stored.dtype() == DType::U8, stored.dims());
    let (Some((dtype, rows, cols)), 3) = (storage, stored.rank()) else {
        return Err(Error::Config(format!(
            "ggml block storage: {:?} {:?} is not a rank-3 stored GGML block tensor",
            stored.dtype(),
            stored.shape()
        )));
    };
    let bytes = stored
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<u8>()?;
    // candle reinterprets the byte slice as a block slice (2-byte aligned `f16` fields), so hand
    // it u64-backed storage rather than rely on the byte vector's allocation alignment.
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: `words` owns `words.len() * 8 >= bytes.len()` initialized bytes, and u8 has no
    // alignment requirement; the byte view does not outlive `words`.
    let aligned =
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    aligned.copy_from_slice(&bytes);
    Ok(candle_core::quantized::ggml_file::qtensor_from_ggml(
        dtype,
        aligned,
        vec![rows, cols],
        device,
    )?)
}

/// A token-embedding table a fully quantized tier stores as Q8_0 GGML blocks (SW-7201), held on the
/// device as one byte per weight plus one f16 scale per 32-weight block — the stored blocks' own
/// precision and ~their size — so a lookup gathers the requested rows and dequantizes only those.
/// The codes are kept offset by 128 (`u8 = i8 + 128`), which `index_select` and the `u8 → f32`
/// cast every backend has can read directly.
#[derive(Clone)]
pub struct Q8RowEmbedding {
    /// `[rows, cols]` offset-binary codes.
    codes: Tensor,
    /// `[rows, cols / 32]` per-block scales (GGML `d`).
    scales: Tensor,
}

impl Q8RowEmbedding {
    /// Read the stored Q8_0 GGML block tensor `stored` (`[rows, cols / 32, 34]` `U8`, host) of the
    /// table `stem` onto `device`. Any other block type is refused: both fully quantized tiers
    /// store their embedding Q8_0.
    pub fn from_stored(stem: &str, stored: &Tensor, device: &Device) -> Result<Self> {
        let storage = ggml_block_storage(stored.dtype() == DType::U8, stored.dims());
        let (dtype, rows, cols) = match (storage, stored.rank()) {
            (Some(found), 3) => found,
            _ => {
                return Err(Error::Config(format!(
                    "embedding `{stem}` {:?} {:?} is not a rank-3 stored GGML block tensor",
                    stored.dtype(),
                    stored.shape()
                )))
            }
        };
        if dtype != GgmlDType::Q8_0 {
            return Err(Error::Unsupported(format!(
                "embedding `{stem}` is stored {dtype:?}; a quantized embedding table must be Q8_0"
            )));
        }
        // Read the host blocks in place (a weight reader keeps stored blocks on the host): only
        // the device-bound codes and scales are allocated beside them.
        let host = stored.to_device(&Device::Cpu)?.contiguous()?;
        let (storage, layout) = host.storage_and_layout();
        let candle_core::Storage::Cpu(cpu) = &*storage else {
            return Err(Error::Config(format!(
                "embedding `{stem}` blocks are not on the host"
            )));
        };
        let start = layout.start_offset();
        let bytes = &cpu.as_slice::<u8>()?[start..start + host.elem_count()];
        let block_bytes = GgmlDType::Q8_0.type_size();
        let mut codes = Vec::with_capacity(rows * cols);
        let mut scales = Vec::with_capacity(bytes.len() / block_bytes);
        for block in bytes.chunks_exact(block_bytes) {
            scales.push(half::f16::from_le_bytes([block[0], block[1]]));
            codes.extend(block[2..].iter().map(|&b| b ^ 0x80));
        }
        drop(storage);
        let blocks = cols / GgmlDType::Q8_0.block_size();
        Ok(Self {
            codes: Tensor::from_vec(codes, (rows, cols), device)?,
            scales: Tensor::from_vec(scales, (rows, blocks), device)?,
        })
    }

    /// The rows of `ids` (`[batch, seq]`), dequantized to `[batch, seq, cols]` f32: `d · q` per
    /// block, exactly as GGML dequantizes Q8_0.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let (b, s) = ids.dims2()?;
        let (_, cols) = self.codes.dims2()?;
        let blocks = self.scales.dim(1)?;
        let flat = ids.flatten_all()?;
        let n = flat.dim(0)?;
        let codes = self
            .codes
            .index_select(&flat, 0)?
            .to_dtype(DType::F32)?
            .affine(1.0, -128.0)?
            .reshape((n, blocks, cols / blocks))?;
        let scales = self
            .scales
            .index_select(&flat, 0)?
            .to_dtype(DType::F32)?
            .unsqueeze(2)?;
        Ok(codes.broadcast_mul(&scales)?.reshape((b, s, cols))?)
    }

    /// `(rows, cols)` of the logical table.
    pub fn dims(&self) -> Result<(usize, usize)> {
        Ok(self.codes.dims2()?)
    }

    /// Resident bytes: the codes plus the scales.
    pub fn resident_bytes(&self) -> u64 {
        (self.codes.elem_count() + self.scales.elem_count() * self.scales.dtype().size_in_bytes())
            as u64
    }

    /// The device bytes [`Self::from_stored`] allocates for a `[rows, cols]` Q8_0 table, from its
    /// shape alone (load admission). `None` on overflow.
    pub fn device_bytes(rows: usize, cols: usize) -> Option<u64> {
        let elems = (rows as u64).checked_mul(cols as u64)?;
        let blocks = elems / GgmlDType::Q8_0.block_size() as u64;
        elems.checked_add(blocks.checked_mul(2)?)
    }
}

/// A linear projection whose weight is stored GGML block-quantized.
pub struct QuantizedLinear {
    #[doc(hidden)]
    pub inner: QuantizedWeight,
    /// Optional additive bias applied after the matmul.
    pub bias: Option<Tensor>,
}

#[doc(hidden)]
pub enum QuantizedWeight {
    /// Existing dense load-time quantization path.
    Matmul(QMatMul),
    /// MLX affine packed tiers keep activations full precision and dequantize the resident weight
    /// per forward, matching the shared Candle packed-tier policy.
    Dequant(std::sync::Arc<QTensor>),
}

impl QuantizedLinear {
    /// Wrap a tensor which was already stored in a GGUF quantized representation. This preserves
    /// the compact resident payload and lets [`QMatMul`] dispatch the matching CPU/CUDA kernel;
    /// loading a Q8 projector must not materialize a second dense copy of the matrix.
    pub fn from_qtensor(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            inner: QuantizedWeight::Matmul(QMatMul::from_qtensor(weight)?),
            bias,
        })
    }

    /// Quantize a dense `[out, in]` weight (the input dim must be a multiple of `dtype`'s block
    /// size). `bias`, if present, is added after the matmul.
    ///
    /// `weight` may be a view — an expert sliced out of a stacked qwen3_5 MoE tensor, or one part
    /// of a fused Phi-3 `qkv_proj` / `gate_up_proj`. Candle's quantizer reads its source's storage
    /// from the start, ignoring the view's offset and extent, so the f32 source it is handed is
    /// always compacted first: a cast to f32 builds a fresh tensor, but an f32 weight (every
    /// weight on a host device, whose compute dtype is f32) would otherwise be passed through as
    /// the view itself and quantize the wrong rows — or trip candle's size check (sc-24140).
    pub fn quantize(weight: &Tensor, dtype: GgmlDType, bias: Option<Tensor>) -> Result<Self> {
        let source = match weight.dtype() {
            DType::F32 => weight.force_contiguous()?,
            _ => weight.to_dtype(DType::F32)?,
        };
        let qt = QTensor::quantize(&source, dtype)?;
        Ok(Self {
            inner: QuantizedWeight::Matmul(QMatMul::from_qtensor(qt)?),
            bias,
        })
    }

    /// [`Self::quantize`], but served weight-only: the resident GGML tensor is dequantized to the
    /// activation dtype per forward and multiplied densely, so the activation is never quantized
    /// ([`QuantSpec::with_dequantized_forward`]).
    /// The blocks are those [`Self::quantize`] writes, quantized on the CPU and placed on
    /// `weight`'s device.
    pub fn quantize_dequantized(
        weight: &Tensor,
        dtype: GgmlDType,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let source = weight.to_device(&Device::Cpu)?;
        let source = match source.dtype() {
            DType::F32 => source.force_contiguous()?,
            _ => source.to_dtype(DType::F32)?,
        };
        let qt = QTensor::quantize_onto(&source, dtype, weight.device())?;
        Ok(Self {
            inner: QuantizedWeight::Dequant(std::sync::Arc::new(qt)),
            bias,
        })
    }

    /// Convert an MLX affine Q8 triple (`U32` byte-packed codes plus per-group scale/bias) into the
    /// resident Q8_0 tensor used by this primitive. This matches the established Candle packed-tier
    /// policy: reconstruct the source affine grid exactly, then perform the accepted Q8_0 re-pack.
    pub fn from_mlx_affine_q8(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::from_mlx_affine(weight, scales, biases, bias, group_size, 8, device)
    }

    /// [`from_mlx_affine_q8`](Self::from_mlx_affine_q8) for a `bits`-wide triple (4 or 8; `32 /
    /// bits` little-endian codes per `U32` word, MLX's `quantize` layout): the exact affine grid
    /// `scale · code + bias`, re-packed to the resident Q8_0 form.
    pub fn from_mlx_affine(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        group_size: usize,
        bits: usize,
        device: &Device,
    ) -> Result<Self> {
        let (out_dim, packed_cols) = weight.dims2()?;
        let (_, scale_cols) = scales.dims2()?;
        let Some(in_dim) = mlx_affine_in_dim(
            bits,
            weight.dtype() == DType::U32,
            (out_dim, packed_cols),
            scales.dims2()?,
            biases.dims2()?,
            group_size,
        ) else {
            return Err(Error::Config(format!(
                "invalid MLX affine Q{bits} triple: weight {:?} {:?}, scales {:?}, biases {:?}, group {group_size}",
                weight.dtype(),
                weight.shape(),
                scales.shape(),
                biases.shape()
            )));
        };

        let cpu = Device::Cpu;
        let words = weight.to_device(&cpu)?.flatten_all()?.to_vec1::<u32>()?;
        let scales = scales
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let biases = biases
            .to_device(&cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let per_word = 32 / bits;
        let mask = (1u32 << bits) - 1;
        let mut grid = vec![0f32; out_dim * in_dim];
        for row in 0..out_dim {
            let word_row = row * packed_cols;
            let group_row = row * scale_cols;
            let value_row = row * in_dim;
            for col in 0..in_dim {
                let word = words[word_row + col / per_word];
                let code = ((word >> (bits * (col % per_word))) & mask) as f32;
                let group = group_row + col / group_size;
                grid[value_row + col] = scales[group] * code + biases[group];
            }
        }
        let dense = Tensor::from_vec(grid, (out_dim, in_dim), &cpu)?;
        let qt = QTensor::quantize_onto(&dense, GgmlDType::Q8_0, device)?;
        Ok(Self {
            inner: QuantizedWeight::Dequant(std::sync::Arc::new(qt)),
            bias,
        })
    }

    /// Resident bytes of the weight as stored (the GGML block payload, or the dense tensor
    /// `QMatMul` expanded a float-typed GGUF matrix into) plus the bias — the load telemetry's
    /// per-projection footprint (sc-24135).
    pub fn resident_bytes(&self) -> usize {
        let dense = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        let weight = match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                q.storage_size_in_bytes()
            }
            QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => dense(t),
        };
        weight + self.bias.as_ref().map_or(0, dense)
    }

    /// The logical `(out, in)` weight shape.
    pub fn dims(&self) -> (usize, usize) {
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                q.shape().dims2().unwrap_or((0, 0))
            }
            QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => {
                t.dims2().unwrap_or((0, 0))
            }
        }
    }

    /// Logical weight elements (`out · in`).
    pub fn weight_elems(&self) -> usize {
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                q.shape().elem_count()
            }
            QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)) => t.elem_count(),
        }
    }

    /// The resident GGML block tensor and how this projection's forward uses it, for an MoE bank's
    /// indexed expert table (sc-24440): `(weight, dequant)` — `dequant` for the MLX-affine Q8
    /// tier, whose forward dequantizes the weight to the activation dtype; otherwise the forward
    /// is `QMatMul`'s. `None` for a float-typed GGUF matrix (no blocks) or a projection with a bias
    /// (the indexed kernels add none).
    pub fn indexed_source(&self) -> Option<(std::sync::Arc<QTensor>, bool)> {
        if self.bias.is_some() {
            return None;
        }
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) => Some((q.clone(), false)),
            QuantizedWeight::Dequant(q) => Some((q.clone(), true)),
            QuantizedWeight::Matmul(_) => None,
        }
    }

    /// The dense weight `QMatMul` expanded a float-typed GGUF matrix into, when it holds one and no
    /// bias — its forward is then a plain dense matmul over these rows.
    pub fn dense_weight(&self) -> Option<&Tensor> {
        match (&self.inner, &self.bias) {
            (QuantizedWeight::Matmul(QMatMul::Tensor(t) | QMatMul::TensorF16(t)), None) => Some(t),
            _ => None,
        }
    }

    /// The GGML block dtype the weight is stored in, when it is block-quantized.
    pub fn ggml_dtype(&self) -> Option<GgmlDType> {
        match &self.inner {
            QuantizedWeight::Matmul(QMatMul::QTensor(q)) | QuantizedWeight::Dequant(q) => {
                Some(q.dtype())
            }
            QuantizedWeight::Matmul(_) => None,
        }
    }

    /// Whether the forward dequantizes the weight and multiplies the activation densely (built by
    /// [`Self::quantize_dequantized`] or an MLX affine source) instead of through `QMatMul`, which
    /// quantizes the activation.
    pub fn dequantizes_forward(&self) -> bool {
        matches!(self.inner, QuantizedWeight::Dequant(_))
    }

    /// Forward pass: `x @ dequant(weight)ᵀ (+ bias)`. The quantized matmul runs in f32; the result is
    /// cast back to `x`'s dtype so it composes with a bf16 decoder.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = match &self.inner {
            QuantizedWeight::Matmul(inner) if native_mmvq(inner, x) => {
                // CUDA MMVQ quantizes a bf16/f16 activation and writes a bf16/f16 result with the
                // same arithmetic as the f32 round trip below (exact widening in, one RN
                // narrowing out), without the two cast kernels around every decode matvec.
                __model_proof_output!(inner.forward(x))?
            }
            QuantizedWeight::Matmul(inner) => {
                __model_proof_output!(inner.forward(&x.to_dtype(DType::F32)?))?
                    .to_dtype(x.dtype())?
            }
            QuantizedWeight::Dequant(weight) => {
                let dense = weight.dequantize(x.device())?.to_dtype(x.dtype())?;
                __model_proof_output!(Linear::new(dense, None).forward(x))?
            }
        };
        match &self.bias {
            Some(b) => Ok(__model_proof_output!(
                y.broadcast_add(&b.to_dtype(y.dtype())?)
            )?),
            None => Ok(y),
        }
    }
}

/// Whether `x` can go to Candle's CUDA MMVQ kernels in its own dtype: a contiguous bf16/f16
/// activation of at most 8 rows (the decode and speculative-verify shapes) against a GGML weight
/// type those kernels serve. Larger batches keep the f32 path (the MMQ/dequantize fallbacks).
fn native_mmvq(inner: &QMatMul, x: &Tensor) -> bool {
    use candle_core::quantized::GgmlDType as G;
    let QMatMul::QTensor(w) = inner else {
        return false;
    };
    let rows = match x.dims() {
        [m, _] => *m,
        [b, m, _] => b * m,
        _ => return false,
    };
    x.device().is_cuda()
        && matches!(x.dtype(), DType::BF16 | DType::F16)
        && x.is_contiguous()
        && (1..=8).contains(&rows)
        && matches!(
            w.dtype(),
            G::Q4_0
                | G::Q4_1
                | G::Q5_0
                | G::Q5_1
                | G::Q8_0
                | G::Q2K
                | G::Q3K
                | G::Q4K
                | G::Q5K
                | G::Q6K
        )
}

/// A linear projection weight, dense or quantized.
pub enum Projection {
    /// A dense `[out, in]` weight wrapped in a Candle linear.
    Dense(Linear),
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
    /// A compact Prism/Bonsai affine-2 or native ternary weight.
    Prism(std::sync::Arc<PrismPackedWeight>),
    /// An NVFP4 weight quantized at load (sc-24135), resident as packed E2M1 + UE4M3 scales.
    /// Behind an `Arc` (its device handles would otherwise grow every projection-holding enum in
    /// the decoders) that an MoE bank's indexed expert table shares (sc-24440).
    Nvfp4(std::sync::Arc<Nvfp4Weight>),
}

/// Which representation a loaded [`Projection`] actually holds — the load telemetry's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectionKind {
    /// Dense float weight.
    Dense,
    /// GGML block-quantized (Q4_K / Q8_0 / GGUF block types).
    Ggml,
    /// Prism/Bonsai packed.
    Prism,
    /// NVFP4.
    Nvfp4,
}

impl ProjectionKind {
    /// Stable lower-case label for logs and evidence rows.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Ggml => "ggml",
            Self::Prism => "prism",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// Count, logical parameters and resident bytes of the projections of one [`ProjectionKind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectionTally {
    /// Projections of this kind.
    pub count: u64,
    /// Logical weight elements (`Σ out · in`).
    pub params: u64,
    /// Resident weight (+ bias) bytes of the measured projections.
    pub resident_bytes: u64,
    /// Projections whose resident bytes could not be measured (Prism); excluded from
    /// `resident_bytes` rather than counted as zero.
    pub unmeasured: u64,
}

impl ProjectionTally {
    /// Resident bits per logical parameter. `None` when the tally is empty or any of its
    /// projections is unmeasured (a partial sum would understate the footprint).
    pub fn bits_per_param(&self) -> Option<f64> {
        (self.params > 0 && self.unmeasured == 0)
            .then(|| self.resident_bytes as f64 * 8.0 / self.params as f64)
    }

    fn add(&mut self, other: &Self) {
        self.count += other.count;
        self.params += other.params;
        self.resident_bytes += other.resident_bytes;
        self.unmeasured += other.unmeasured;
    }
}

/// Which projection kinds a loaded model holds, with per-kind parameter and residency totals — the
/// load telemetry that makes the served representation visible (epic sc-24128 E2): an NVFP4 request
/// reports every requested projection under `nvfp4`, never a dense projection under an NVFP4 label.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectionCensus {
    /// Dense float projections.
    pub dense: ProjectionTally,
    /// GGML block-quantized projections.
    pub ggml: ProjectionTally,
    /// Prism/Bonsai packed projections.
    pub prism: ProjectionTally,
    /// NVFP4 projections.
    pub nvfp4: ProjectionTally,
    /// How many of the `ggml` projections are served weight-only (sc-26266): the forward
    /// dequantizes the weight and multiplies the activation densely, never quantizing it
    /// ([`QuantSpec::with_dequantized_forward`]). The rest of `ggml` runs through `QMatMul`.
    pub ggml_weight_only: u64,
    /// How many of the `ggml` projections hold 4-bit blocks (`Q4_0`, `Q4_1`, `Q4_K`) and 8-bit
    /// blocks (`Q8_0`, `Q8_1`, `Q8_K`) — the resident bit width, whether quantized at load or
    /// stored in a prepared snapshot (SW-7200). The rest of `ggml` holds another block type.
    pub ggml_4bit: u64,
    /// See [`ggml_4bit`](Self::ggml_4bit).
    pub ggml_8bit: u64,
}

impl ProjectionCensus {
    /// Add one projection.
    pub fn record(&mut self, p: &Projection) {
        let tally = match p.kind() {
            ProjectionKind::Dense => &mut self.dense,
            ProjectionKind::Ggml => &mut self.ggml,
            ProjectionKind::Prism => &mut self.prism,
            ProjectionKind::Nvfp4 => &mut self.nvfp4,
        };
        if let Projection::Quantized(q) = p {
            self.ggml_weight_only += u64::from(q.dequantizes_forward());
            match q.ggml_dtype().and_then(ggml_bits) {
                Some(4) => self.ggml_4bit += 1,
                Some(8) => self.ggml_8bit += 1,
                _ => {}
            }
        }
        tally.count += 1;
        tally.params += p.params();
        match p.resident_bytes() {
            Some(bytes) => tally.resident_bytes += bytes,
            None => tally.unmeasured += 1,
        }
    }

    /// The tally for one kind.
    pub fn tally(&self, kind: ProjectionKind) -> &ProjectionTally {
        match kind {
            ProjectionKind::Dense => &self.dense,
            ProjectionKind::Ggml => &self.ggml,
            ProjectionKind::Prism => &self.prism,
            ProjectionKind::Nvfp4 => &self.nvfp4,
        }
    }

    /// All projections together.
    pub fn total(&self) -> ProjectionTally {
        let mut total = ProjectionTally::default();
        for t in [&self.dense, &self.ggml, &self.prism, &self.nvfp4] {
            total.add(t);
        }
        total
    }

    /// Merge another census into this one.
    pub fn merge(&mut self, other: &Self) {
        self.dense.add(&other.dense);
        self.ggml.add(&other.ggml);
        self.prism.add(&other.prism);
        self.nvfp4.add(&other.nvfp4);
        self.ggml_weight_only += other.ggml_weight_only;
        self.ggml_4bit += other.ggml_4bit;
        self.ggml_8bit += other.ggml_8bit;
    }

    /// What the quantized projections hold (SW-7200): dense when there are none, the GGML bit
    /// width when every one is a 4-bit or every one an 8-bit block, NVFP4 when every one is
    /// NVFP4; unknown for mixed widths or kinds, another block type, or Prism. Dense projections
    /// beside quantized ones (an LM head) do not make it mixed. A llama.cpp mixed-precision GGUF
    /// (`Q4_K_M`: `Q4_K` beside `Q6_K`) is deliberately unknown, not `q4`: every recorded
    /// decision was measured on a uniform-width prepared snapshot, so none applies to it.
    pub fn resident_quantization(&self) -> core_llm::ResidentQuantization {
        use core_llm::ResidentQuantization as R;
        let (ggml, nvfp4, prism) = (self.ggml.count, self.nvfp4.count, self.prism.count);
        match (ggml, nvfp4, prism) {
            (0, 0, 0) => R::Dense,
            (0, _, 0) => R::Nvfp4,
            (n, 0, 0) if self.ggml_4bit == n => R::Bits(4),
            (n, 0, 0) if self.ggml_8bit == n => R::Bits(8),
            _ => R::Unknown,
        }
    }
}

/// The bit width of a 4- or 8-bit GGML block type; `None` for any other.
pub fn ggml_bits(dtype: GgmlDType) -> Option<u32> {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q4K => Some(4),
        GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Q8K => Some(8),
        _ => None,
    }
}

/// A loaded model's whole resident weight set: its [`ProjectionCensus`] plus every other weight
/// tensor (embeddings, norms, recurrent parameters) under `other`. The load telemetry the
/// NVFP4 evidence reads its bits/param from (sc-24135).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeightCensus {
    /// The projections, by kind.
    pub projections: ProjectionCensus,
    /// Every non-projection weight tensor (`count` is tensors, not projections).
    pub other: ProjectionTally,
}

impl WeightCensus {
    /// Add a non-projection dense weight tensor.
    pub fn record_tensor(&mut self, t: &Tensor) {
        self.other.count += 1;
        self.other.params += t.elem_count() as u64;
        self.other.resident_bytes += (t.elem_count() * t.dtype().size_in_bytes()) as u64;
    }

    /// Add a non-projection weight of `params` logical elements stored in `bytes` (a quantized
    /// embedding table).
    pub fn record_packed(&mut self, params: u64, bytes: u64) {
        self.other.count += 1;
        self.other.params += params;
        self.other.resident_bytes += bytes;
    }

    /// Add a non-projection weight whose storage is not measured here (a Prism embedding).
    pub fn record_unmeasured(&mut self, params: u64) {
        self.other.count += 1;
        self.other.params += params;
        self.other.unmeasured += 1;
    }

    /// Everything together.
    pub fn total(&self) -> ProjectionTally {
        let mut total = self.projections.total();
        total.add(&self.other);
        total
    }

    /// Merge another census into this one.
    pub fn merge(&mut self, other: &Self) {
        self.projections.merge(&other.projections);
        self.other.add(&other.other);
    }
}

impl Projection {
    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Tensor, quant: Option<QuantSpec>) -> Result<Self> {
        Self::load_with_bias(weight, None, quant)
    }

    /// Load from a dense `[out, in]` weight plus an optional `[out]` bias (Qwen2 attention carries
    /// q/k/v bias), quantizing the weight if `quant` is set. The bias is always applied dense.
    pub fn load_with_bias(
        weight: Tensor,
        bias: Option<Tensor>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense(Linear::new(weight, bias))),
            Some(q) => {
                let in_dim = weight.dims().last().copied().unwrap_or(0);
                let dtype = q.dtype_for_in_dim(in_dim);
                Ok(Projection::Quantized(if q.dequantizes_forward() {
                    QuantizedLinear::quantize_dequantized(&weight, dtype, bias)?
                } else {
                    QuantizedLinear::quantize(&weight, dtype, bias)?
                }))
            }
        }
    }

    /// Load from a dense `[out, in]` weight plus optional bias in the requested `format` (`None`
    /// keeps it dense). The one entry point a loader needs for every format, NVFP4 included.
    ///
    /// NVFP4 quantizes on the weight's device (it must be `format`'s device) and refuses an
    /// ineligible shape with the typed [`Nvfp4Refused`](Error::Nvfp4Refused) — it never keeps a
    /// projection dense under an NVFP4 request.
    pub fn load_as(
        weight: Tensor,
        bias: Option<Tensor>,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        match format {
            None => Self::load_with_bias(weight, bias, None),
            Some(ProjectionFormat::Ggml(q)) => Self::load_with_bias(weight, bias, Some(*q)),
            Some(ProjectionFormat::Nvfp4(ctx)) => {
                let (rows, cols) = weight.dims2()?;
                nvfp4_shape_refusal(rows, cols)?;
                Ok(Self::Nvfp4(std::sync::Arc::new(Nvfp4Weight::quantize(
                    &weight, bias, ctx,
                )?)))
            }
        }
    }

    /// [`Self::load_as`] with the llama-family shape policy (sc-24140): under an NVFP4 `format`, a
    /// weight whose shape the FP4 GEMM cannot serve ([`nvfp4_shape_refusal`] — `N % 16 != 0`, or
    /// too large for the fused quantizer) stays **dense** instead of refusing the load, so a
    /// checkpoint with one odd projection (a vocabulary that is not a multiple of 16, say) still
    /// loads with every eligible projection NVFP4. The kept projection reports
    /// [`ProjectionKind::Dense`], so the load census shows it under `dense` — never under an NVFP4
    /// label. Every other format (and every eligible shape) behaves exactly as `load_as`.
    pub fn load_eligible(
        weight: Tensor,
        bias: Option<Tensor>,
        format: Option<&ProjectionFormat>,
    ) -> Result<Self> {
        if let Some(ProjectionFormat::Nvfp4(_)) = format {
            let (rows, cols) = weight.dims2()?;
            if nvfp4_shape_refusal(rows, cols).is_err() {
                return Self::load_with_bias(weight, bias, None);
            }
        }
        Self::load_as(weight, bias, format)
    }

    /// Wrap a resident compact Prism weight. Prism projections do not carry an additive bias.
    pub fn load_prism(weight: std::sync::Arc<PrismPackedWeight>) -> Self {
        Self::Prism(weight)
    }

    /// Wrap a pre-quantized GGUF matrix without expanding it. Plain F16/BF16/F32 GGUF matrices are
    /// dequantized by the caller and use [`Self::load_with_bias`]; block-quantized matrices stay in
    /// this representation for their full resident lifetime.
    pub fn load_qtensor(weight: QTensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self::Quantized(QuantizedLinear::from_qtensor(
            weight, bias,
        )?))
    }

    /// Load the stored GGML block tensor `source` ([`is_ggml_block_tensor`]) as the projection its
    /// blocks hold (a prepared Q4 / Q8 snapshot, sc-19375), rebuilt on `device` exactly as stored.
    /// `stem` names the projection in a refusal.
    ///
    /// The blocks already are the lossy code, so `format` must be the GGML format of their bit
    /// width: any other request would re-quantize a quantized weight, and is refused rather than
    /// silently served at the stored precision. Every loader that reads a projection — the
    /// separate, fused-part and stacked-expert sites of both decoders — comes through here.
    pub fn load_stored_blocks(
        stem: &str,
        source: &Tensor,
        bias: Option<Tensor>,
        format: Option<&ProjectionFormat>,
        device: &Device,
    ) -> Result<Self> {
        let Some((stored, _, _)) =
            ggml_block_storage(source.dtype() == candle_core::DType::U8, source.dims())
        else {
            return Err(Error::Config(format!(
                "projection `{stem}` is not a stored GGML block tensor"
            )));
        };
        let bits = if stored == GgmlDType::Q8_0 { 8 } else { 4 };
        match format {
            Some(ProjectionFormat::Ggml(quant)) if quant.bits() == bits => {}
            Some(ProjectionFormat::Ggml(quant)) => {
                return Err(Error::Unsupported(format!(
                    "projection `{stem}` is stored {stored:?} (a prepared Q{bits} snapshot); it \
                     cannot be re-quantized to Q{}",
                    quant.bits()
                )))
            }
            Some(ProjectionFormat::Nvfp4(_)) => {
                return Err(Error::Unsupported(format!(
                    "nvfp4: projection `{stem}` is stored {stored:?} (a prepared Q{bits} \
                     snapshot); NVFP4 projections are quantized from a dense snapshot"
                )))
            }
            None => {
                return Err(Error::Config(format!(
                    "projection `{stem}` is stored {stored:?} but the model config has no \
                     quantization block"
                )))
            }
        }
        let qt = from_ggml_block_tensor(source, device)?;
        Self::load_qtensor(qt, bias)
    }

    /// Load a pre-quantized MLX affine Q8 triple without interpreting its shortened U32 code
    /// matrix as a dense projection. The source is converted once to the resident Q8_0 form used by
    /// the existing quantized forward.
    pub fn load_mlx_affine_q8(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bias: Option<Tensor>,
        quant: QuantSpec,
        device: &candle_core::Device,
    ) -> Result<Self> {
        if quant.bits() != 8 {
            return Err(Error::Config(format!(
                "MLX affine projection requires Q8, got Q{}",
                quant.bits()
            )));
        }
        Ok(Projection::Quantized(QuantizedLinear::from_mlx_affine_q8(
            weight,
            scales,
            biases,
            bias,
            quant.group_size(),
            device,
        )?))
    }

    /// Load a `bits`-wide (4 or 8) MLX affine triple — a companion MTP head's stored projections
    /// (sc-24444) — as its exact affine grid re-packed to the resident Q8_0 form
    /// ([`QuantizedLinear::from_mlx_affine`]).
    pub fn load_mlx_affine(
        weight: &Tensor,
        scales: &Tensor,
        biases: &Tensor,
        bits: usize,
        group_size: usize,
        device: &candle_core::Device,
    ) -> Result<Self> {
        Ok(Projection::Quantized(QuantizedLinear::from_mlx_affine(
            weight, scales, biases, None, group_size, bits, device,
        )?))
    }

    /// Construct a [`Projection`] from a [`QLinear`] on the shared `candle-layers` spine
    /// (`BACK-1`, `BACK-4`).
    pub fn from_qlinear(q: &QLinear) -> Result<Self> {
        match q {
            QLinear::Dense(DenseLinear::Linear(l)) => Ok(Projection::Dense(l.clone())),
            QLinear::Dense(DenseLinear::Transposed { weight_t, bias }) => Ok(Projection::Dense(
                Linear::new(weight_t.t()?.contiguous()?, bias.clone()),
            )),
            QLinear::Quantized {
                weight: QuantWeight::Matmul(m),
                bias,
                ..
            } => Ok(Projection::Quantized(QuantizedLinear {
                inner: QuantizedWeight::Matmul(m.clone()),
                bias: bias.clone(),
            })),
            QLinear::Quantized {
                weight: QuantWeight::Dequant(w),
                bias,
                ..
            } => Ok(Projection::Quantized(QuantizedLinear {
                inner: QuantizedWeight::Dequant(w.clone()),
                bias: bias.clone(),
            })),
            QLinear::W4A8(_) => Err(Error::Unsupported(
                "W4A8 QLinear has no GGML/dense LLM Projection".into(),
            )),
        }
    }

    /// Convert a dense or GGML-quantized [`Projection`] into a [`QLinear`] on the shared
    /// `candle-layers` spine (`BACK-1`, `BACK-4`).
    pub fn to_qlinear(&self) -> Result<QLinear> {
        match self {
            Projection::Dense(l) => Ok(QLinear::Dense(DenseLinear::Linear(l.clone()))),
            Projection::Quantized(q) => {
                let weight = match &q.inner {
                    QuantizedWeight::Matmul(m) => QuantWeight::Matmul(m.clone()),
                    QuantizedWeight::Dequant(w) => QuantWeight::Dequant(w.clone()),
                };
                Ok(QLinear::Quantized {
                    weight,
                    bias: q.bias.clone(),
                    fwd: QuantForward {
                        flatten_leading: false,
                        cast_back: true,
                    },
                })
            }
            Projection::Prism(_) | Projection::Nvfp4(_) => Err(Error::Unsupported(
                "Prism/NVFP4 Projection cannot convert to a GGUF/dense QLinear".into(),
            )),
        }
    }

    /// Construct a [`Projection`] from an [`AdaptLinear`] (`crate::Linear`) on the shared
    /// `candle-layers` spine (`BACK-1`, `BACK-4`).
    pub fn from_adapt_linear(linear: &AdaptLinear) -> Result<Self> {
        match &linear.base {
            Base::Dense(l) => Ok(Projection::Dense(l.clone())),
            Base::Packed(q) => Self::from_qlinear(q),
            _ => Err(Error::Unsupported(
                "AdaptLinear base has no direct GGML/dense LLM Projection".into(),
            )),
        }
    }

    /// Convert a dense or GGML-quantized [`Projection`] into an [`AdaptLinear`] (`crate::Linear`)
    /// on the shared `candle-layers` spine (`BACK-1`, `BACK-4`).
    pub fn to_adapt_linear(&self) -> Result<AdaptLinear> {
        let (out_dim, in_dim) = self.dims();
        match self {
            Projection::Dense(l) => Ok(AdaptLinear::from_dense(l.clone(), in_dim, out_dim)),
            Projection::Quantized(_) => Ok(AdaptLinear::from_packed(
                self.to_qlinear()?,
                in_dim,
                out_dim,
            )),
            Projection::Prism(_) | Projection::Nvfp4(_) => Err(Error::Unsupported(
                "Prism/NVFP4 Projection cannot convert to a GGUF/dense AdaptLinear".into(),
            )),
        }
    }

    /// `x @ weightᵀ`.
    ///
    /// An NVFP4 projection dispatches between the fused decode GEMV (≤ 8 bf16 rows) and the
    /// cuBLASLt W4A4 GEMM, recording which ran ([`nvfp4_path`](super::nvfp4_path), sc-24136).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Projection::Dense(l) => Ok(__model_proof_output!(l.forward(x))?),
            Projection::Quantized(q) => __model_proof_output!(q.forward(x)),
            Projection::Prism(weight) => __model_proof_output!(weight.forward(x)),
            Projection::Nvfp4(weight) => super::nvfp4_path::forward(weight, x),
        }
    }

    /// Evaluate `x @ weightᵀ (+ bias)` under the shared `candle-layers` spine's request-scoped
    /// [`ScaleContext`] signature (`BACK-1`, `BACK-4`), preserving `Projection`'s exact operation
    /// order (`AP13 class (a)`).
    pub fn forward_with_scales(
        &self,
        x: &Tensor,
        _scales: Option<&ScaleContext<'_>>,
    ) -> Result<Tensor> {
        self.forward(x)
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        !matches!(self, Projection::Dense(_))
    }

    /// Which representation this projection holds.
    pub fn kind(&self) -> ProjectionKind {
        match self {
            Projection::Dense(_) => ProjectionKind::Dense,
            Projection::Quantized(_) => ProjectionKind::Ggml,
            Projection::Prism(_) => ProjectionKind::Prism,
            Projection::Nvfp4(_) => ProjectionKind::Nvfp4,
        }
    }

    /// Logical weight elements (`out · in`).
    pub fn params(&self) -> u64 {
        (match self {
            Projection::Dense(l) => l.weight().elem_count(),
            Projection::Quantized(q) => q.weight_elems(),
            Projection::Prism(w) => w.rows() * w.input_width(),
            Projection::Nvfp4(w) => {
                let (rows, cols) = w.shape();
                rows * cols
            }
        }) as u64
    }

    /// The logical `(out, in)` weight shape.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Projection::Dense(l) => l.weight().dims2().unwrap_or((0, 0)),
            Projection::Quantized(q) => q.dims(),
            Projection::Prism(w) => (w.rows(), w.input_width()),
            Projection::Nvfp4(w) => w.shape(),
        }
    }

    /// Resident weight (+ bias) bytes, or `None` for a representation whose storage is not
    /// measured here (Prism).
    pub fn resident_bytes(&self) -> Option<u64> {
        let dense = |t: &Tensor| (t.elem_count() * t.dtype().size_in_bytes()) as u64;
        match self {
            Projection::Dense(l) => Some(dense(l.weight()) + l.bias().map_or(0, dense)),
            Projection::Quantized(q) => Some(q.resident_bytes() as u64),
            Projection::Prism(_) => None,
            Projection::Nvfp4(w) => Some(w.resident_bytes() as u64),
        }
    }
}

/// A layer's key **and** value projections, which may be one shared weight.
///
/// Gemma 4's `attention_k_eq_v` makes the `full_attention` layers reuse the key projection's output
/// as the value projection's — there is no `v_proj` weight in the checkpoint at all. That is a
/// projection-layer fact, not a decoder one: the value path still gets its own (scale-free) per-head
/// norm afterwards, so K and V remain different tensors; only the matmul and the weight are shared.
///
/// Holding it here keeps the saving real. A decoder that "supported" `k_eq_v` by running the same
/// weight through two projections would produce identical numbers while paying twice the matmul and
/// twice the (quantized) weight footprint — the whole point of the flag.
pub struct KvProjection {
    k: Projection,
    /// `None` means `attention_k_eq_v`: the value heads come from `k`'s output.
    v: Option<Projection>,
}

impl KvProjection {
    /// Independent key and value projections (every architecture before Gemma 4, and Gemma 4's
    /// `sliding_attention` layers).
    pub fn separate(k: Projection, v: Projection) -> Self {
        Self { k, v: Some(v) }
    }

    /// One shared projection feeding both key and value heads (`attention_k_eq_v: true`).
    pub fn shared(k: Projection) -> Self {
        Self { k, v: None }
    }

    /// Whether K and V share a projection.
    pub fn k_eq_v(&self) -> bool {
        self.v.is_none()
    }

    /// The key projection.
    pub fn key(&self) -> &Projection {
        &self.k
    }

    /// The value projection, or `None` when it is shared with the key's.
    pub fn value(&self) -> Option<&Projection> {
        self.v.as_ref()
    }

    /// Project `x` into the **raw** key and value tensors, before any per-head norm or RoPE.
    ///
    /// When shared, the key projection runs **once** and both returned handles reference that one
    /// result (Candle tensors are refcounted, so this is a handle clone, not a copy).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.k.forward(x)?;
        match &self.v {
            Some(v) => {
                let v = v.forward(x)?;
                Ok((k, v))
            }
            None => Ok((k.clone(), k)),
        }
    }

    /// Whether either half is quantized.
    pub fn is_quantized(&self) -> bool {
        self.k.is_quantized() || self.v.as_ref().is_some_and(Projection::is_quantized)
    }

    /// Add both halves (one when shared) to a load census.
    pub fn record(&self, census: &mut ProjectionCensus) {
        census.record(&self.k);
        if let Some(v) = &self.v {
            census.record(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_spec_and_in_dim_dtype_selection_contract() {
        let q4 = QuantSpec::q4();
        assert_eq!(q4.bits(), 4);
        assert_eq!(q4.group_size(), 64);
        assert!(!q4.dequantizes_forward());
        assert!(q4.with_dequantized_forward().dequantizes_forward());
        assert_eq!(q4.dtype_for_in_dim(4096), GgmlDType::Q4K);
        assert_eq!(q4.dtype_for_in_dim(5504), GgmlDType::Q4_0);
        assert_eq!(q4.dtype_for_in_dim(100), GgmlDType::Q4K);

        let q8 = QuantSpec::q8();
        assert_eq!(q8.bits(), 8);
        assert_eq!(q8.dtype_for_in_dim(5504), GgmlDType::Q8_0);
        assert_eq!(QuantSpec::from_bits(4), Some(q4));
        assert_eq!(QuantSpec::from_bits(8), Some(q8));
        assert_eq!(QuantSpec::from_bits(6), None);
        assert_eq!(QuantSpec::from_bits_and_group_size(8, 0), None);
    }

    #[test]
    fn projection_tally_and_census_metadata_contract() {
        use core_llm::ResidentQuantization as R;
        assert_eq!(ProjectionKind::Dense.label(), "dense");
        assert_eq!(ProjectionKind::Ggml.label(), "ggml");
        assert_eq!(ProjectionKind::Prism.label(), "prism");
        assert_eq!(ProjectionKind::Nvfp4.label(), "nvfp4");

        let tally = ProjectionTally {
            count: 2,
            params: 1024,
            resident_bytes: 512,
            unmeasured: 0,
        };
        assert_eq!(tally.bits_per_param(), Some(4.0));
        let unmeasured = ProjectionTally {
            unmeasured: 1,
            ..tally
        };
        assert_eq!(unmeasured.bits_per_param(), None);

        let mut census = ProjectionCensus {
            ggml: tally,
            ggml_4bit: 2,
            ..Default::default()
        };
        assert_eq!(census.resident_quantization(), R::Bits(4));
        census.merge(&ProjectionCensus {
            dense: tally,
            ..Default::default()
        });
        assert_eq!(census.resident_quantization(), R::Bits(4));
        census.merge(&ProjectionCensus {
            ggml: ProjectionTally {
                count: 1,
                ..Default::default()
            },
            ggml_8bit: 1,
            ..Default::default()
        });
        assert_eq!(census.resident_quantization(), R::Unknown);
    }

    #[test]
    fn ggml_block_storage_and_mlx_affine_geometry_detection() {
        assert_eq!(
            ggml_block_storage(true, &[8, 2, 34]),
            Some((GgmlDType::Q8_0, 8, 64))
        );
        assert_eq!(
            ggml_block_storage(true, &[4, 8, 2, 18]),
            Some((GgmlDType::Q4_0, 32, 64))
        );
        assert_eq!(
            ggml_block_storage(true, &[2, 1, 144]),
            Some((GgmlDType::Q4K, 2, 256))
        );
        assert_eq!(ggml_block_storage(false, &[8, 2, 34]), None);

        assert_eq!(
            mlx_affine_q8_in_dim(true, (16, 32), (16, 2), (16, 2), 64),
            Some(128)
        );
        assert_eq!(
            mlx_affine_in_dim(4, true, (16, 16), (16, 2), (16, 2), 64),
            Some(128)
        );
        assert_eq!(
            mlx_affine_q8_in_dim(true, (16, 16), (16, 2), (16, 2), 64),
            None
        );
        assert_eq!(Q8RowEmbedding::device_bytes(6, 64), Some(6 * 64 + 12 * 2));
    }
}

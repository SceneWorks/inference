//! Selected output rows of an LM head (story SW-7197): `y[i, s, j] = x[i, s] · W[ids[i, j]]` —
//! each of a group's rows scored against only the `K` vocabulary rows its ids name.
//!
//! A DSpine block draft rescores every refined `(row, sender)` pair against the row's `K`
//! candidates. Run through the whole head that is `drafts × K` full-vocabulary logit rows — about
//! `2 · drafts · K · vocab · hidden` FLOPs a proposal (0.6 TFLOP for the 27B at `K = 16`) — of
//! which `K` columns per row are read. [`HeadRows`] reads only those rows of the head, whatever
//! its format:
//!
//! * **Dense** (a bf16 head, or a float GGUF matrix): the `K` rows of each group gathered, then
//!   one batched matmul — the head's own `x · Wᵀ` over fewer rows.
//! * **GGML blocks on the host**: each group's `K` rows of blocks copied out (a row is whole
//!   blocks, so its bytes are contiguous) into a `[K, hidden]` block matrix that runs the head's
//!   own forward — `QMatMul` (activation quantized as the head's is), or, for a head whose
//!   forward dequantizes, the dequantized rows' matmul. Bit-identical to the head's columns.
//! * **GGML blocks on CUDA**: the indexed GEMV over a one-row-per-id address table
//!   ([`IndexedExperts::ggml_rows`]) — every row read in place, one launch, no copy of the head.
//! * **Anything else** (an NVFP4 or Prism head, GGML blocks on Metal): no in-place row access, so
//!   the whole head runs and the columns are gathered — [`HeadRows::Full`], named.

use std::sync::Arc;

use candle_core::quantized::{QMatMul, QTensor};
use candle_core::{DType, Device, Module, Tensor};
use candle_quant_kernels::{IndexedExperts, MoeGemvError, MoeRows};

use crate::error::{Error, Result};
use crate::primitives::projection::Projection;

/// How an LM head's selected rows are read (see the module docs).
pub enum HeadRows {
    /// A dense `[vocab, hidden]` weight: rows gathered by id.
    Dense(Tensor),
    /// GGML blocks on the host: each group's rows of blocks copied out and run as the head runs.
    Host {
        /// The head's resident blocks.
        weight: Arc<QTensor>,
        /// The head's forward dequantizes the weight (the MLX-affine Q8 tier) instead of running
        /// `QMatMul`.
        dequant: bool,
    },
    /// GGML blocks on CUDA: every row addressed in place by the indexed GEMV. Boxed: the bank
    /// (device table, kernels) dwarfs the other variants, and it is built once per load.
    Indexed(Box<IndexedExperts>),
    /// No in-place row access: the whole head, then the columns. The reason is a stable label.
    Full(&'static str),
}

impl std::fmt::Debug for HeadRows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HeadRows({})", self.label())
    }
}

impl HeadRows {
    /// The row access `head` allows. Building the CUDA table uploads one 8-byte address per
    /// vocabulary row ([`Self::resident_bytes`]) and compiles the GEMV once.
    pub fn new(head: &Projection) -> Result<Self> {
        Ok(match head {
            Projection::Dense(l) if l.bias().is_none() => Self::Dense(l.weight().clone()),
            Projection::Dense(_) => Self::Full("bias"),
            Projection::Quantized(q) => {
                if let Some(dense) = q.dense_weight() {
                    return Ok(Self::Dense(dense.clone()));
                }
                let Some((weight, dequant)) = q.indexed_source() else {
                    return Ok(Self::Full("bias"));
                };
                match weight.device() {
                    Device::Cpu => Self::Host { weight, dequant },
                    Device::Cuda(_) => match IndexedExperts::ggml_rows(&weight, dequant)
                        .and_then(|bank| bank.available().map(|()| bank))
                    {
                        Ok(bank) => Self::Indexed(Box::new(bank)),
                        Err(MoeGemvError::Candle(e)) => return Err(Error::Candle(e)),
                        Err(e) => Self::Full(e.label()),
                    },
                    Device::Metal(_) => Self::Full("metal"),
                }
            }
            Projection::Prism(_) => Self::Full("prism"),
            Projection::Nvfp4(_) => Self::Full("nvfp4"),
        })
    }

    /// `dense`, `host`, `indexed`, or the [`Self::Full`] reason.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Dense(_) => "dense",
            Self::Host { .. } => "host",
            Self::Indexed(_) => "indexed",
            Self::Full(why) => why,
        }
    }

    /// Whether the selected rows are read without running the whole head.
    pub fn is_gathered(&self) -> bool {
        !matches!(self, Self::Full(_))
    }

    /// Device bytes held while this access lives: the CUDA address table.
    pub fn resident_bytes(&self) -> u64 {
        match self {
            Self::Indexed(bank) => bank.table_bytes() as u64,
            _ => 0,
        }
    }

    /// Device bytes one [`Self::scores`] of `groups` groups of `senders` rows against `k` ids
    /// each allocates beyond its input, for a `[vocab, hidden]` head and `act`-wide activations:
    /// the gathered rows and the matmul's output (dense); the Q8_1 activation copy and the f32
    /// output (CUDA); the whole head's f32 logits for every row plus their compute-dtype
    /// projection (the [`Self::Full`] fallback). The host path allocates on the host only, priced
    /// as the dense one. `None` on overflow.
    pub fn workspace_bytes(
        &self,
        groups: u64,
        senders: u64,
        k: u64,
        vocab: u64,
        hidden: u64,
        act: u64,
    ) -> Option<u64> {
        let rows = groups.checked_mul(senders)?;
        let out = rows.checked_mul(k)?.checked_mul(4 + act)?;
        match self {
            Self::Dense(_) | Self::Host { .. } => groups
                .checked_mul(k)?
                .checked_mul(hidden)?
                .checked_mul(act.max(4))?
                .checked_add(out),
            Self::Indexed(_) => {
                // The activation widened to f32, then its Q8_1 copy (rows padded to 512).
                let padded = hidden.div_ceil(512).checked_mul(512)?;
                rows.checked_mul(hidden.checked_mul(4)?.checked_add(padded / 32 * 36)?)?
                    .checked_add(out)
            }
            Self::Full(_) => rows
                .checked_mul(vocab)?
                .checked_mul(4 + act)?
                .checked_add(out),
        }
    }

    /// `rows` `[groups, senders, hidden]` (normalized, the head's activation dtype) scored against
    /// `ids` `[groups, k]` (`U32`): `[groups, senders, k]` f32, `out[g, s, j]` = the head's logit of
    /// `ids[g, j]` for row `(g, s)`. `head` is the projection this access was built from (the
    /// [`Self::Full`] fallback runs it).
    pub fn scores(&self, head: &Projection, rows: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let (groups, senders, hidden) = rows.dims3()?;
        let k = ids.dim(1)?;
        if ids.dim(0)? != groups {
            return Err(Error::Msg(format!(
                "head rows: {groups} row groups scored against {} id groups",
                ids.dim(0)?
            )));
        }
        let scores = match self {
            Self::Dense(weight) => {
                let picked = weight
                    .index_select(&ids.flatten_all()?, 0)?
                    .reshape((groups, k, hidden))?
                    .to_dtype(rows.dtype())?;
                __model_proof_output!(rows.contiguous()?.matmul(&picked.t()?))?
            }
            Self::Host { weight, dequant } => {
                let dtype = weight.dtype();
                let (vocab, cols) = weight.shape().dims2()?;
                let row_bytes = cols / dtype.block_size() * dtype.type_size();
                let data = weight.data()?;
                let ids = ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
                let mut out = Vec::with_capacity(groups);
                for (g, group) in ids.iter().enumerate() {
                    let mut bytes = Vec::with_capacity(k * row_bytes);
                    for &id in group {
                        let id = id as usize;
                        if id >= vocab {
                            return Err(Error::Msg(format!(
                                "head rows: id {id} past a {vocab}-row head"
                            )));
                        }
                        bytes.extend_from_slice(&data[id * row_bytes..(id + 1) * row_bytes]);
                    }
                    let picked = host_blocks(dtype, &bytes, k, cols)?;
                    let x = rows.get(g)?;
                    let y = if *dequant {
                        let dense = picked.dequantize(&Device::Cpu)?.to_dtype(x.dtype())?;
                        x.matmul(&dense.t()?)?
                    } else {
                        QMatMul::from_qtensor(picked)?.forward(&x.to_dtype(DType::F32)?)?
                    };
                    out.push(y);
                }
                __model_proof_output!(Tensor::stack(&out, 0))?
            }
            Self::Indexed(bank) => {
                let dtype = bank.format().io_dtype(rows.dtype()).ok_or_else(|| {
                    Error::Msg(format!("head rows: {:?} activations", rows.dtype()))
                })?;
                let x = rows
                    .reshape((groups * senders, hidden))?
                    .to_dtype(dtype)?
                    .contiguous()?;
                // Every sender row of group `g` reads group `g`'s ids.
                let ids = ids
                    .unsqueeze(1)?
                    .broadcast_as((groups, senders, k))?
                    .reshape((groups * senders, k))?
                    .contiguous()?;
                let y = bank
                    .forward(&x, &ids, MoeRows::PerToken)
                    .map_err(|e| Error::Candle(e.into()))?;
                __model_proof_output!(y.reshape((groups, senders, k)))?
            }
            Self::Full(_) => {
                let logits = head
                    .forward(&rows.reshape((groups * senders, hidden))?)?
                    .to_dtype(DType::F32)?;
                let index = ids
                    .unsqueeze(1)?
                    .broadcast_as((groups, senders, k))?
                    .reshape((groups * senders, k))?
                    .contiguous()?;
                logits.gather(&index, 1)?.reshape((groups, senders, k))?
            }
        };
        Ok(scores.to_dtype(DType::F32)?)
    }
}

/// A `[rows, cols]` [`QTensor`] of `dtype` on the host from its raw blocks, read from u64-backed
/// storage (candle reinterprets the bytes as blocks with 2-byte-aligned fields).
fn host_blocks(
    dtype: candle_core::quantized::GgmlDType,
    bytes: &[u8],
    rows: usize,
    cols: usize,
) -> Result<QTensor> {
    let mut words = vec![0u64; bytes.len().div_ceil(8)];
    // SAFETY: `words` owns `words.len() * 8 >= bytes.len()` initialized bytes, and u8 has no
    // alignment requirement; the byte view does not outlive `words`.
    let aligned =
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr().cast::<u8>(), bytes.len()) };
    aligned.copy_from_slice(bytes);
    Ok(candle_core::quantized::ggml_file::qtensor_from_ggml(
        dtype,
        aligned,
        vec![rows, cols],
        &Device::Cpu,
    )?)
}

#[cfg(test)]
mod tests;

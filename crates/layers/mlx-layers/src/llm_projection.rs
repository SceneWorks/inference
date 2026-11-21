//! LLM-specific projection extensions on the shared MLX layer spine (`SW-6957` / `SW-F335`).
//!
//! Unifies `Projection`, `KvProjection`, `QuantSpec`, `QuantGrid`, `QuantizedLinear`,
//! `QuantizedEmbedding`, and `quantize_on_stored_grid` below both media and LLM runtimes while
//! preserving `Projection::{Dense, Quantized, Prism}`'s exact operation order (`AP13 class (a)`).

use mlx_rs::ops::{add, dequantize, matmul, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype};
use thiserror::Error;

use crate::prism::PrismLinear;
use crate::{AdaptableLinear, LinearBase, ScaleContext};

/// Errors surfaced by the MLX LLM projection and tensor primitives.
#[derive(Debug, Error)]
pub enum Error {
    /// An MLX device operation failed (shape mismatch, allocation, command-buffer error, …).
    #[error("mlx op failed: {0}")]
    Mlx(#[from] mlx_rs::error::Exception),

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

    /// Generation was cancelled before it could run. Kept typed so the conformance suite and any
    /// consumer can tell cancellation apart from a real error.
    #[error("cancelled")]
    Canceled,

    /// A freshly loaded weight buffer whose GPU view never converged on the bytes the CPU holds
    /// (sc-22414). Typed so a consumer can tell "the host's Metal mapping is lagging" from a
    /// generic failure; see `primitives::coherence`.
    #[error(
        "GPU view of `{name}` ({bytes} bytes) never matched the CPU view after {attempts} reads \
         (cpu checksum {cpu:#x}, gpu checksum {gpu:#x}): the Metal mapping of a freshly loaded \
         buffer is stale — retry the load once the host's write-back pressure subsides (sc-22414)"
    )]
    IncoherentLoad {
        /// The tensor key.
        name: String,
        /// Its byte size.
        bytes: usize,
        /// The CPU-stream checksum (the reference).
        cpu: u64,
        /// The last GPU-stream checksum observed.
        gpu: u64,
        /// GPU reads attempted.
        attempts: u32,
    },

    /// A constraint mask allowed no token of a logits row (story sc-24434): there is nothing the
    /// sampler may draw. Typed so a caller can tell a dead-ended constraint from a backend fault;
    /// the sampler never falls back to a forbidden id.
    #[error("the constraint mask allows no token of the {vocab}-entry logits row")]
    NoAllowedToken {
        /// The logits row's width.
        vocab: usize,
    },

    /// A forward would take a DeltaNet checkpoint window past the tokens it was armed to record
    /// (story sc-24435) — more than the speculative width admission priced. Refused before the
    /// forward runs, with the cache untouched; typed so the cap is never mistaken for another
    /// fault.
    #[error(
        "the DeltaNet checkpoint window holds {recorded} tokens and is armed for {max_tokens}: a \
         {requested}-token forward would exceed it"
    )]
    CheckpointWindowFull {
        /// Tokens the window has recorded.
        recorded: i32,
        /// Tokens the refused forward carried.
        requested: i32,
        /// The most the window records.
        max_tokens: i32,
    },
    /// A request's memory admission refused it mid-generation (a compressed KV cache's dense
    /// transition that the fresh memory budget cannot hold, sc-20682). Kept typed so the provider
    /// surfaces core-llm's `RequestResourceExhausted` rather than a backend failure.
    #[error("{0}")]
    ResourceExhausted(core_llm::RequestResourceExhausted),

    /// Anything else, with a human-readable message.
    #[error("{0}")]
    Msg(String),
}

/// Result alias for MLX LLM projection primitives.
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

impl From<crate::Error> for Error {
    fn from(e: crate::Error) -> Self {
        match e {
            crate::Error::Mlx(ex) => Error::Mlx(ex),
            crate::Error::MissingTensor(s) => Error::MissingTensor(s),
            crate::Error::Io(io) => Error::Io(io),
            crate::Error::Canceled => Error::Canceled,
            crate::Error::Unsupported(s) => Error::Unsupported(s),
            crate::Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            } => Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            },
            other => Error::Msg(other.to_string()),
        }
    }
}

impl From<Error> for crate::Error {
    fn from(e: Error) -> Self {
        match e {
            Error::Mlx(ex) => crate::Error::Mlx(ex),
            Error::MissingTensor(s) => crate::Error::MissingTensor(s),
            Error::Io(io) => crate::Error::Io(io),
            Error::Canceled => crate::Error::Canceled,
            Error::Unsupported(s) => crate::Error::Unsupported(s),
            Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            } => crate::Error::IncoherentLoad {
                name,
                bytes,
                cpu,
                gpu,
                attempts,
            },
            other => crate::Error::Msg(other.to_string()),
        }
    }
}

/// `x @ weight.t() (+ bias)`. `weight` is `[out, in]` (HF layout); `x` is `[..., in]`.
pub fn linear(x: &Array, weight: &Array, bias: Option<&Array>) -> Result<Array> {
    let y = matmul(x, weight.t())?;
    match bias {
        Some(b) => Ok(add(&y, b)?),
        None => Ok(y),
    }
}

/// Group-wise affine quantization parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// Elements per quantization group.
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
    /// How load-time quantization rounds each weight to its group's grid. Describes the
    /// quantizer, not the packed layout: both grids produce the same `weight` / `scales` /
    /// `biases` shapes and dtypes.
    pub grid: QuantGrid,
}

/// The grid a load-time quantization rounds each weight to (sc-26266).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuantGrid {
    /// MLX's `quantize` as-is: codes rounded to the f32 grid, scales and biases then cast to
    /// the weight's dtype.
    #[default]
    Computed,
    /// Codes rounded to the scales and biases as stored
    /// ([`quantize_on_stored_grid`]) — the same bytes, ~0.64 x the
    /// weight error at 8 bits on a 16-bit weight.
    Stored,
}

impl QuantSpec {
    /// 4-bit, group size 64.
    pub fn q4() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            grid: QuantGrid::Computed,
        }
    }

    /// 8-bit, group size 64.
    pub fn q8() -> Self {
        Self {
            group_size: 64,
            bits: 8,
            grid: QuantGrid::Computed,
        }
    }
}

/// A token embedding table stored in MLX's group-wise packed affine format.
///
/// Unlike a linear projection, embedding lookup selects packed rows first and dequantizes only the
/// requested tokens. This preserves the snapshot's physical representation instead of expanding a
/// vocabulary-sized dense table at load time.
#[derive(Debug, Clone)]
pub struct QuantizedEmbedding {
    /// Packed `[vocab, hidden * bits / 32]` words.
    pub weight: Array,
    /// Per-row, per-group scales.
    pub scales: Array,
    /// Per-row, per-group affine biases.
    pub biases: Array,
    /// Elements per quantization group.
    pub group_size: i32,
    /// Bits per weight.
    pub bits: i32,
    hidden_size: i32,
}

impl QuantizedEmbedding {
    /// Validate and retain already-quantized embedding parts from a snapshot, holding the affine
    /// `scales`/`biases` in the model's `compute` dtype (sc-20671: a stored F16 scale must not
    /// promote BF16 activations to F32).
    pub fn from_quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
        compute: Dtype,
    ) -> Result<Self> {
        let weight_shape = weight.shape();
        let scales_shape = scales.shape();
        let biases_shape = biases.shape();
        if weight_shape.len() != 2 || scales_shape.len() != 2 || biases_shape != scales_shape {
            return Err(Error::Config(format!(
                "quantized embedding parts must be rank-2 with matching scale/bias shapes: weight={weight_shape:?}, scales={scales_shape:?}, biases={biases_shape:?}"
            )));
        }
        if weight_shape[0] != scales_shape[0] || group_size <= 0 || !matches!(bits, 4 | 8) {
            return Err(Error::Config(format!(
                "invalid quantized embedding geometry: weight={weight_shape:?}, scales={scales_shape:?}, group_size={group_size}, bits={bits}"
            )));
        }
        let values_per_word = 32 / bits;
        let hidden_size = weight_shape[1]
            .checked_mul(values_per_word)
            .ok_or_else(|| Error::Config("quantized embedding width overflow".into()))?;
        if hidden_size % group_size != 0 || scales_shape[1] != hidden_size / group_size {
            return Err(Error::Config(format!(
                "quantized embedding metadata does not reconstruct hidden size {hidden_size}: scales={scales_shape:?}, group_size={group_size}"
            )));
        }
        Ok(Self {
            weight,
            scales: in_compute_dtype(scales, compute)?,
            biases: in_compute_dtype(biases, compute)?,
            group_size,
            bits,
            hidden_size,
        })
    }

    /// The embedding width the packed table reconstructs (`packed words · 32 / bits`).
    pub fn hidden_size(&self) -> i32 {
        self.hidden_size
    }

    /// Gather packed vocabulary rows and dequantize only the requested `[batch, sequence]` tokens.
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let ids_shape = ids.shape();
        if ids_shape.len() != 2 {
            return Err(Error::Msg(format!(
                "quantized embedding ids must be [batch, sequence], got {ids_shape:?}"
            )));
        }
        let flat = ids.reshape(&[-1])?;
        let weight = self.weight.take_axis(&flat, 0)?;
        let scales = self.scales.take_axis(&flat, 0)?;
        let biases = self.biases.take_axis(&flat, 0)?;
        let rows = dequantize(&weight, &scales, Some(&biases), self.group_size, self.bits)?;
        Ok(rows.reshape(&[ids_shape[0], ids_shape[1], self.hidden_size])?)
    }

    /// Reuse a tied quantized embedding as the vocabulary projection without materializing it.
    pub fn tied_linear(&self) -> QuantizedLinear {
        QuantizedLinear {
            weight: self.weight.clone(),
            scales: self.scales.clone(),
            biases: self.biases.clone(),
            group_size: self.group_size,
            bits: self.bits,
            bias: None,
        }
    }
}

/// Group-wise affine quantization whose integer codes are chosen against the scales and biases
/// **as stored** (sc-26266).
///
/// MLX's `quantize` derives each group's grid in f32, rounds every element to it, and only then
/// casts the scale and bias to the weight's dtype. For a 16-bit weight the stored grid is
/// therefore not the grid the codes were rounded to: a bf16 scale is off by up to 2⁻⁹ of
/// itself, which at code 255 is half a quantization step on its own, and the bf16 bias (the
/// group's edge value) shifts the whole group by up to a quarter step. At 8 bits that is ~1.6 x
/// the rounding error the bytes pay for (measured on the Iris-3B and Qwen3-VL-4B projections:
/// 0.86 % relative weight error against 0.55 % with the codes recomputed); at 4 bits the
/// step is 17 x wider and the mismatch is noise.
///
/// This keeps MLX's per-group `scales` / `biases` exactly (same dtype, same values, so the
/// layout and the resident bytes are `quantize`'s) and re-rounds every element against them:
/// `q = clip(round((w − bias) / scale), 0, 2^bits − 1)`, packed in MLX's layout (element `j`
/// of each 32-bit word at bits `[j·bits, (j + 1)·bits)`). The result is a drop-in for
/// `quantize`'s triple (`dequantize` / `quantized_matmul` read it unchanged). `bits` must
/// divide 32 and `group_size` the last axis.
pub fn quantize_on_stored_grid(
    weight: &Array,
    group_size: i32,
    bits: i32,
) -> Result<(Array, Array, Array)> {
    let shape = weight.shape().to_vec();
    let inner = shape.last().copied().unwrap_or(0);
    if !matches!(bits, 2 | 4 | 8) || group_size <= 0 || inner % group_size != 0 {
        return Err(Error::Config(format!(
            "quantize_on_stored_grid: {bits}-bit groups of {group_size} cannot pack a weight of \
             shape {shape:?}"
        )));
    }
    let (_, scales, biases) = quantize(weight, group_size, bits)?;
    let lead = &shape[..shape.len() - 1];
    let grouped = |tail: &[i32]| -> Vec<i32> { lead.iter().chain(tail).copied().collect() };
    let groups = inner / group_size;
    let w = weight
        .as_dtype(Dtype::Float32)?
        .reshape(&grouped(&[groups, group_size]))?;
    let s = scales
        .as_dtype(Dtype::Float32)?
        .reshape(&grouped(&[groups, 1]))?;
    let b = biases
        .as_dtype(Dtype::Float32)?
        .reshape(&grouped(&[groups, 1]))?;
    let top = ((1u32 << bits) - 1) as f32;
    let codes = mlx_rs::ops::clip(&w.subtract(&b)?.divide(&s)?.round(None)?, (0.0f32, top))?
        .as_dtype(Dtype::Uint32)?;
    // Disjoint bit fields, so the weighted sum is the bitwise OR (and never overflows a word).
    let per_word = 32 / bits;
    let place: Vec<u32> = (0..per_word).map(|j| 1u32 << (bits * j)).collect();
    let packed = codes
        .reshape(&grouped(&[inner / per_word, per_word]))?
        .multiply(Array::from_slice(&place, &[per_word]))?
        .sum_axis(-1, false)?
        .as_dtype(Dtype::Uint32)?;
    Ok((packed, scales, biases))
}

/// Cast a stored affine quantization parameter to the model's compute dtype.
///
/// MLX's affine `quantized_matmul` produces `promote_types(x, scales)` (and `dequantize` the scale
/// dtype), so a stored scale dtype that differs from the activations silently widens every
/// downstream tensor: BF16
/// activations against the F16 scales mlx-community checkpoints ship promote to **F32**, and the
/// whole decoder — including the K/V handed to the cache — then runs 4-byte (sc-20671). Holding the
/// parameters in the compute dtype at load keeps activations, and therefore the KV cache, in the
/// compute dtype whatever dtype the snapshot stored them in.
pub fn in_compute_dtype(parameter: Array, compute: Dtype) -> Result<Array> {
    if parameter.dtype() == compute {
        Ok(parameter)
    } else {
        Ok(parameter.as_dtype(compute)?)
    }
}

/// A linear projection whose weight is stored group-wise quantized.
///
/// Forward is `quantized_matmul(x, weight, scales, biases, transpose = true, ...)`, which computes
/// `x @ weight.t()` against the dequantized weight — the quantized analogue of [`linear`].
/// `transpose = true` matches the HF `[out, in]` weight layout.
#[derive(Debug, Clone)]
pub struct QuantizedLinear {
    /// Packed quantized weight.
    pub weight: Array,
    /// Per-group scales.
    pub scales: Array,
    /// Per-group biases (zero-points).
    pub biases: Array,
    /// Elements per quantization group (e.g. 64).
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
    /// Optional additive bias applied after the matmul.
    pub bias: Option<Array>,
}

impl QuantizedLinear {
    /// Retain already-quantized projection parts from a snapshot (the packed `weight` plus its
    /// per-group `scales`/`biases`), holding the affine parameters — and the optional additive
    /// `bias` — in the model's `compute` dtype (sc-20671: a stored F16 scale must not promote BF16
    /// activations to F32). The packed weight is used as-is.
    pub fn from_stored(
        weight: Array,
        scales: Array,
        biases: Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
        compute: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            weight,
            scales: in_compute_dtype(scales, compute)?,
            biases: in_compute_dtype(biases, compute)?,
            group_size,
            bits,
            bias: bias.map(|b| in_compute_dtype(b, compute)).transpose()?,
        })
    }

    /// Quantize a dense `[out, in]` weight into a `QuantizedLinear`. `group_size` must divide the
    /// input dimension; `bits` is typically 4 or 8.
    pub fn quantize(
        weight: &Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
    ) -> Result<Self> {
        let (w, scales, biases) = quantize(weight, group_size, bits)?;
        Ok(Self {
            weight: w,
            scales,
            biases,
            group_size,
            bits,
            bias,
        })
    }

    /// [`quantize`](Self::quantize) with every code rounded to the stored scales and biases
    /// ([`quantize_on_stored_grid`]): the same layout and bytes, a lower weight error.
    pub fn quantize_on_stored_grid(
        weight: &Array,
        group_size: i32,
        bits: i32,
        bias: Option<Array>,
    ) -> Result<Self> {
        let (w, scales, biases) = quantize_on_stored_grid(weight, group_size, bits)?;
        Ok(Self {
            weight: w,
            scales,
            biases,
            group_size,
            bits,
            bias,
        })
    }

    /// Forward pass: `x @ dequant(weight).t() (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            Some(&self.biases),
            true, // transpose: weight is [out, in]
            self.group_size,
            self.bits,
        )?;
        match &self.bias {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }

    /// Reconstruct the dense weight (mostly for parity tests / inspection).
    pub fn dequantize_weight(&self) -> Result<Array> {
        Ok(dequantize(
            &self.weight,
            &self.scales,
            Some(&self.biases),
            self.group_size,
            self.bits,
        )?)
    }

    /// Convert into an [`AdaptableLinear`] on the shared `mlx-layers` spine (`BACK-1`, `BACK-4`).
    pub fn to_adaptable_linear(&self) -> AdaptableLinear {
        AdaptableLinear::from_quantized_parts(
            self.weight.clone(),
            self.scales.clone(),
            self.biases.clone(),
            self.bias.clone(),
            self.group_size,
            self.bits,
        )
    }

    /// Construct from an [`mlx_rs::nn::QuantizedLinear`] base on the shared `mlx-layers` spine.
    pub fn from_mlx_quantized(q: &mlx_rs::nn::QuantizedLinear) -> Self {
        Self {
            weight: q.inner.weight.value.clone(),
            scales: q.scales.value.clone(),
            biases: q.biases.value.clone(),
            group_size: q.group_size,
            bits: q.bits,
            bias: q.inner.bias.value.clone(),
        }
    }
}

/// A linear projection weight, dense or quantized.
#[derive(Debug)]
pub enum Projection {
    /// A dense `[out, in]` weight with an optional `[out]` bias.
    Dense {
        /// The `[out, in]` weight (HF layout).
        weight: Array,
        /// Optional additive `[out]` bias (Qwen2 / GLM-4 attention carry q/k/v bias; Llama / Qwen3 /
        /// Phi-3 do not).
        bias: Option<Array>,
    },
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
    /// Prism ternary affine weight with a folded Hadamard activation rotation.
    Prism(PrismLinear),
    /// Original signed codes, F32 row scales and shared rotation.
    Native(crate::NativeProjection),
}

impl Projection {
    pub fn expert_shape(&self) -> Result<[i32; 2]> {
        match self {
            Self::Native(p) => Ok(p.shape()),
            Self::Dense { weight, bias: None } => weight
                .shape()
                .try_into()
                .map_err(|_| Error::Config("expert requires a rank-two matrix".into())),
            Self::Quantized(q) if q.bias.is_none() && q.scales.ndim() == 2 => {
                Ok([q.scales.shape()[0], q.scales.shape()[1] * q.group_size])
            }
            _ => Err(Error::Unsupported(
                "expert projection has unsupported layout or additive bias".into(),
            )),
        }
    }

    /// Bind a described packed projection before a model casts a source to its compute dtype.
    pub fn native(
        w: &(impl crate::MlxWeightSource + ?Sized),
        key: &str,
        bias: bool,
    ) -> Result<Option<Self>> {
        let Some(base) = key.strip_suffix(".weight") else {
            return Ok(None);
        };
        if !crate::NativeProjection::is_declared(w, base, key) {
            return Ok(None);
        }
        Ok(crate::NativeProjection::from_weights(w, base, bias)?.map(Self::Native))
    }

    /// Bind the original weight before applying the model's dense compute cast.
    pub fn from_named(
        w: &(impl crate::MlxWeightSource + ?Sized),
        key: &str,
        bias: Option<Array>,
        compute: Option<Dtype>,
    ) -> Result<Self> {
        let base = key
            .strip_suffix(".weight")
            .or_else(|| key.strip_suffix("_weight"))
            .unwrap_or(key);
        if let Some(mut native) = crate::NativeProjection::from_key(w, base, key, false)? {
            if let Some(bias) = bias {
                native.set_bias(bias)?;
            }
            return Ok(Self::Native(native));
        }
        let stored = w.require(key)?;
        let weight = match compute {
            Some(dtype) => stored.as_dtype(dtype)?,
            None => stored.clone(),
        };
        Ok(Self::Dense { weight, bias })
    }

    /// Preserve a decoder's dense accumulation scheme when its base is dense.
    pub fn forward_with_dense(
        &self,
        x: &Array,
        dense: fn(&Array, &Array, Option<&Array>) -> Result<Array>,
    ) -> Result<Array> {
        match self {
            Self::Dense { weight, bias } => dense(x, weight, bias.as_ref()),
            _ => self.forward(x),
        }
    }

    /// Unpack a fused QKV or gate/up projection along its output axis. Native subprojections
    /// retain their own matching slices of codes and scales and share the source rotation.
    pub fn load_split(
        w: &(impl crate::MlxWeightSource + ?Sized),
        key: &str,
        ends: &[i32],
        dtype: Dtype,
        quant: Option<QuantSpec>,
    ) -> Result<Vec<Self>> {
        if let Some(Self::Native(native)) = Self::native(w, key, false)? {
            let mut start = 0;
            return ends
                .iter()
                .copied()
                .chain(std::iter::once(native.shape()[0]))
                .map(|end| {
                    let part = native.slice_rows(start, end)?;
                    start = end;
                    Ok(Self::Native(part))
                })
                .collect();
        }
        let weight = w.require(key)?.as_dtype(dtype)?;
        mlx_rs::ops::split_sections(&weight, ends, 0)?
            .into_iter()
            .map(|part| Self::load(part, quant))
            .collect()
    }

    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Array, quant: Option<QuantSpec>) -> Result<Self> {
        Self::load_with_bias(weight, None, quant)
    }

    /// Load from a dense `[out, in]` weight plus an optional `[out]` bias (Qwen2 / GLM-4 attention
    /// carry q/k/v bias), quantizing the weight if `quant` is set. The bias is always applied dense.
    pub fn load_with_bias(
        weight: Array,
        bias: Option<Array>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense { weight, bias }),
            Some(q) => Ok(Projection::Quantized(match q.grid {
                QuantGrid::Computed => {
                    QuantizedLinear::quantize(&weight, q.group_size, q.bits, bias)?
                }
                QuantGrid::Stored => {
                    QuantizedLinear::quantize_on_stored_grid(&weight, q.group_size, q.bits, bias)?
                }
            })),
        }
    }

    /// Load from **already-quantized** parts stored in a snapshot (the packed `weight`, per-group
    /// `scales`/`biases`) — the read side of the GGUF converter's optional MLX requant and of
    /// mlx-community checkpoints. No quantization happens here: the packed weight is used as-is and
    /// the affine parameters are held in the model's `compute` dtype
    /// ([`QuantizedLinear::from_stored`]).
    pub fn from_quantized(
        weight: Array,
        scales: Array,
        biases: Array,
        spec: QuantSpec,
        compute: Dtype,
    ) -> Result<Self> {
        Ok(Projection::Quantized(QuantizedLinear::from_stored(
            weight,
            scales,
            biases,
            spec.group_size,
            spec.bits,
            None,
            compute,
        )?))
    }

    /// Construct a [`Projection`] from an [`AdaptableLinear`] (`crate::Linear`) on the shared
    /// `mlx-layers` spine (`BACK-1`, `BACK-4`).
    pub fn from_adaptable_linear(linear: &AdaptableLinear) -> Result<Self> {
        if !linear.adapters().is_empty() {
            return Err(Error::Unsupported(
                "AdaptableLinear with installed LoRA/LoKr residuals cannot convert to a base-only LLM Projection".into(),
            ));
        }
        match &linear.base {
            LinearBase::Dense(l) => Ok(Projection::Dense {
                weight: l.weight.value.clone(),
                bias: l.bias.value.clone(),
            }),
            LinearBase::Quantized(q) => Ok(Projection::Quantized(
                QuantizedLinear::from_mlx_quantized(q),
            )),
            LinearBase::ConvRot(p) => Ok(Self::Native(crate::NativeProjection::ConvRot(p.clone()))),
            LinearBase::Fp8(p) => Ok(Self::Native(crate::NativeProjection::Fp8(p.clone()))),
            LinearBase::W4A8(_) => Err(Error::Unsupported(
                "W4A8 adaptable linear has no affine/dense LLM Projection".into(),
            )),
        }
    }

    /// Convert a dense or affine-quantized [`Projection`] into an [`AdaptableLinear`] (`crate::Linear`)
    /// on the shared `mlx-layers` spine (`BACK-1`, `BACK-4`).
    pub fn to_adaptable_linear(&self) -> Result<AdaptableLinear> {
        match self {
            Projection::Dense { weight, bias } => {
                Ok(AdaptableLinear::dense(weight.clone(), bias.clone()))
            }
            Projection::Quantized(q) => Ok(q.to_adaptable_linear()),
            Projection::Native(p) => Ok(AdaptableLinear::from_native(p.clone())),
            Projection::Prism(_) => Err(Error::Unsupported(
                "Prism projection cannot convert to an affine/dense AdaptableLinear".into(),
            )),
        }
    }

    /// `x @ weightᵀ (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Projection::Dense { weight, bias } => linear(x, weight, bias.as_ref()),
            Projection::Quantized(q) => q.forward(x),
            Projection::Prism(p) => p.forward(x),
            Projection::Native(p) => Ok(p.forward(x)?),
        }
    }

    /// Evaluate `x @ weightᵀ (+ bias)` under the shared `mlx-layers` spine's request-scoped
    /// [`ScaleContext`] signature (`BACK-1`, `BACK-4`), preserving `Projection`'s exact operation
    /// order (`AP13 class (a)`).
    pub fn forward_with_scales(
        &self,
        x: &Array,
        _scales: Option<&ScaleContext<'_>>,
    ) -> Result<Array> {
        self.forward(x)
    }

    /// The weight rows for `ids` (any integer shape `[..]`) as dense `[.., in]` in the stored
    /// compute dtype: a gather of the dense weight, or of the packed codes, scales and biases
    /// followed by a dequantize of just those rows. A Prism projection is refused.
    pub fn rows(&self, ids: &Array) -> Result<Array> {
        let shape = ids.shape().to_vec();
        let flat = ids.reshape(&[-1])?;
        let rows = match self {
            Projection::Dense { weight, .. } => weight.take_axis(&flat, 0)?,
            Projection::Quantized(q) => {
                let biases = q.biases.take_axis(&flat, 0)?;
                mlx_rs::ops::dequantize(
                    q.weight.take_axis(&flat, 0)?,
                    q.scales.take_axis(&flat, 0)?,
                    Some(&biases),
                    q.group_size,
                    q.bits,
                )?
                .as_dtype(q.scales.dtype())?
            }
            Projection::Native(p) => p.rows(&flat)?,
            Projection::Prism(_) => {
                return Err(Error::Unsupported(
                    "a Prism projection has no dense rows to gather".into(),
                ))
            }
        };
        let width = *rows.shape().last().unwrap_or(&0);
        let mut out = shape;
        out.push(width);
        Ok(rows.reshape(&out)?)
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Projection::Quantized(_) | Projection::Prism(_) | Projection::Native(_)
        )
    }

    /// Elements a forward materializes when its activations are **wider** than this projection's
    /// stored dtype (sc-24446): MLX promotes a BF16 weight to the activation dtype for a dense
    /// matmul (a full copy of the weight), and a quantized matmul's BF16 scales and biases. A
    /// Prism projection rotates its input in its own dtype and promotes nothing here.
    pub fn promoted_elements(&self) -> u64 {
        let n = |a: &Array| a.size() as u64;
        let bias = |b: &Option<Array>| b.as_ref().map_or(0, n);
        match self {
            Projection::Dense { weight, bias: b } => n(weight) + bias(b),
            Projection::Quantized(q) => n(&q.scales) + n(&q.biases) + bias(&q.bias),
            Projection::Prism(_) | Projection::Native(_) => 0,
        }
    }

    /// The arrays this projection holds, for load-time materialization (sc-24446).
    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        match self {
            Projection::Dense { weight, bias } => {
                out.push(weight.clone());
                out.extend(bias.iter().cloned());
            }
            Projection::Quantized(q) => {
                out.extend([q.weight.clone(), q.scales.clone(), q.biases.clone()]);
                out.extend(q.bias.iter().cloned());
            }
            Projection::Prism(p) => p.push_arrays(out),
            Projection::Native(p) => p.push_arrays(out),
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
#[derive(Debug)]
pub struct KvProjection {
    k: Projection,
    /// `None` ⇒ `attention_k_eq_v`: the value heads come from `k`'s output.
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
    /// result (MLX arrays are refcounted, so this is a handle clone, not a copy).
    pub fn forward(&self, x: &Array) -> Result<(Array, Array)> {
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

    /// The arrays both halves hold, for load-time materialization (sc-24446).
    pub fn push_arrays(&self, out: &mut Vec<Array>) {
        self.k.push_arrays(out);
        if let Some(v) = &self.v {
            v.push_arrays(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_spec_defaults_and_grid_selection() {
        let q4 = QuantSpec::q4();
        assert_eq!(q4.group_size, 64);
        assert_eq!(q4.bits, 4);
        assert_eq!(q4.grid, QuantGrid::Computed);

        let q8 = QuantSpec::q8();
        assert_eq!(q8.group_size, 64);
        assert_eq!(q8.bits, 8);
        assert_eq!(q8.grid, QuantGrid::Computed);
        assert_eq!(QuantGrid::default(), QuantGrid::Computed);
    }

    #[test]
    fn error_bridges_preserve_typed_variants_across_mlx_layers_spine() {
        let missing: crate::Error = Error::MissingTensor("model.embed_tokens.weight".into()).into();
        assert!(
            matches!(missing, crate::Error::MissingTensor(ref k) if k == "model.embed_tokens.weight")
        );
        let back: Error = missing.into();
        assert!(matches!(back, Error::MissingTensor(ref k) if k == "model.embed_tokens.weight"));

        let unsupported: crate::Error = Error::Unsupported("prism gather".into()).into();
        assert!(matches!(unsupported, crate::Error::Unsupported(ref s) if s == "prism gather"));
        let back: Error = unsupported.into();
        assert!(matches!(back, Error::Unsupported(ref s) if s == "prism gather"));

        let canceled: crate::Error = Error::Canceled.into();
        assert!(matches!(canceled, crate::Error::Canceled));
        let back: Error = canceled.into();
        assert!(matches!(back, Error::Canceled));
    }
}

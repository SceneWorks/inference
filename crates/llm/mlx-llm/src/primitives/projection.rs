//! A linear projection that is either dense or group-wise quantized.
//!
//! The decoders hold their attention/MLP projections behind this so quantize-on-load (story 7163)
//! is a load-time choice with no decoder changes: a dense `[out, in]` weight either stays dense
//! (`matmul(x, wᵀ)`) or is quantized to Q4/Q8 ([`QuantizedLinear`]).

use mlx_rs::Array;

use crate::error::Result;
use crate::primitives::nn::linear;
use crate::primitives::quant::QuantizedLinear;

/// Group-wise affine quantization parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// Elements per quantization group.
    pub group_size: i32,
    /// Bits per weight (4 or 8).
    pub bits: i32,
}

impl QuantSpec {
    /// 4-bit, group size 64.
    pub fn q4() -> Self {
        Self {
            group_size: 64,
            bits: 4,
        }
    }

    /// 8-bit, group size 64.
    pub fn q8() -> Self {
        Self {
            group_size: 64,
            bits: 8,
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
}

impl Projection {
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
            Some(q) => Ok(Projection::Quantized(QuantizedLinear::quantize(
                &weight,
                q.group_size,
                q.bits,
                bias,
            )?)),
        }
    }

    /// Load from **already-quantized** parts stored in a snapshot (the packed `weight`, per-group
    /// `scales`/`biases`) — the read side of the GGUF converter's optional MLX requant. No
    /// quantization happens here; the parts are used as-is.
    pub fn from_quantized(weight: Array, scales: Array, biases: Array, spec: QuantSpec) -> Self {
        Projection::Quantized(QuantizedLinear {
            weight,
            scales,
            biases,
            group_size: spec.group_size,
            bits: spec.bits,
            bias: None,
        })
    }

    /// `x @ weightᵀ (+ bias)`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Projection::Dense { weight, bias } => linear(x, weight, bias.as_ref()),
            Projection::Quantized(q) => q.forward(x),
        }
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Projection::Quantized(_))
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
}

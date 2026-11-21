//! Shared CUDA checkpoint FP8 projection.
use crate::cublaslt::{quantize_weight_fp8, CublasLt};
use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// An fp8 E4M3 linear: `y = (X · Wᵀ) · scale_w · scale_x` with a per-tensor-quantized weight and
/// dynamic per-tensor activation quant. Optional bias added back in the output dtype.
pub type ActivationQuantizer = fn(&Tensor) -> Result<(Tensor, f32)>;

pub struct Fp8Linear {
    dynamic_quantizer: Option<ActivationQuantizer>,
    w_fp8: crate::cublaslt::DevFp8,
    device: Device,
    dims: (usize, usize),
    input_scale: Option<f32>,
    block_scales: Option<Tensor>,
    scale_w: f32,
    bias: Option<Tensor>,
    lt: Arc<CublasLt>,
}

impl Fp8Linear {
    /// Quantize a dense `(N, K)` weight to fp8 E4M3 once and bind it to a cuBLASLt handle.
    pub fn new(weight: &Tensor, bias: Option<Tensor>, lt: Arc<CublasLt>) -> Result<Self> {
        let qw = quantize_weight_fp8(weight)?;
        Self::from_codes(qw.q, qw.scale, None, bias, lt)
    }

    /// Bind checkpoint E4M3FN codes and dequantization scales without re-quantizing.
    pub fn from_parts(
        codes: Tensor,
        scale_w: f32,
        input_scale: Option<f32>,
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        if codes.dtype() != candle_core::DType::F8E4M3 {
            candle_core::bail!("expected E4M3FN weight codes");
        }
        Self::from_codes(codes, scale_w, Some(input_scale.unwrap_or(1.0)), bias, lt)
    }

    /// A from-dense recipe may define its own activation rounding/scale floor, while sharing the original GEMM.
    pub fn from_dynamic_codes(
        codes: Tensor,
        scale_w: f32,
        lt: Arc<CublasLt>,
        activation_quantizer: ActivationQuantizer,
    ) -> Result<Self> {
        if codes.dtype() != candle_core::DType::F8E4M3 {
            candle_core::bail!("dynamic codes must be E4M3FN");
        }
        let mut base = Self::from_codes(codes, scale_w, None, None, lt)?;
        base.dynamic_quantizer = Some(activation_quantizer);
        Ok(base)
    }

    pub fn from_e5m2_parts(
        codes: Tensor,
        scale_w: f32,
        input_scale: Option<f32>,
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        if codes.dtype() != candle_core::DType::U8 {
            candle_core::bail!("expected raw E5M2 weight codes");
        }
        Self::from_codes(codes, scale_w, Some(input_scale.unwrap_or(1.0)), bias, lt)
    }

    fn from_codes(
        codes: Tensor,
        scale_w: f32,
        input_scale: Option<f32>,
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        if !crate::cublaslt::eight_bit_layout_supported(codes.dims())
            || !scale_w.is_finite()
            || input_scale.is_some_and(|s| !s.is_finite() || s <= 0.0)
        {
            candle_core::bail!("invalid scalar FP8 projection geometry or scales");
        }
        if bias.as_ref().is_some_and(|b| {
            b.dims() != [codes.dims()[0]] || !b.device().same_device(codes.device())
        }) {
            candle_core::bail!("FP8 bias must match weight output width and device");
        }
        let device = codes.device().clone();
        Ok(Self {
            dynamic_quantizer: None,
            dims: codes.dims2()?,
            block_scales: None,
            w_fp8: lt.stage_fp8(&codes)?,
            device,
            scale_w,
            input_scale,
            bias,
            lt,
        })
    }

    pub fn from_mxfp8_parts(
        codes: Tensor,
        block_scales: Tensor,
        logical: (usize, usize),
        bias: Option<Tensor>,
        lt: Arc<CublasLt>,
    ) -> Result<Self> {
        if codes.dtype() != candle_core::DType::F8E4M3
            || block_scales.dtype() != candle_core::DType::U8
            || codes.dims()
                != [
                    logical.0.next_multiple_of(32),
                    logical.1.next_multiple_of(32),
                ]
            || !codes.device().same_device(block_scales.device())
            || bias
                .as_ref()
                .is_some_and(|b| b.dims() != [logical.0] || !b.device().same_device(codes.device()))
        {
            candle_core::bail!("invalid MXFP8 shape, bias or scale storage");
        }
        let mut layer = Self::from_codes(codes, 1.0, None, None, lt)?;
        layer.dims = logical;
        layer.block_scales = Some(block_scales.contiguous()?);
        layer.bias = bias;
        Ok(layer)
    }

    pub fn format_id(&self) -> &'static str {
        use gen_core::checkpoint_codec::{
            FP8_E4M3_SCALAR_CODEC, FP8_E5M2_SCALAR_CODEC, MXFP8_CODEC,
        };
        if self.block_scales.is_some() {
            MXFP8_CODEC.codec_id
        } else if self.w_fp8.is_e5m2() {
            FP8_E5M2_SCALAR_CODEC.codec_id
        } else {
            FP8_E4M3_SCALAR_CODEC.codec_id
        }
    }

    pub fn dims(&self) -> (usize, usize) {
        self.dims
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Build sharing an existing handle for the device (constructs a fresh handle when `None`).
    pub fn from_device(weight: &Tensor, bias: Option<Tensor>, dev: &Device) -> Result<Self> {
        Self::new(weight, bias, Arc::new(CublasLt::new(dev)?))
    }

    /// `x`: `(..., K)`; flattened to `(M, K)` for the GEMM, then reshaped back. Output bf16.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_bias(x, self.bias.as_ref())
    }

    /// Checkpoint matmul rounds to the consuming dtype before adding bias, as in candle_nn::Linear.
    /// The legacy dynamically quantized recipe returns BF16 as before.
    pub fn forward_with_bias(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        use candle_core::DType;
        let source = self.input_scale.is_some() || self.block_scales.is_some();
        let output_dtype = if source { x.dtype() } else { DType::BF16 };
        if !matches!(output_dtype, DType::BF16 | DType::F16 | DType::F32) {
            candle_core::bail!("unsupported FP8 projection output {output_dtype:?}");
        }
        let f32_accumulator = source && output_dtype != DType::BF16;
        let (flat, restore) = flatten_tokens(x)?;
        if flat.dim(1)? != self.dims.1 || !flat.device().same_device(self.device()) {
            candle_core::bail!("FP8 input width or device differs from weight");
        }
        let y = if let Some(scales) = &self.block_scales {
            if f32_accumulator {
                self.lt
                    .mxfp8_linear_f32(&self.w_fp8, scales, &flat, self.dims.0)?
            } else {
                self.lt
                    .mxfp8_linear(&self.w_fp8, scales, &flat, self.dims.0)?
            }
        } else {
            let (qx, sx) = if let Some(quantize) = self.dynamic_quantizer {
                let (codes, scale) = quantize(&flat)?;
                (self.lt.stage_fp8(&codes)?, scale)
            } else {
                self.lt
                    .scaled_fp8_activation(&flat, self.input_scale, self.w_fp8.is_e5m2())?
            };
            if f32_accumulator {
                self.lt
                    .matmul_fp8_staged_f32(&self.w_fp8, self.scale_w, &qx, sx)?
            } else {
                self.lt
                    .matmul_fp8_staged(&self.w_fp8, self.scale_w, &qx, sx)?
            }
        };
        let y = restore(y)?.to_dtype(output_dtype)?;
        let y = match bias {
            Some(b) => y.broadcast_add(&b.to_dtype(y.dtype())?)?,
            None => y,
        };
        __model_proof_output!(Ok(y))
    }
}

/// Collapse leading dims to a `(M, K)` matrix and return a closure that restores the original
/// leading shape on the `(M, N)` output.
pub(crate) fn flatten_tokens(x: &Tensor) -> Result<(Tensor, impl Fn(Tensor) -> Result<Tensor>)> {
    let dims = x.dims().to_vec();
    let Some(&k) = dims.last() else {
        candle_core::bail!("FP8 linear expects at least one input dimension")
    };
    let m: usize = dims[..dims.len() - 1].iter().product();
    let flat = x.reshape((m, k))?;
    let lead = dims[..dims.len() - 1].to_vec();
    let restore = move |y: Tensor| -> Result<Tensor> {
        let n = y.dim(1)?;
        let mut out_shape = lead.clone();
        out_shape.push(n);
        y.reshape(out_shape)
    };
    Ok((flat, restore))
}

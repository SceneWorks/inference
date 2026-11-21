//! Shared CUDA checkpoint FP8 projection.
use crate::cublaslt::{quantize_weight_fp8, CublasLt};
use candle_core::{Device, Result, Tensor};
use std::sync::Arc;

/// An fp8 E4M3 linear: `y = (X · Wᵀ) · scale_w · scale_x` with a per-tensor-quantized weight and
/// dynamic per-tensor activation quant. Optional bias added back in the output dtype.
pub struct Fp8Linear {
    w_fp8: crate::cublaslt::DevFp8,
    device: Device,
    dims: (usize, usize),
    input_scale: Option<f32>,
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
            dims: codes.dims2()?,
            w_fp8: lt.stage_fp8(&codes)?,
            device,
            scale_w,
            input_scale,
            bias,
            lt,
        })
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
        let (flat, restore) = flatten_tokens(x)?;
        let (qx, scale_x) =
            self.lt
                .scaled_fp8_activation(&flat, self.input_scale, self.w_fp8.is_e5m2())?;
        let y = self
            .lt
            .matmul_fp8_staged(&self.w_fp8, self.scale_w, &qx, scale_x)?;
        let y = restore(y)?;
        match &self.bias {
            Some(b) => __model_proof_output!(y.broadcast_add(&b.to_dtype(y.dtype())?)),
            None => Ok(y),
        }
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

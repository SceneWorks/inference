//! SW-6126 rows: the fused fp8 / int8 activation quantizer against its reference op chains, and
//! the two 8-bit linears whose forwards it serves, at representative DiT activation shapes
//! (`(M tokens, K)`; `4118 x 6144` is the story's shape, `4096 x 3840` Z-Image at 1024², `16 x
//! 3840` a small batch), bf16 activations.

use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Result, Tensor};
use candle_gen::quant::act_quant::{
    fused_fp8, Fp8ScaleRule, ACT_QUANT_FP8_PATH, ACT_QUANT_INT8_PATH,
};
use candle_gen::quant::cublaslt::{
    quantize_activation_fp8, quantize_activation_fp8_reference, quantize_activation_int8,
    quantize_activation_int8_reference, quantize_weight_int8_per_channel, CublasLt, F8E4M3_MAX,
};
use candle_gen::quant::{Fp8Linear, Int8Linear};

use crate::harness::{Bench, Row};

const QUANT_SHAPES: &[(usize, usize)] = &[(4118, 6144), (4096, 3840), (16, 3840)];
/// `(M, K, N)` for the linear rows.
const LINEAR_SHAPES: &[(usize, usize, usize)] = &[(4118, 6144, 6144), (16, 3840, 3840)];

fn activation(dev: &Device, m: usize, k: usize) -> Result<Tensor> {
    Tensor::randn(0f32, 1.0, (m, k), dev)?.to_dtype(DType::BF16)
}

/// candle's own on-device chain with the stock `cast_f32_f8_e4m3` kernel (no CPU round trip):
/// the "stock cast chain" the fused kernel must beat. `None` when candle cannot cast on device.
fn stock_device_fp8(x: &Tensor) -> Result<Tensor> {
    let absmax = x.abs()?.flatten_all()?.max(0)?.to_dtype(DType::F32)?;
    let scale = (absmax.to_scalar::<f32>()? / F8E4M3_MAX).max(f32::MIN_POSITIVE);
    (x.to_dtype(DType::F32)? / scale as f64)?.to_dtype(DType::F8E4M3)
}

pub fn rows(bench: &Bench) -> Result<Vec<Row>> {
    let dev = &bench.dev;
    let mut rows = Vec::new();
    let _fp8 = ACT_QUANT_FP8_PATH.guard(Some(true));
    let _int8 = ACT_QUANT_INT8_PATH.guard(Some(true));
    let stock_cast = activation(dev, 4, 16).and_then(|x| stock_device_fp8(&x));
    if let Err(e) = &stock_cast {
        eprintln!("act_quant: candle's stock device fp8 cast is unavailable here: {e}");
    }
    for &(m, k) in QUANT_SHAPES {
        let x = activation(dev, m, k)?;
        let shape = format!("{m}x{k} bf16");
        rows.push(bench.compare(
            "act_quant_fp8 (vs CPU-cast chain)",
            &shape,
            || quantize_activation_fp8_reference(&x).map(drop),
            || quantize_activation_fp8(&x).map(drop),
        )?);
        if stock_cast.is_ok() {
            rows.push(bench.compare(
                "act_quant_fp8 (vs stock device cast, device scale)",
                &shape,
                || stock_device_fp8(&x).map(drop),
                || fused_fp8(&x, Fp8ScaleRule::Reciprocal).map(drop),
            )?);
        }
        rows.push(bench.compare(
            "act_quant_int8 (vs candle chain)",
            &shape,
            || quantize_activation_int8_reference(&x).map(drop),
            || quantize_activation_int8(&x).map(drop),
        )?);
    }

    let lt = Arc::new(CublasLt::new(dev)?);
    if !lt.meets_fp8_floor()? {
        eprintln!("act_quant: the 8-bit linears need sm_89; skipping their rows");
        return Ok(rows);
    }
    for &(m, k, n) in LINEAR_SHAPES {
        let x = activation(dev, m, k)?;
        let w = (Tensor::randn(0f32, 0.02, (n, k), dev)?).to_dtype(DType::BF16)?;
        let shape = format!("{m}x{k}->{n} bf16");
        let fp8 = Fp8Linear::new(&w, None, lt.clone())?;
        rows.push(bench.compare(
            "fp8_linear dynamic (forward)",
            &shape,
            || {
                ACT_QUANT_FP8_PATH.set_enabled(Some(false));
                fp8.forward(&x).map(drop)
            },
            || {
                ACT_QUANT_FP8_PATH.set_enabled(Some(true));
                fp8.forward(&x).map(drop)
            },
        )?);
        let q = quantize_weight_int8_per_channel(&w)?;
        let int8 = Int8Linear::from_per_channel_parts(q.q, q.scale, None, lt.clone())?;
        rows.push(bench.compare(
            "int8_linear per-channel (forward)",
            &shape,
            || {
                ACT_QUANT_INT8_PATH.set_enabled(Some(false));
                int8.forward(&x).map(drop)
            },
            || {
                ACT_QUANT_INT8_PATH.set_enabled(Some(true));
                int8.forward(&x).map(drop)
            },
        )?);
    }
    Ok(rows)
}

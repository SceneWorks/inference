//! Native packed W4A8 projections (sc-26685).
//!
//! Only packed codes and their original companions survive load. CUDA decodes bounded output-row
//! tiles to INT8, reusing one full-row ConvRot activation quantization and the shared S4 integer
//! GEMM/epilogue arithmetic. There is no resident dense INT8 shadow or floating-weight fallback.
//!
//! # The activation rotation is the pinned F32 butterfly, not a dense `x·H` matmul
//!
//! `int8_linear.cu::quantize_int8_rowwise_convrot_kernel` rotates inside the quantizer: the row
//! lands in shared memory as F32, four radix-4 stages (strides `1,4,16,64`, `0.5f` each) build the
//! rotated values in F32, and the row absmax — hence `x_scale` — is taken from those F32 values
//! before `quant_div_float_to_float<T>` rounds the numerator, the divisor and the quotient to the
//! activation dtype. [`super::convrot_fht_f32`] replays exactly that `+`-order in F32; feeding the
//! quantizer a dtype-rounded dense rotation instead moves the peak (and the divisor) by whole
//! dtypes' worth of ulps and changes the int8 codes.
use crate::{CandleError, Result, Weights};
use candle_core::{DType, Device, Tensor};
use gen_core::w4a8_int8::{W4A8Int8, MAX_DESCRIPTOR_BYTES};

/// Output rows decoded at a time; every tile spans the full contraction dimension.
pub const TILE_ROWS: usize = 64;

/// Immutable packed codebook projection with forward-time additive adapter support.
pub struct W4A8Linear {
    pub(crate) descriptor: W4A8Int8,
    device: Device,
    #[cfg(feature = "cuda")]
    packed: Tensor,
    #[cfg(feature = "cuda")]
    s_rel: Tensor,
    #[cfg(feature = "cuda")]
    codebook: Tensor,
    #[cfg(feature = "cuda")]
    s_channel: Tensor,
    #[cfg(feature = "cuda")]
    bias: Option<Tensor>,
    #[cfg(feature = "cuda")]
    lt: std::sync::Arc<super::CublasLt>,
}

impl std::fmt::Debug for W4A8Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("W4A8Linear")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}

impl W4A8Linear {
    pub fn descriptor(&self) -> W4A8Int8 {
        self.descriptor
    }

    /// Packed operand bytes, excluding bias and the shared rotation/workspace.
    pub fn resident_bytes(&self) -> usize {
        self.descriptor.resident_bytes()
    }

    /// Peak transient decoded weight bytes on the compute device (not a resident shadow).
    /// Activation/output buffers are request-shaped and are not included in this weight-only bound.
    pub fn weight_scratch_bytes(&self) -> usize {
        self.descriptor.out_features.min(TILE_ROWS) * self.descriptor.in_features
    }

    /// Detect the authoritative dialect before another loader can interpret the shortened weight
    /// as dense. Candle carries original I8 two's-complement weight bytes as U8, as in S4.
    pub fn from_weights(w: &Weights, base: &str, bias: bool) -> Result<Option<Self>> {
        let companions = ["weight_codebook", "weight_s_rel", "weight_s_channel"];
        let has_companions = companions
            .iter()
            .any(|s| w.contains(&format!("{base}.{s}")));
        let dk = format!("{base}.comfy_quant");
        if !w.contains(&dk) {
            if has_companions {
                return Err(CandleError::Msg(format!(
                    "{base}: W4A8 operands require an authoritative .comfy_quant descriptor"
                )));
            }
            return Ok(None);
        }
        let payload = w.require(&dk)?;
        if payload.dtype() != DType::U8
            || payload.rank() != 1
            || payload.elem_count() > MAX_DESCRIPTOR_BYTES
        {
            return Err(CandleError::Msg(format!(
                "{base}: descriptor must be a bounded U8 vector"
            )));
        }
        let payload = payload.to_device(&Device::Cpu)?.to_vec1::<u8>()?;
        let json: serde_json::Value = serde_json::from_slice(&payload)
            .map_err(|e| CandleError::Msg(format!("{base}: invalid quant descriptor: {e}")))?;
        if json.get("format").and_then(serde_json::Value::as_str) != Some("asym_w4a8_int8") {
            if has_companions {
                return Err(CandleError::Msg(format!(
                    "{base}: W4A8 companions contradict the quant descriptor"
                )));
            }
            return Ok(None);
        }
        for suffix in ["weight_scale", "scales", "biases", "weight_correction"] {
            if w.contains(&format!("{base}.{suffix}")) {
                return Err(CandleError::Msg(format!("{base}: W4A8 forbids .{suffix}")));
            }
        }
        let codes = w.require(&format!("{base}.weight"))?;
        let codebook = w.require(&format!("{base}.weight_codebook"))?;
        let s_rel = w.require(&format!("{base}.weight_s_rel"))?;
        let s_channel = w.require(&format!("{base}.weight_s_channel"))?;
        if codes.dtype() != DType::U8
            || codebook.dtype() != DType::F32
            || s_rel.dtype() != DType::F8E4M3
            || s_channel.dtype() != DType::F32
        {
            return Err(CandleError::Msg(format!(
                "{base}: W4A8 requires original I8 byte patterns, F8_E4M3 s_rel and uncast F32 codebook/s_channel"
            )));
        }
        let descriptor = W4A8Int8::parse(
            &payload,
            codes.dims(),
            s_rel.dims(),
            s_channel.dims(),
            codebook.dims(),
        )
        .map_err(CandleError::Msg)?;
        // Read raw FP8 bits, never a numeric U8 cast (which would destroy the relative scales).
        let rel_cpu = s_rel.to_device(&Device::Cpu)?.flatten_all()?.contiguous()?;
        let s_rel = {
            let (storage, layout) = rel_cpu.storage_and_layout();
            match &*storage {
                candle_core::Storage::Cpu(candle_core::CpuStorage::F8E4M3(values)) => values
                    [layout.start_offset()..layout.start_offset() + rel_cpu.elem_count()]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<u8>>(),
                _ => {
                    return Err(CandleError::Msg(format!(
                        "{base}: expected CPU FP8 storage"
                    )))
                }
            }
        };
        let levels = codebook.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        let channels = s_channel.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        descriptor
            .validate_s_rel(&s_rel)
            .map_err(CandleError::Msg)?;
        descriptor
            .validate_codebook(&levels)
            .map_err(CandleError::Msg)?;
        descriptor
            .validate_s_channel(&channels)
            .map_err(CandleError::Msg)?;
        let bias = if bias {
            Some(w.require(&format!("{base}.bias"))?)
        } else {
            None
        };
        if bias.as_ref().is_some_and(|b| {
            b.dims() != [descriptor.out_features]
                || !matches!(b.dtype(), DType::F32 | DType::F16 | DType::BF16)
        }) {
            return Err(CandleError::Msg(format!("{base}: invalid W4A8 bias")));
        }
        #[cfg(feature = "cuda")]
        {
            let ctx = w.int8_context(codes.device())?;
            let lt = ctx.handle_for(codes.device())?.clone();
            if !lt.meets_fp8_floor()? {
                return Err(CandleError::Unsupported(
                    "W4A8 requires the shared sm_89 integer lane".into(),
                ));
            }
            if !descriptor.out_features.is_multiple_of(16) {
                return Err(CandleError::Msg(
                    "W4A8 CUDA output width must be a multiple of 16".into(),
                ));
            }
            let packed = codes.clone().contiguous()?;
            let s_rel = Tensor::from_vec(
                s_rel,
                (
                    descriptor.out_features,
                    descriptor.in_features / descriptor.group_size,
                ),
                codes.device(),
            )?;
            let codebook = Tensor::from_vec(levels, 16, codes.device())?;
            let s_channel = s_channel.to_device(codes.device())?;
            let bias = bias.map(|b| b.to_device(codes.device())).transpose()?;
            Ok(Some(Self {
                descriptor,
                device: codes.device().clone(),
                packed,
                s_rel,
                codebook,
                s_channel,
                bias,
                lt,
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = bias;
            Err(CandleError::Unsupported(
                "W4A8 requires native CUDA; no floating fallback".into(),
            ))
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            let d = self.descriptor;
            if x.dims().last() != Some(&d.in_features)
                || !matches!(x.dtype(), DType::F32 | DType::BF16 | DType::F16)
                || x.elem_count() == 0
                || !self.device.same_device(x.device())
            {
                candle_core::bail!("W4A8 activation shape/dtype/device mismatch");
            }
            // The pinned fused FHT, in F32 and in kernel butterfly order. NOT
            // `convrot_rotate` (a dense matmul): see the module doc — it re-associates the
            // additions and, in the source dtype, rounds the rotated values before the row
            // absmax, which is what moves `x_scale` and the int8 codes.
            let rotated = super::convrot_fht_f32(x, d.rotation_group)?
                .reshape((x.elem_count() / d.in_features, d.in_features))?;
            // Full-row scale and codes are computed ONCE, outside the weight tile loop. Never
            // quantize per K-chunk: that would change the published activation arithmetic.
            let (qx, xs) = self.lt.quantize_kitchen_int8(&rotated, x.dtype())?;
            let mut outputs = Vec::with_capacity(d.out_features.div_ceil(TILE_ROWS));
            for row in (0..d.out_features).step_by(TILE_ROWS) {
                let rows = (d.out_features - row).min(TILE_ROWS);
                // Device-side weight decode of this bounded output-row tile: the packed codes,
                // F8_E4M3 s_rel and F32 codebook are resident; the int8 tile is a transient scratch
                // that dies with this turn. No host round-trip and no per-forward CPU→GPU upload.
                let staged = self.lt.dequant_w4a8_packed_tile(
                    &self.packed,
                    &self.s_rel,
                    &self.codebook,
                    super::W4A8DecodeTile {
                        rows,
                        in_features: d.in_features,
                        group: d.group_size,
                        out0: row,
                    },
                )?;
                let acc = self
                    .lt
                    .matmul_int8_staged(&staged, &qx)?
                    .to_dtype(DType::F32)?;
                let scales = self.s_channel.narrow(0, row, rows)?.reshape((1, rows))?;
                // As in S4, separate eager FP32 tensor ops prohibit multiply/add fusion. Do not
                // pre-fold xs*s_channel, and round to the output dtype only after adding bias.
                let y = acc.broadcast_mul(&xs)?.broadcast_mul(&scales)?;
                let y = match &self.bias {
                    Some(b) => y.broadcast_add(&b.narrow(0, row, rows)?.to_dtype(DType::F32)?)?,
                    None => y,
                };
                outputs.push(y.to_dtype(x.dtype())?);
            }
            let out = Tensor::cat(&outputs, 1)?;
            let mut shape = x.dims().to_vec();
            *shape.last_mut().unwrap() = d.out_features;
            out.reshape(shape)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = x;
            candle_core::bail!("W4A8 requires native CUDA; no floating fallback")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn operands() -> HashMap<String, Tensor> {
        let device = &Device::Cpu;
        let mut map = HashMap::new();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), device).unwrap(),
        );
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 128), DType::U8, device).unwrap(),
        );
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x38; 256], DType::F8E4M3, &[16, 16], device).unwrap(),
        );
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::ones(16, DType::F32, device).unwrap(),
        );
        map.insert(
            "p.weight_codebook".into(),
            Tensor::from_vec((0..16).map(|v| v as f32 - 8.).collect(), 16, device).unwrap(),
        );
        map
    }

    #[test]
    fn refuses_orphaned_and_contradictory_operands() {
        let mut map = operands();
        map.remove("p.comfy_quant");
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("authoritative"));
        for suffix in ["weight_scale", "scales", "biases", "weight_correction"] {
            let mut map = operands();
            map.insert(
                format!("p.{suffix}"),
                Tensor::ones(16, DType::F32, &Device::Cpu).unwrap(),
            );
            let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
            assert!(err.to_string().contains("forbids"));
        }
    }

    #[test]
    fn refuses_noncanonical_descriptor_and_invalid_values_before_cuda() {
        let mut map = operands();
        let payload = br#"{"convrot_groupsize":256,"format":"asym_w4a8_int8","group_size":16,"convrot":true}"#;
        map.insert(
            "p.comfy_quant".into(),
            Tensor::from_slice(payload, payload.len(), &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("descriptor"));
        let mut map = operands();
        map.insert(
            "p.weight_s_channel".into(),
            Tensor::zeros(16, DType::F32, &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("finite positive F32"));
        let mut map = operands();
        map.insert(
            "p.weight_s_rel".into(),
            Tensor::from_raw_buffer(&[0x7f; 256], DType::F8E4M3, &[16, 16], &Device::Cpu).unwrap(),
        );
        let err = W4A8Linear::from_weights(&Weights::from_map(map), "p", false).unwrap_err();
        assert!(err.to_string().contains("non-negative finite E4M3FN"));
    }

    #[test]
    fn dense_projection_is_not_claimed_as_w4a8() {
        let mut map = HashMap::new();
        map.insert(
            "p.weight".into(),
            Tensor::zeros((16, 256), DType::F32, &Device::Cpu).unwrap(),
        );
        assert!(
            W4A8Linear::from_weights(&Weights::from_map(map), "p", false)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn validated_w4a8_refuses_a_floating_fallback() {
        let err = W4A8Linear::from_weights(&Weights::from_map(operands()), "p", false).unwrap_err();
        assert!(err
            .to_string()
            .contains("requires native CUDA; no floating fallback"));
    }
}

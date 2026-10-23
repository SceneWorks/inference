//! Native descriptor-driven ConvRot INT8 base (sc-26683), shared with adapter consumers.
use crate::{CandleError, Result, Weights};
use candle_core::{DType, Device, Tensor};
use gen_core::convrot_int8::{ConvRotInt8, MAX_DESCRIPTOR_BYTES};

/// One compact signed INT8 projection. Non-CUDA builds refuse, never dequantize.
pub struct ConvRotLinear {
    pub(crate) descriptor: ConvRotInt8,
    device: Device,
    #[cfg(feature = "cuda")]
    rotation: Tensor,
    #[cfg(feature = "cuda")]
    linear: super::Int8Linear,
}
impl std::fmt::Debug for ConvRotLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvRotLinear")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}
impl ConvRotLinear {
    /// Immutable validated logical geometry.
    pub fn descriptor(&self) -> ConvRotInt8 {
        self.descriptor
    }

    /// Missing descriptor is dense only when there are no native INT8 operands.
    pub fn from_weights(w: &Weights, base: &str, bias: bool) -> Result<Option<Self>> {
        let codes = w.require(&format!("{base}.weight"))?;
        let dk = format!("{base}.comfy_quant");
        let sk = format!("{base}.weight_scale");
        if !w.contains(&dk) {
            if w.contains(&sk) || codes.dtype() == DType::U8 || codes.dtype() == DType::I64 {
                return Err(CandleError::Msg(format!(
                    "{base}: INT8 requires an authoritative .comfy_quant descriptor"
                )));
            }
            return Ok(None);
        }
        if w.contains(&format!("{base}.scales")) || w.contains(&format!("{base}.biases")) {
            return Err(CandleError::Msg(format!(
                "{base}: contradictory affine and ConvRot operands"
            )));
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
        let scales = w.require(&sk)?;
        if codes.dtype() != DType::U8 || scales.dtype() != DType::F32 {
            return Err(CandleError::Msg(format!(
                "{base}: expected exact I8 byte patterns and uncast F32 scales"
            )));
        }
        let descriptor = ConvRotInt8::parse(
            &payload.to_device(&Device::Cpu)?.to_vec1::<u8>()?,
            codes.dims(),
            scales.dims(),
        )
        .map_err(CandleError::Msg)?;
        descriptor
            .validate_scales(
                &scales
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            )
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
            return Err(CandleError::Msg(format!("{base}: invalid ConvRot bias")));
        }
        #[cfg(feature = "cuda")]
        {
            let ctx = w.int8_context(codes.device())?;
            let lt = ctx.handle_for(codes.device())?;
            if !lt.meets_fp8_floor()? {
                return Err(CandleError::Unsupported(
                    "ConvRot INT8 requires the shared sm_89 integer lane".into(),
                ));
            }
            if !descriptor.out_features.is_multiple_of(16) {
                return Err(CandleError::Msg(
                    "ConvRot CUDA output width must be a multiple of16".into(),
                ));
            }
            let linear = super::Int8Linear::from_native_bytes(&codes, scales, bias, lt.clone())?;
            let rotation = w.convrot_rotation(descriptor.rotation_group, codes.device())?;
            Ok(Some(Self {
                descriptor,
                device: codes.device().clone(),
                rotation,
                linear,
            }))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (descriptor, bias);
            Err(CandleError::Unsupported(
                "ConvRot INT8 requires native CUDA; no floating fallback".into(),
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
                candle_core::bail!("ConvRot INT8 activation shape/dtype/device mismatch");
            }
            let rotated = super::convrot_rotate(x, &self.rotation)?
                .to_dtype(x.dtype())?
                .reshape((x.elem_count() / d.in_features, d.in_features))?;
            let out = self.linear.forward_kitchen(&rotated, x.dtype())?;
            let mut shape = x.dims().to_vec();
            *shape.last_mut().unwrap() = d.out_features;
            out.reshape(shape)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = x;
            candle_core::bail!("ConvRot INT8 requires native CUDA; no floating fallback")
        }
    }
}

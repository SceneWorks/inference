//! Native ConvRot projections from a media weight map.
use crate::{CandleError, Result, Weights};
use candle_core::{DType, Device, Tensor};
use gen_core::convrot_int8::{ConvRotInt8, MAX_DESCRIPTOR_BYTES};

#[derive(Debug)]
pub struct ConvRotLinear(candle_quant_kernels::convrot_int8::ConvRotLinear);

impl std::ops::Deref for ConvRotLinear {
    type Target = candle_quant_kernels::convrot_int8::ConvRotLinear;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ConvRotLinear {
    pub fn from_native(base: candle_quant_kernels::convrot_int8::ConvRotLinear) -> Self {
        Self(base)
    }

    pub(crate) fn from_native_parts(
        resources: &crate::weights::ProjectionResources,
        descriptor: ConvRotInt8,
        codes: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        if !codes.device().is_cuda() {
            return Err(CandleError::Unsupported(
                "ConvRot INT8 requires native CUDA".into(),
            ));
        }
        Ok(Self(
            candle_quant_kernels::convrot_int8::ConvRotLinear::from_native_parts(
                resources, descriptor, codes, scales, bias,
            )?,
        ))
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
        Self::from_native_parts(
            &w.projection_resources,
            descriptor,
            codes.clone(),
            scales.clone(),
            bias,
        )
        .map(Some)
    }
}

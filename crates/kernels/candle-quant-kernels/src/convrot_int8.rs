//! Native descriptor-driven ConvRot INT8 base (sc-26683), shared with adapter consumers.
use candle_core::Result;
use candle_core::{DType, Device, Tensor};
use gen_core::convrot_int8::ConvRotInt8;
use std::collections::HashMap;

/// One compact signed INT8 projection. Non-CUDA builds refuse, never dequantize.
pub struct ConvRotLinear {
    descriptor: ConvRotInt8,
    resident_bytes: u64,
    device: Device,
    #[cfg(feature = "cuda")]
    rotation: Tensor,
    #[cfg(feature = "cuda")]
    linear: crate::eight_bit_linear::Int8Linear,
}
impl std::fmt::Debug for ConvRotLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvRotLinear")
            .field("descriptor", &self.descriptor)
            .finish_non_exhaustive()
    }
}
impl ConvRotLinear {
    /// Bytes held by this projection's code, scale and bias tensors. The rotation belongs to
    /// the model-wide resource cache and is shared by every projection of the same group size.
    pub fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Immutable validated logical geometry.
    pub fn descriptor(&self) -> ConvRotInt8 {
        self.descriptor
    }

    /// Construct from original signed byte patterns and validated native F32 row scales.
    pub fn from_native_parts(
        resources: &ProjectionResources,
        descriptor: ConvRotInt8,
        codes: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        if codes.dtype() != DType::U8
            || codes.dims() != [descriptor.out_features, descriptor.in_features]
            || scales.dtype() != DType::F32
            || scales.dims() != [descriptor.out_features, 1]
            || !codes.device().same_device(scales.device())
        {
            return Err(candle_core::Error::Msg(
                "ConvRot code or scale layout/device mismatch".into(),
            ));
        }
        if bias.as_ref().is_some_and(|b| {
            !b.device().same_device(codes.device())
                || b.dims() != [descriptor.out_features]
                || !matches!(b.dtype(), DType::F32 | DType::F16 | DType::BF16)
        }) {
            return Err(candle_core::Error::Msg("invalid ConvRot bias".into()));
        }
        #[cfg(feature = "cuda")]
        {
            let ctx = resources.int8_context(codes.device())?;
            let lt = ctx.handle_for(codes.device())?;
            if !lt.meets_fp8_floor()? {
                return Err(candle_core::Error::Msg(
                    "ConvRot INT8 requires the shared sm_89 integer lane".into(),
                ));
            }
            if !descriptor.out_features.is_multiple_of(16) {
                return Err(candle_core::Error::Msg(
                    "ConvRot CUDA output width must be a multiple of 16".into(),
                ));
            }
            let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
            let resident_bytes =
                (bytes(&codes) + bytes(&scales) + bias.as_ref().map_or(0, bytes)) as u64;
            let linear = crate::eight_bit_linear::Int8Linear::from_native_bytes(
                &codes,
                scales,
                bias,
                lt.clone(),
            )?;
            let rotation = resources.convrot_rotation(descriptor.rotation_group, codes.device())?;
            Ok(Self {
                descriptor,
                resident_bytes,
                device: codes.device().clone(),
                rotation,
                linear,
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (descriptor, bias, resources);
            Err(candle_core::Error::Msg(
                "ConvRot INT8 requires native CUDA; no floating fallback".into(),
            ))
        }
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_inner(x, None)
    }

    /// Add a caller-owned bias in the same FP32 epilogue as a checkpoint-owned bias.
    pub fn forward_with_bias(&self, x: &Tensor, bias: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, Some(bias))
    }

    fn forward_inner(&self, x: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
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
            let rotated = crate::convrot::convrot_rotate(x, &self.rotation)?
                .to_dtype(x.dtype())?
                .reshape((x.elem_count() / d.in_features, d.in_features))?;
            let out = match bias {
                Some(b) => self
                    .linear
                    .forward_kitchen_with_bias(&rotated, x.dtype(), Some(b))?,
                None => self.linear.forward_kitchen(&rotated, x.dtype())?,
            };
            let mut shape = x.dims().to_vec();
            *shape.last_mut().unwrap() = d.out_features;
            out.reshape(shape)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (x, bias);
            candle_core::bail!("ConvRot INT8 requires native CUDA; no floating fallback")
        }
    }
}

/// Model-scoped resources for native projections on a single device. Shared by eager maps
/// and lazy logical readers across layers and successive block windows.
#[derive(Default)]
pub struct ProjectionResources {
    int8: std::sync::Mutex<Option<crate::Int8Context>>,
    convrot_rotations: std::sync::Mutex<HashMap<usize, Tensor>>,
}

impl ProjectionResources {
    pub fn set_int8_context(&self, context: crate::Int8Context) {
        *self.int8.lock().unwrap_or_else(|e| e.into_inner()) = Some(context);
    }
    /// One lazy cuBLASLt handle/workspace for every native INT8 projection in this map.
    pub fn int8_context(&self, device: &Device) -> Result<crate::Int8Context> {
        let mut cached = self.int8.lock().unwrap_or_else(|p| p.into_inner());
        if cached.is_none() {
            *cached = Some(crate::Int8Context::new(device)?);
        }
        Ok(cached.as_ref().expect("initialized above").clone())
    }

    /// Share one immutable rotation per group on this map's device (no per-layer matrix copies).
    pub fn convrot_rotation(&self, group: usize, device: &Device) -> Result<Tensor> {
        let mut cache = self
            .convrot_rotations
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(r) = cache.get(&group) {
            if !r.device().same_device(device) {
                return Err(candle_core::Error::Msg(
                    "ConvRot map contains multiple devices".into(),
                ));
            }
            return Ok(r.clone());
        }
        let r = crate::convrot::regular_hadamard(group, device)?;
        cache.insert(group, r.clone());
        Ok(r)
    }
}

//! Tensor codec specs and residency policies.

use super::*;

/// Where a scalar fp8 codec's per-tensor scale comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarScaleSource {
    /// `{layer}.weight_scale` (descriptor-gated ComfyUI convention).
    Companion { physical_key: String },
    /// No descriptor and no companion: the plain `weight_dtype=fp8_*` cast. The reference decode is
    /// `to(compute_dtype)`, i.e. the same scalar math at exactly 1.0 — explicit here so the plan
    /// records the convention rather than defaulting silently.
    Unit,
}

/// How one planned tensor is decoded — the per-layer half of the codec dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorCodecSpec {
    /// Byte-preserving dense pass-through.
    Dense,
    /// Scalar-scaled fp8 (E4M3FN or E5M2 per the plan entry's encoding).
    ScalarFp8 {
        scale: ScalarScaleSource,
        /// Consumed activation-scale companion, irrelevant to weight decode.
        input_scale: Option<String>,
        full_precision_matrix_mult: bool,
    },
    /// MXFP8: stored 32×32-padded E4M3 values + swizzled E8M0 block scales.
    Mxfp8 {
        scale: String,
        stored_shape: [usize; 2],
        /// Whether the logical shape came from the adapter (`true`) or is the stored shape because
        /// the adapter declares none (`false`; the family's shape validation is the backstop).
        logical_shape_declared: bool,
        full_precision_matrix_mult: bool,
    },
    /// Int8 per-output-row codes + `F32` row scales.
    Int8PerRow {
        scale: String,
        full_precision_matrix_mult: bool,
    },
    /// NVFP4: `U8` `[rows, cols/2]` E2M1 nibbles + `to_blocked` E4M3 block scales + one `F32`
    /// per-tensor scale.
    Nvfp4 {
        /// `{layer}.weight_scale` — the swizzled per-16-block E4M3 micro-scales.
        block_scale: String,
        /// `{layer}.weight_scale_2` — the per-tensor `F32` second level.
        global_scale: String,
        /// Consumed activation-scale companion, irrelevant to weight decode.
        input_scale: Option<String>,
        /// The 16-padded **logical** `[rows, cols]` the payload holds (the on-disk byte shape is
        /// `[rows, cols / 2]`).
        stored_shape: [usize; 2],
        /// The layer's true `[out_features, in_features]`. ComfyUI pads NVFP4 storage to 16, so
        /// this is element-wise `≤ stored_shape`; the two are equal only when the layer's own
        /// geometry needed no padding. A packed-native plan **requires** equality — repacking the
        /// padded grid would hand `Nvfp4Linear` padding as real contraction elements (sc-20641).
        logical_shape: [usize; 2],
        /// Whether the logical shape came from the adapter (`true`) or is the stored shape because
        /// the adapter declares none (`false`; the family's shape validation is the backstop).
        logical_shape_declared: bool,
        full_precision_matrix_mult: bool,
    },
}

impl TensorCodecSpec {
    pub fn full_precision_matrix_mult(&self) -> bool {
        match self {
            Self::Dense => false,
            Self::ScalarFp8 {
                full_precision_matrix_mult,
                ..
            }
            | Self::Mxfp8 {
                full_precision_matrix_mult,
                ..
            }
            | Self::Int8PerRow {
                full_precision_matrix_mult,
                ..
            }
            | Self::Nvfp4 {
                full_precision_matrix_mult,
                ..
            } => *full_precision_matrix_mult,
        }
    }
}

/// Whether a planned tensor stays packed (its stored bytes + scale companions remain resident) or
/// decodes to the codec's dense resident encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyMode {
    Dense,
    Packed,
}

/// The residency the plan prices for one tensor. `resident_bytes` covers the weight tensor itself;
/// retained companions price on their own [`CompanionTensorPlan`] rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedResidency {
    pub mode: ResidencyMode,
    pub resident_bytes: u64,
}

/// The backend's plan-time decision: keep a codec's stored packing resident (native path) or take
/// the dense fallback. Layout facts are in the spec; hardware facts are the policy's own (device
/// generation, feature gates). A layer whose descriptor sets `full_precision_matrix_mult` never
/// runs a packed matmul, so a policy must answer [`ResidencyMode::Dense`] for it.
pub trait CodecResidencyPolicy {
    fn residency(
        &self,
        codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        stored_shape: &[usize],
    ) -> ResidencyMode;
}

/// The always-dense policy: every codec decodes to its dense resident encoding. MLX (no packed
/// fp8/int8 matmul on the seam) and any backend without the matching hardware use this.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseResidencyPolicy;

impl CodecResidencyPolicy for DenseResidencyPolicy {
    fn residency(
        &self,
        _codec: &CheckpointCodecRegistration,
        _spec: &TensorCodecSpec,
        _stored_shape: &[usize],
    ) -> ResidencyMode {
        ResidencyMode::Dense
    }
}

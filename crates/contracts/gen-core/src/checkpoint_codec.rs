//! Backend-neutral checkpoint **codec registry** and **mapped logical-weight plan** (epic 20398,
//! sc-20634; descriptor-gated codecs sc-20385).
//!
//! Two tensor-free halves that every backend reader consumes the same way:
//!
//! 1. A [`CheckpointCodecRegistration`] declares how one **stored tensor format** — element
//!    encoding plus the ComfyUI `.comfy_quant` descriptor format governing it, if any — becomes
//!    resident weights. Codecs register **once**, at engine/catalog level; a family adapter never
//!    carries its own codec table (that was the duplicate-row defect of the withdrawn sc-20638
//!    draft), so adding a codec never touches an adapter and adding an adapter never re-registers a
//!    codec.
//! 2. A [`LogicalWeightPlan`] is compiled from a checkpoint's **header**
//!    ([`SafetensorsTensorHeader`]) plus the raw bytes of its (small) `.comfy_quant` descriptor
//!    tensors, the family adapter's [`LogicalKeyMapping`], the codec registry, and the backend's
//!    [`CodecResidencyPolicy`]. It fails closed before any backend array exists: an unmapped
//!    physical key, two physical keys mapping onto one logical key, a stored format with no
//!    registered codec, a malformed descriptor, a missing/mis-shaped scale companion, or bad MXFP8
//!    block geometry are typed [`LogicalWeightPlanError`]s naming the exact tensor. The backend
//!    reader then materializes exactly the planned tensors and returns a [`LogicalWeightReceipt`]
//!    whose resident bytes are measured from the decoded arrays, never copied from the header.
//!
//! Mixed checkpoints dispatch **per layer**: each planned tensor carries its own
//! [`TensorCodecSpec`] (dense pass-through next to scalar-fp8 next to MXFP8 in one file), and each
//! carries its own [`PlannedResidency`] — packed-native where the backend's policy keeps the stored
//! packing resident, dense-fallback bytes otherwise — so admission prices the two differently.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::comfy_quant::{
    parse_comfy_quant_descriptor, validate_mxfp8_geometry, validate_nvfp4_geometry,
    ComfyQuantDescriptor, ComfyQuantDescriptorError, ComfyQuantFormat, Mxfp8GeometryError,
    Nvfp4GeometryError, PartialComfyQuantDescriptor, QuantizationMetadataError,
};
use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};

/// How a tensor's elements are stored on disk (or, for
/// [`CheckpointCodecRegistration::resident_encoding`], how a codec leaves them resident).
/// Classified from the safetensors header dtype. The *format* of a tensor is this encoding plus
/// its optional descriptor — see [`StoredTensorFormat`].
mod encoding;
pub use encoding::*;

mod transform;
pub use transform::*;

mod spec;
pub use spec::*;

mod plan;
pub use plan::*;

mod compile;
pub use compile::*;

mod receipt;
pub use receipt::*;

#[cfg(test)]
mod tests;

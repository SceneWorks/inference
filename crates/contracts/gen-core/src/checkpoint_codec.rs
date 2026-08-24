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
    Nvfp4GeometryError, QuantizationMetadataError,
};
use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};

/// How a tensor's elements are stored on disk (or, for
/// [`CheckpointCodecRegistration::resident_encoding`], how a codec leaves them resident).
/// Classified from the safetensors header dtype. The *format* of a tensor is this encoding plus
/// its optional descriptor — see [`StoredTensorFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightEncoding {
    DenseBf16,
    DenseF16,
    DenseF32,
    DenseF64,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    UInt8,
    Int16,
    Int32,
    Int64,
    UInt16,
    UInt32,
    UInt64,
    Bool,
}

impl WeightEncoding {
    /// Classify one stored safetensors dtype. `F8_E8M0` deliberately maps to `None`: a shared
    /// exponent is only ever a *companion* (MXFP8 block scales), never a weight element type, so a
    /// checkpoint storing weights as `F8_E8M0` is unclassifiable rather than mis-decoded.
    pub fn from_dtype(dtype: Dtype) -> Option<Self> {
        Some(match dtype {
            Dtype::BF16 => Self::DenseBf16,
            Dtype::F16 => Self::DenseF16,
            Dtype::F32 => Self::DenseF32,
            Dtype::F64 => Self::DenseF64,
            Dtype::F8_E4M3 => Self::Fp8E4M3,
            Dtype::F8_E5M2 => Self::Fp8E5M2,
            Dtype::I8 => Self::Int8,
            Dtype::U8 => Self::UInt8,
            Dtype::I16 => Self::Int16,
            Dtype::I32 => Self::Int32,
            Dtype::I64 => Self::Int64,
            Dtype::U16 => Self::UInt16,
            Dtype::U32 => Self::UInt32,
            Dtype::U64 => Self::UInt64,
            Dtype::BOOL => Self::Bool,
            _ => return None,
        })
    }

    /// Bytes one element occupies when resident in this encoding.
    pub fn element_bytes(self) -> u64 {
        match self {
            Self::Bool | Self::Int8 | Self::UInt8 | Self::Fp8E4M3 | Self::Fp8E5M2 => 1,
            Self::DenseBf16 | Self::DenseF16 | Self::Int16 | Self::UInt16 => 2,
            Self::DenseF32 | Self::Int32 | Self::UInt32 => 4,
            Self::DenseF64 | Self::Int64 | Self::UInt64 => 8,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::DenseBf16 => "dense-bf16",
            Self::DenseF16 => "dense-f16",
            Self::DenseF32 => "dense-f32",
            Self::DenseF64 => "dense-f64",
            Self::Fp8E4M3 => "fp8-e4m3",
            Self::Fp8E5M2 => "fp8-e5m2",
            Self::Int8 => "int8",
            Self::UInt8 => "uint8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Bool => "bool",
        }
    }
}

/// One stored tensor format a codec can claim: the element encoding plus the `.comfy_quant`
/// descriptor format governing the tensor (`None` = an undescribed tensor). Two codecs share the
/// `Fp8E4M3` encoding today — the scalar-scaled row and MXFP8 — separated exactly by this
/// descriptor half.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StoredTensorFormat {
    pub encoding: WeightEncoding,
    pub descriptor: Option<ComfyQuantFormat>,
}

impl StoredTensorFormat {
    pub const fn undescribed(encoding: WeightEncoding) -> Self {
        Self {
            encoding,
            descriptor: None,
        }
    }

    pub const fn described(encoding: WeightEncoding, descriptor: ComfyQuantFormat) -> Self {
        Self {
            encoding,
            descriptor: Some(descriptor),
        }
    }
}

impl fmt::Display for StoredTensorFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.descriptor {
            None => f.write_str(self.encoding.label()),
            Some(descriptor) => {
                write!(f, "{} (comfy_quant {})", self.encoding.label(), descriptor)
            }
        }
    }
}

/// One registered codec: the stored formats it claims and the encoding its dense fallback leaves
/// resident.
///
/// The portable row is the declaration; the backend engine that registers it owns the matching
/// implementation, and the engine's conformance test proves every registered row has one. Whether a
/// backend keeps the stored packing resident instead of the dense fallback is not declared here —
/// it is a per-layer, per-hardware decision the backend's [`CodecResidencyPolicy`] makes at plan
/// time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCodecRegistration {
    pub codec_id: &'static str,
    /// The stored formats this codec decodes (≥ 1; each may be claimed by only one codec per
    /// registry).
    pub stored: &'static [StoredTensorFormat],
    /// What the dense fallback leaves resident.
    pub resident_encoding: WeightEncoding,
}

/// The baseline dense bf16 codec: stored bf16 stays bf16, byte-for-byte (no cast, no remap).
pub const DENSE_BF16_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-bf16-v1",
    stored: &[StoredTensorFormat::undescribed(WeightEncoding::DenseBf16)],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// Dense f16 pass-through (stored f16 stays f16).
pub const DENSE_F16_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-f16-v1",
    stored: &[StoredTensorFormat::undescribed(WeightEncoding::DenseF16)],
    resident_encoding: WeightEncoding::DenseF16,
};

/// Dense f32 pass-through (stored f32 stays f32). A "dense bf16" community DiT routinely keeps a
/// handful of embedding/projection biases and norms in f32 (e.g. `kreamania_variant5`: 415 bf16 +
/// 15 f32 tensors); those tensors decode through this row and report their own resident bytes
/// rather than being cast or silently folded into the bf16 report.
pub const DENSE_F32_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-f32-v1",
    stored: &[StoredTensorFormat::undescribed(WeightEncoding::DenseF32)],
    resident_encoding: WeightEncoding::DenseF32,
};

/// The portable dense pass-through codec rows every backend can implement without a kernel.
pub const DENSE_CODECS: &[CheckpointCodecRegistration] =
    &[DENSE_BF16_CODEC, DENSE_F16_CODEC, DENSE_F32_CODEC];

/// ComfyUI scalar-scaled FP8 **E4M3FN** (sc-20385): an `F8_E4M3` weight with a per-tensor `F32`
/// `weight_scale` under a `float8_e4m3fn` descriptor — or, undescribed, the plain
/// `weight_dtype=fp8_e4m3fn` cast ComfyUI's UNETLoader/ModelSave writes, whose reference decode is
/// the same math at unit scale. Dense fallback decodes `e4m3(v)·scale` to bf16; Candle's cuBLASLt
/// fp8 leg is the packed-native path where its layout + sm_89 hardware contract match.
pub const FP8_E4M3_SCALAR_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "fp8-e4m3-scalar-v1",
    stored: &[
        StoredTensorFormat::described(WeightEncoding::Fp8E4M3, ComfyQuantFormat::Float8E4M3Fn),
        StoredTensorFormat::undescribed(WeightEncoding::Fp8E4M3),
    ],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// ComfyUI scalar-scaled FP8 **E5M2** (sc-20385). Same convention as
/// [`FP8_E4M3_SCALAR_CODEC`] with E5M2 element decode; no backend has a native E5M2-weight matmul,
/// so residency is always the dense fallback.
pub const FP8_E5M2_SCALAR_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "fp8-e5m2-scalar-v1",
    stored: &[
        StoredTensorFormat::described(WeightEncoding::Fp8E5M2, ComfyQuantFormat::Float8E5M2),
        StoredTensorFormat::undescribed(WeightEncoding::Fp8E5M2),
    ],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// ComfyUI **MXFP8** (sc-20385): E4M3FN values padded to 32×32 with E8M0 shared exponents per
/// 32-element block along the last axis, block scales stored in the cuBLAS 128×4 swizzled layout
/// (`mxfp8` descriptor). Dense fallback dequantizes and **unpads** to the logical shape; no
/// compatible block-scaled kernel ships in this workspace, so residency is always dense.
pub const MXFP8_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "mxfp8-v1",
    stored: &[StoredTensorFormat::described(
        WeightEncoding::Fp8E4M3,
        ComfyQuantFormat::Mxfp8,
    )],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// ComfyUI plain **int8 per-row** (sc-14023, moved onto the registry by sc-20385): `I8` codes with
/// an `F32` per-output-row `weight_scale` under an `int8_tensorwise`/`per_row: true` descriptor.
/// Dense fallback decodes `code·scale[row]` to bf16 — the exact math of the former bespoke MLX
/// loader arm.
pub const INT8_PER_ROW_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "int8-per-row-v1",
    stored: &[StoredTensorFormat::described(
        WeightEncoding::Int8,
        ComfyQuantFormat::Int8TensorwisePerRow,
    )],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// ComfyUI **NVFP4** (sc-20641): E2M1 nibbles packed two per `U8` byte (even element in the high
/// nibble), one FP8 **E4M3** micro-scale per 16-element block in the cuBLAS 128×4 `to_blocked`
/// swizzle (`{layer}.weight_scale`), and one `F32` per-tensor scale (`{layer}.weight_scale_2`) —
/// the two-level NVFP4 scaling. Storage is 16-padded on both axes.
///
/// Dense fallback dequantizes both levels and unpads to the logical shape. Candle's `Nvfp4Linear`
/// (cuBLASLt `CUDA_R_4F_E2M1` + `VEC16_UE4M3`) is the packed-native path where the layout and the
/// `sm_120` hardware contract both hold; every other device takes the dense fallback.
pub const NVFP4_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "nvfp4-v1",
    stored: &[StoredTensorFormat::described(
        WeightEncoding::UInt8,
        ComfyQuantFormat::Nvfp4,
    )],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// The ComfyUI descriptor-gated codec rows (sc-20385, sc-20641), registered alongside
/// [`DENSE_CODECS`].
pub const COMFY_QUANT_CODECS: &[CheckpointCodecRegistration] = &[
    FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC,
    MXFP8_CODEC,
    INT8_PER_ROW_CODEC,
    NVFP4_CODEC,
];

/// Validated, immutable set of codecs keyed by stored format. Built by
/// [`CheckpointCodecRegistry::new`] (and by `ProviderRegistryBuilder::build` from the rows a catalog
/// registers); duplicate ids or two codecs claiming one stored format fail closed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointCodecRegistry {
    rows: Vec<CheckpointCodecRegistration>,
    by_format: BTreeMap<StoredTensorFormat, usize>,
}

impl CheckpointCodecRegistry {
    pub fn new(
        codecs: impl IntoIterator<Item = CheckpointCodecRegistration>,
    ) -> crate::Result<Self> {
        let mut ids = BTreeSet::new();
        let mut rows = Vec::new();
        let mut by_format = BTreeMap::new();
        for codec in codecs {
            if codec.codec_id.is_empty()
                || codec
                    .codec_id
                    .chars()
                    .any(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'))
            {
                return Err(crate::Error::Msg(format!(
                    "checkpoint codec id {:?} must be a non-empty lowercase kebab-case identifier",
                    codec.codec_id
                )));
            }
            if !ids.insert(codec.codec_id) {
                return Err(crate::Error::Msg(format!(
                    "duplicate checkpoint codec id '{}'",
                    codec.codec_id
                )));
            }
            if codec.stored.is_empty() {
                return Err(crate::Error::Msg(format!(
                    "checkpoint codec '{}' claims no stored format",
                    codec.codec_id
                )));
            }
            let row = rows.len();
            for format in codec.stored {
                if let Some(previous) = by_format.insert(*format, row) {
                    let previous: &CheckpointCodecRegistration = &rows[previous];
                    return Err(crate::Error::Msg(format!(
                        "checkpoint codecs '{}' and '{}' both claim stored format {}",
                        previous.codec_id, codec.codec_id, format
                    )));
                }
            }
            rows.push(codec);
        }
        Ok(Self { rows, by_format })
    }

    pub fn codecs(&self) -> impl ExactSizeIterator<Item = &CheckpointCodecRegistration> {
        self.rows.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// The codec registered for a stored format, or `None` (the caller refuses).
    pub fn for_format(&self, format: StoredTensorFormat) -> Option<&CheckpointCodecRegistration> {
        self.by_format.get(&format).map(|row| &self.rows[*row])
    }

    /// The codec registered for an **undescribed** stored encoding, or `None`.
    pub fn for_encoding(&self, encoding: WeightEncoding) -> Option<&CheckpointCodecRegistration> {
        self.for_format(StoredTensorFormat::undescribed(encoding))
    }

    pub fn by_id(&self, codec_id: &str) -> Option<&CheckpointCodecRegistration> {
        self.rows.iter().find(|codec| codec.codec_id == codec_id)
    }
}

/// The family adapter's physical-key → canonical-logical-key authority for one dialect.
///
/// `mapping_id` must equal the adapter's registered
/// [`CheckpointCanonicalMappingRegistration::mapping_id`](crate::registry::CheckpointCanonicalMappingRegistration)
/// for that dialect; the provider crate's conformance test proves the registry row is backed by a
/// real implementation.
pub trait LogicalKeyMapping {
    fn mapping_id(&self) -> &'static str;
    /// The canonical logical key for one on-disk key, or `None` when the key is foreign to this
    /// dialect (the plan compiler refuses; nothing is skipped silently).
    fn logical_key(&self, physical_key: &str) -> Option<String>;
    /// The architecture's expected logical shape for one logical key, when the adapter declares
    /// shapes. MXFP8 storage is 32-padded on both axes and the file does not record the true shape,
    /// so a declared shape here lets the plan unpad exactly; without one the plan uses the stored
    /// (padded) shape and the family's own shape validation is the backstop.
    fn logical_shape(&self, _logical_key: &str) -> Option<Vec<usize>> {
        None
    }
}

/// The identity mapping for checkpoints already stored under canonical keys.
///
/// # Not for undescribed-fp8 imports
///
/// This mapping accepts **every** on-disk key as a logical weight. That is safe for a checkpoint
/// whose scale companions follow the `{layer}.weight_scale` / `{layer}.input_scale` /
/// `{layer}.comfy_quant` suffixes this contract recognises — those are claimed as companions before
/// the mapping is consulted. It is **not** safe for an fp8 checkpoint carrying an *unknown*
/// scale convention (a different suffix, an inline `scale_weight`, a sidecar naming scheme): the
/// compiler would not recognise the scale tensor as a companion, `logical_key` would accept it as
/// an ordinary weight, and — because an undescribed fp8 tensor plans as
/// [`FP8_E4M3_SCALAR_CODEC`] at [`ScalarScaleSource::Unit`] — the layer's real weights would
/// decode at **unit scale**, silently wrong rather than refused.
///
/// So a provider importing undescribed fp8 must supply an adapter-owned [`LogicalKeyMapping`] that
/// knows its family's key surface (and refuses keys it does not recognise), not this one. The
/// orphan-companion check catches only the suffixes named above; an unknown convention's scale
/// tensor is invisible to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IdentityKeyMapping;

impl IdentityKeyMapping {
    pub const MAPPING_ID: &'static str = "identity-v1";
}

impl LogicalKeyMapping for IdentityKeyMapping {
    fn mapping_id(&self) -> &'static str {
        Self::MAPPING_ID
    }

    fn logical_key(&self, physical_key: &str) -> Option<String> {
        Some(physical_key.to_owned())
    }
}

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

/// One planned tensor: where it lives on disk, what it is called logically, how it is decoded, and
/// what it costs resident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTensorPlan {
    pub logical_key: String,
    pub physical_key: String,
    pub encoding: WeightEncoding,
    /// The **logical** shape the decoded tensor has (for MXFP8 this is the unpadded shape; the
    /// stored padded shape is in the codec spec).
    pub shape: Vec<usize>,
    /// Bytes the weight tensor occupies in the source file (companions not included).
    pub source_bytes: u64,
    pub codec_id: &'static str,
    /// What this tensor's codec row leaves resident after its **dense fallback** — copied from the
    /// registered [`CheckpointCodecRegistration::resident_encoding`] at compile time, when the
    /// registry row is in hand. Pricing consumers read it here rather than re-finding the row by
    /// id, so a codec registered outside the built-in const tables prices from its own declaration
    /// instead of silently falling back to the tensor's *stored* encoding.
    pub resident_encoding: WeightEncoding,
    pub codec: TensorCodecSpec,
    pub residency: PlannedResidency,
}

/// What a companion tensor is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompanionRole {
    /// `{layer}.comfy_quant` — consumed at plan time, never resident.
    Descriptor,
    /// `{layer}.weight_scale` — consumed by a dense decode; retained by a packed-native load.
    WeightScale,
    /// `{layer}.input_scale` — activation scale; consumed (dense) or retained (packed).
    InputScale,
    /// `{layer}.weight_scale_2` — NVFP4's `F32` per-tensor second-level scale; consumed by a dense
    /// decode, retained by a packed-native load (cuBLASLt folds it into `alpha`).
    GlobalScale,
}

impl CompanionRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Descriptor => "comfy_quant",
            Self::WeightScale => "weight_scale",
            Self::InputScale => "input_scale",
            Self::GlobalScale => "weight_scale_2",
        }
    }
}

/// One companion tensor the plan consumes: it exists on disk (and counts toward
/// [`LogicalWeightPlan::source_bytes`]) but is not a logical weight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompanionTensorPlan {
    pub physical_key: String,
    pub role: CompanionRole,
    /// The `{layer}.weight` tensor this companion belongs to.
    pub owner_physical_key: String,
    pub source_bytes: u64,
    /// Bytes this companion keeps resident: zero when the decode consumes it, its stored bytes when
    /// a packed-native load retains it.
    pub resident_bytes: u64,
}

/// The complete mapped read plan for one safetensors file, sorted by logical key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWeightPlan {
    pub mapping_id: &'static str,
    pub tensors: Vec<LogicalTensorPlan>,
    /// Companion tensors (descriptors/scales), sorted by physical key.
    pub companions: Vec<CompanionTensorPlan>,
    /// Every byte of the file's data region: weights plus companions.
    pub source_bytes: u64,
}

impl LogicalWeightPlan {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Physical keys of the logical weights (companions excluded).
    pub fn physical_keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.physical_key.as_str())
    }

    /// Physical keys of everything the plan accounts for: weights and companions.
    pub fn all_physical_keys(&self) -> impl Iterator<Item = &str> {
        self.physical_keys().chain(
            self.companions
                .iter()
                .map(|companion| companion.physical_key.as_str()),
        )
    }

    pub fn logical_keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.logical_key.as_str())
    }

    /// Codec ids used by this plan, deduplicated and sorted.
    pub fn codec_ids(&self) -> Vec<&'static str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.codec_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// The bytes this plan predicts resident after the read: per-tensor planned residency plus
    /// retained companions. A materializing reader's receipt must measure exactly this; the pair is
    /// the packed-native vs dense pricing seam.
    pub fn resident_bytes(&self) -> u64 {
        let tensors = self
            .tensors
            .iter()
            .map(|tensor| tensor.residency.resident_bytes)
            .sum::<u64>();
        let companions = self
            .companions
            .iter()
            .map(|companion| companion.resident_bytes)
            .sum::<u64>();
        tensors.saturating_add(companions)
    }

    /// The resident form of every planned tensor as synthesized tensor headers — logical key,
    /// resident dtype, logical shape — for byte-pricing projections (quantization projections
    /// included) that operate on tensor headers. Dense-fallback view: packed entries keep their
    /// stored encoding and shape.
    pub fn resident_tensor_headers(&self) -> Vec<SafetensorsTensorHeader> {
        self.tensors
            .iter()
            .map(|tensor| {
                let (dtype, shape) = match tensor.residency.mode {
                    ResidencyMode::Dense => (
                        match tensor.resident_encoding {
                            WeightEncoding::DenseBf16 => Dtype::BF16,
                            WeightEncoding::DenseF16 => Dtype::F16,
                            WeightEncoding::DenseF32 => Dtype::F32,
                            WeightEncoding::DenseF64 => Dtype::F64,
                            WeightEncoding::Fp8E4M3 => Dtype::F8_E4M3,
                            WeightEncoding::Fp8E5M2 => Dtype::F8_E5M2,
                            WeightEncoding::Int8 => Dtype::I8,
                            WeightEncoding::UInt8 => Dtype::U8,
                            WeightEncoding::Int16 => Dtype::I16,
                            WeightEncoding::Int32 => Dtype::I32,
                            WeightEncoding::Int64 => Dtype::I64,
                            WeightEncoding::UInt16 => Dtype::U16,
                            WeightEncoding::UInt32 => Dtype::U32,
                            WeightEncoding::UInt64 => Dtype::U64,
                            WeightEncoding::Bool => Dtype::BOOL,
                        },
                        tensor.shape.clone(),
                    ),
                    ResidencyMode::Packed => (
                        match tensor.encoding {
                            WeightEncoding::Fp8E4M3 => Dtype::F8_E4M3,
                            WeightEncoding::Fp8E5M2 => Dtype::F8_E5M2,
                            WeightEncoding::Int8 => Dtype::I8,
                            _ => Dtype::U8,
                        },
                        match &tensor.codec {
                            TensorCodecSpec::Mxfp8 { stored_shape, .. } => stored_shape.to_vec(),
                            // NVFP4's resident packed form is the on-disk `U8` byte matrix
                            // `[rows, cols / 2]`, not the logical element grid.
                            TensorCodecSpec::Nvfp4 { stored_shape, .. } => {
                                vec![stored_shape[0], stored_shape[1] / 2]
                            }
                            _ => tensor.shape.clone(),
                        },
                    ),
                };
                SafetensorsTensorHeader {
                    name: tensor.logical_key.clone(),
                    dtype,
                    shape,
                    data_bytes: tensor.residency.resident_bytes,
                }
            })
            .collect()
    }
}

/// Why a header could not compile to a plan. Every variant names the exact tensor it fails on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalWeightPlanError {
    /// No tensors at all — an empty checkpoint cannot be a model component.
    EmptyCheckpoint,
    /// The adapter's mapping recognises no logical key for this on-disk tensor.
    UnmappedKey { physical_key: String },
    /// Two on-disk tensors map onto one logical key (the mapping is not injective over this file).
    KeyCollision {
        logical_key: String,
        first_physical_key: String,
        second_physical_key: String,
    },
    /// The stored dtype is not one this contract can classify as a weight element type.
    UnclassifiedDtype { physical_key: String, dtype: String },
    /// The stored format has no registered codec on this backend.
    UnsupportedFormat {
        physical_key: String,
        format: String,
    },
    /// The tensor's declared shape and byte length disagree with its encoding.
    GeometryMismatch {
        physical_key: String,
        encoding: WeightEncoding,
        declared_bytes: u64,
        expected_bytes: u64,
    },
    /// The `.comfy_quant` tensor itself is not the rank-1 `U8` blob the convention stores.
    DescriptorTensor { physical_key: String },
    /// The descriptor payload was not supplied to the compiler (the backend reads `.comfy_quant`
    /// payloads before compiling; their absence is a caller defect, surfaced rather than skipped).
    DescriptorPayloadUnavailable { physical_key: String },
    /// The descriptor blob is malformed or names a format without a codec; `layer` is the layer the
    /// fault appears on, `defect` the exact problem.
    Descriptor {
        layer: String,
        physical_key: String,
        defect: ComfyQuantDescriptorError,
    },
    /// A `.comfy_quant` descriptor with no `{layer}.weight` to govern.
    OrphanDescriptor { physical_key: String },
    /// A scale companion whose layer has no descriptor (this includes the FLUX.2-style
    /// inline-scale convention, which is not this route's format), or one a format does not use.
    UnexpectedCompanion {
        physical_key: String,
        reason: &'static str,
    },
    /// A companion the descriptor's format requires is absent.
    MissingCompanion {
        physical_key: String,
        companion: String,
    },
    /// A companion exists but has the wrong dtype/shape for the format.
    CompanionMalformed {
        physical_key: String,
        defect: String,
    },
    /// The legacy `scaled_fp8` marker convention (`.scale_weight`/`.scale_input`): a real ComfyUI
    /// format this workspace does not read — ComfyUI itself rewrites it to `.comfy_quant` on load.
    LegacyScaledFp8 { physical_key: String },
    /// The descriptor's format and the weight's stored dtype disagree.
    DescriptorDtypeMismatch {
        physical_key: String,
        format: ComfyQuantFormat,
        dtype: String,
    },
    /// A quantized weight that must be a rank-2 `[out, in]` matrix is not.
    QuantizedWeightRank {
        physical_key: String,
        format: ComfyQuantFormat,
        shape: Vec<usize>,
    },
    /// MXFP8 block geometry (padding, scale swizzle shape, logical-vs-stored) is invalid.
    Mxfp8Geometry {
        physical_key: String,
        error: Mxfp8GeometryError,
    },
    /// NVFP4 block geometry (16-padding, packed byte shape, scale swizzle shape) is invalid.
    Nvfp4Geometry {
        physical_key: String,
        error: Nvfp4GeometryError,
    },
    /// The file-level `__metadata__._quantization_metadata` table is malformed.
    QuantizationMetadata { error: QuantizationMetadataError },
    /// A `_quantization_metadata` layer name matches no `{layer}.weight` tensor under any single
    /// state-dict prefix, or matches under more than one — the compiler will not guess which
    /// tensors a header-declared quantization governs.
    QuantizationMetadataLayer { layer: String, reason: String },
    /// One layer is declared twice, by a `.comfy_quant` tensor and by the file-level metadata, with
    /// two different descriptors. The whole descriptor is compared, not just its `format`:
    /// `full_precision_matrix_mult` decides packed-vs-dense residency, so a silent resolution in
    /// either declaration's favour would change how the layer is priced and run.
    DescriptorConflict {
        physical_key: String,
        tensor: ComfyQuantDescriptor,
        metadata: ComfyQuantDescriptor,
        /// The descriptor field names that disagree, in declaration order.
        disagreement: Vec<&'static str>,
    },
}

impl fmt::Display for LogicalWeightPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCheckpoint => write!(f, "checkpoint declares no tensors"),
            Self::UnmappedKey { physical_key } => write!(
                f,
                "on-disk tensor {physical_key:?} has no canonical logical key under this family adapter (unrecognized checkpoint, wrong family, or a key outside the dialect namespace)"
            ),
            Self::KeyCollision {
                logical_key,
                first_physical_key,
                second_physical_key,
            } => write!(
                f,
                "on-disk tensors {first_physical_key:?} and {second_physical_key:?} both map onto logical key {logical_key:?}"
            ),
            Self::UnclassifiedDtype { physical_key, dtype } => write!(
                f,
                "on-disk tensor {physical_key:?} uses dtype {dtype} that this contract cannot classify as a weight element type"
            ),
            Self::UnsupportedFormat {
                physical_key,
                format,
            } => write!(
                f,
                "on-disk tensor {physical_key:?} is stored as {format} and no checkpoint codec is registered for that format on this backend"
            ),
            Self::GeometryMismatch {
                physical_key,
                encoding,
                declared_bytes,
                expected_bytes,
            } => write!(
                f,
                "on-disk tensor {physical_key:?} declares {declared_bytes} bytes but its shape at {} needs {expected_bytes}",
                encoding.label()
            ),
            Self::DescriptorTensor { physical_key } => write!(
                f,
                "descriptor tensor {physical_key:?} must be a rank-1 U8 JSON blob"
            ),
            Self::DescriptorPayloadUnavailable { physical_key } => write!(
                f,
                "descriptor tensor {physical_key:?} has no payload supplied to the plan compiler"
            ),
            Self::Descriptor {
                layer,
                physical_key,
                defect,
            } => write!(f, "layer {layer:?} ({physical_key:?}): {defect}"),
            Self::OrphanDescriptor { physical_key } => write!(
                f,
                "descriptor tensor {physical_key:?} has no matching `.weight` tensor to govern"
            ),
            Self::UnexpectedCompanion {
                physical_key,
                reason,
            } => write!(
                f,
                "companion tensor {physical_key:?} is not part of its layer's format: {reason}"
            ),
            Self::MissingCompanion {
                physical_key,
                companion,
            } => write!(
                f,
                "quantized weight {physical_key:?} is missing its {companion:?} companion"
            ),
            Self::CompanionMalformed {
                physical_key,
                defect,
            } => write!(f, "companion tensor {physical_key:?}: {defect}"),
            Self::LegacyScaledFp8 { physical_key } => write!(
                f,
                "tensor {physical_key:?} belongs to the legacy `scaled_fp8` convention \
                 (`.scale_weight`/`.scale_input`), which this route does not read; re-export the \
                 checkpoint with `.comfy_quant` descriptors (ComfyUI rewrites the legacy form on load)"
            ),
            Self::DescriptorDtypeMismatch {
                physical_key,
                format,
                dtype,
            } => write!(
                f,
                "weight {physical_key:?} is declared `{format}` but stored as {dtype}"
            ),
            Self::QuantizedWeightRank {
                physical_key,
                format,
                shape,
            } => write!(
                f,
                "weight {physical_key:?} is declared `{format}` and must be a rank-2 [out, in] matrix, got shape {shape:?}"
            ),
            Self::Mxfp8Geometry {
                physical_key,
                error,
            } => write!(f, "weight {physical_key:?}: {error}"),
            Self::Nvfp4Geometry {
                physical_key,
                error,
            } => write!(f, "weight {physical_key:?}: {error}"),
            Self::QuantizationMetadata { error } => write!(f, "{error}"),
            Self::QuantizationMetadataLayer { layer, reason } => write!(
                f,
                "`_quantization_metadata` declares layer {layer:?} but {reason}"
            ),
            Self::DescriptorConflict {
                physical_key,
                tensor,
                metadata,
                disagreement,
            } => write!(
                f,
                "weight {physical_key:?} is declared `{}` (full_precision_matrix_mult={}) by its \
                 `.comfy_quant` tensor and `{}` (full_precision_matrix_mult={}) by \
                 `__metadata__._quantization_metadata`; the two declarations disagree on {}",
                tensor.format,
                tensor.full_precision_matrix_mult,
                metadata.format,
                metadata.full_precision_matrix_mult,
                disagreement.join(", ")
            ),
        }
    }
}

impl std::error::Error for LogicalWeightPlanError {}

impl From<LogicalWeightPlanError> for crate::Error {
    fn from(error: LogicalWeightPlanError) -> Self {
        crate::Error::Msg(format!("logical weight plan: {error}"))
    }
}

/// The companion suffixes of the `.comfy_quant` convention plus the refused legacy names.
const DESCRIPTOR_SUFFIX: &str = ".comfy_quant";
const WEIGHT_SCALE_SUFFIX: &str = ".weight_scale";
const INPUT_SCALE_SUFFIX: &str = ".input_scale";
const NVFP4_SCALE_2_SUFFIX: &str = ".weight_scale_2";
const LEGACY_SCALE_SUFFIXES: &[&str] = &[".scale_weight", ".scale_input"];
const LEGACY_MARKER_KEY: &str = "scaled_fp8";

fn scalar_scale_shape_ok(shape: &[usize]) -> bool {
    shape.is_empty() || shape == [1]
}

/// Compile a header into a [`LogicalWeightPlan`]. Pure and deterministic: the same header,
/// descriptor payloads, mapping, codec registry, and residency policy always produce the same plan,
/// in logical-key order. Fails on the first tensor (in header order, sorted by physical key) that
/// cannot be planned; every error names the exact tensor the fault appears on.
///
/// `descriptor_payloads` holds the raw bytes of each `.comfy_quant` tensor, keyed by its full
/// physical key — the backend reads those (small) payloads before compiling
/// (`read_safetensors_tensor_payloads`); the plan itself never touches weight data.
pub fn compile_logical_weight_plan(
    headers: &[SafetensorsTensorHeader],
    descriptor_payloads: &BTreeMap<String, Vec<u8>>,
    mapping: &dyn LogicalKeyMapping,
    codecs: &CheckpointCodecRegistry,
    residency: &dyn CodecResidencyPolicy,
) -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
    compile_logical_weight_plan_with_metadata(
        headers,
        descriptor_payloads,
        None,
        mapping,
        codecs,
        residency,
    )
}

/// [`compile_logical_weight_plan`] plus the file-level
/// `__metadata__._quantization_metadata` payload, when the checkpoint carries one instead of (or
/// alongside) per-layer `.comfy_quant` tensors — the form the ComfyUI Kitchen NVFP4 converters write
/// (sc-20641).
///
/// The metadata's layer names are commonly *relative* to the state-dict prefix its tensors carry
/// (`blocks.0.attn.wq` for `model.diffusion_model.blocks.0.attn.wq.weight`). The prefix is resolved
/// **once**, from the whole declared layer set, and must be unique and consistent across every
/// declared layer; anything else refuses rather than guessing which tensors a declaration governs.
/// A layer declared by both routes must agree, or the plan refuses.
pub fn compile_logical_weight_plan_with_metadata(
    headers: &[SafetensorsTensorHeader],
    descriptor_payloads: &BTreeMap<String, Vec<u8>>,
    quantization_metadata: Option<&str>,
    mapping: &dyn LogicalKeyMapping,
    codecs: &CheckpointCodecRegistry,
    residency: &dyn CodecResidencyPolicy,
) -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
    if headers.is_empty() {
        return Err(LogicalWeightPlanError::EmptyCheckpoint);
    }
    let by_name: BTreeMap<&str, &SafetensorsTensorHeader> = headers
        .iter()
        .map(|header| (header.name.as_str(), header))
        .collect();

    // ---- file-level descriptor table, resolved onto this file's layer bases ------------------
    let mut descriptors: BTreeMap<String, ComfyQuantDescriptor> = BTreeMap::new(); // by layer base
    if let Some(payload) = quantization_metadata {
        let table = crate::comfy_quant::parse_quantization_metadata(payload)
            .map_err(|error| LogicalWeightPlanError::QuantizationMetadata { error })?;
        let prefix = resolve_metadata_prefix(&by_name, &table)?;
        for (layer, descriptor) in table {
            descriptors.insert(format!("{prefix}{layer}"), descriptor);
        }
    }

    // ---- classify companions and validate descriptors, per layer ----------------------------
    for header in by_name.values() {
        if header.name == LEGACY_MARKER_KEY
            || LEGACY_SCALE_SUFFIXES
                .iter()
                .any(|suffix| header.name.ends_with(suffix))
        {
            return Err(LogicalWeightPlanError::LegacyScaledFp8 {
                physical_key: header.name.clone(),
            });
        }
        let Some(base) = header.name.strip_suffix(DESCRIPTOR_SUFFIX) else {
            continue;
        };
        if header.dtype != Dtype::U8 || header.shape.len() != 1 {
            return Err(LogicalWeightPlanError::DescriptorTensor {
                physical_key: header.name.clone(),
            });
        }
        let weight_key = format!("{base}.weight");
        if !by_name.contains_key(weight_key.as_str()) {
            return Err(LogicalWeightPlanError::OrphanDescriptor {
                physical_key: header.name.clone(),
            });
        }
        let payload = descriptor_payloads
            .get(header.name.as_str())
            .ok_or_else(|| LogicalWeightPlanError::DescriptorPayloadUnavailable {
                physical_key: header.name.clone(),
            })?;
        let descriptor = parse_comfy_quant_descriptor(payload).map_err(|defect| {
            LogicalWeightPlanError::Descriptor {
                layer: base.to_owned(),
                physical_key: header.name.clone(),
                defect,
            }
        })?;
        if let Some(previous) = descriptors.insert(base.to_owned(), descriptor) {
            if previous != descriptor {
                let mut disagreement = Vec::new();
                if previous.format != descriptor.format {
                    disagreement.push("format");
                }
                if previous.full_precision_matrix_mult != descriptor.full_precision_matrix_mult {
                    disagreement.push("full_precision_matrix_mult");
                }
                return Err(LogicalWeightPlanError::DescriptorConflict {
                    physical_key: weight_key,
                    tensor: descriptor,
                    metadata: previous,
                    disagreement,
                });
            }
        }
    }

    let companion_owner = |name: &str| -> Option<(String, CompanionRole)> {
        if let Some(base) = name.strip_suffix(DESCRIPTOR_SUFFIX) {
            return Some((base.to_owned(), CompanionRole::Descriptor));
        }
        // `.weight_scale_2` does not end with `.weight_scale` (different suffix), but check it
        // first anyway so the NVFP4 second-level scale is always named as such.
        if let Some(base) = name.strip_suffix(NVFP4_SCALE_2_SUFFIX) {
            return Some((base.to_owned(), CompanionRole::GlobalScale));
        }
        if let Some(base) = name.strip_suffix(WEIGHT_SCALE_SUFFIX) {
            return Some((base.to_owned(), CompanionRole::WeightScale));
        }
        if let Some(base) = name.strip_suffix(INPUT_SCALE_SUFFIX) {
            return Some((base.to_owned(), CompanionRole::InputScale));
        }
        None
    };

    let mut consumed_companions: BTreeMap<String, CompanionTensorPlan> = BTreeMap::new();
    let mut consume_companion =
        |header: &SafetensorsTensorHeader, role: CompanionRole, owner: &str, resident: u64| {
            consumed_companions.insert(
                header.name.clone(),
                CompanionTensorPlan {
                    physical_key: header.name.clone(),
                    role,
                    owner_physical_key: owner.to_owned(),
                    source_bytes: header.data_bytes,
                    resident_bytes: resident,
                },
            );
        };

    // ---- plan every weight tensor -----------------------------------------------------------
    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    let mut tensors = Vec::new();
    let mut source_bytes = 0_u64;
    for header in by_name.values() {
        source_bytes = source_bytes.saturating_add(header.data_bytes);
        if let Some((base, role)) = companion_owner(&header.name) {
            if role == CompanionRole::Descriptor {
                // Validated above; recorded when its weight is planned.
                continue;
            }
            let Some(descriptor) = descriptors.get(&base) else {
                return Err(LogicalWeightPlanError::UnexpectedCompanion {
                    physical_key: header.name.clone(),
                    reason: "its layer has no `.comfy_quant` descriptor (an undescribed scale \
                             companion is the FLUX.2 inline-scale convention, which is not this \
                             route's format)",
                });
            };
            if role == CompanionRole::InputScale
                && !matches!(
                    descriptor.format,
                    ComfyQuantFormat::Float8E4M3Fn
                        | ComfyQuantFormat::Float8E5M2
                        | ComfyQuantFormat::Mxfp8
                        | ComfyQuantFormat::Nvfp4
                )
            {
                return Err(LogicalWeightPlanError::UnexpectedCompanion {
                    physical_key: header.name.clone(),
                    reason: "`input_scale` is an fp8/fp4-format companion; this layer's format \
                             does not use it",
                });
            }
            if role == CompanionRole::GlobalScale && descriptor.format != ComfyQuantFormat::Nvfp4 {
                return Err(LogicalWeightPlanError::UnexpectedCompanion {
                    physical_key: header.name.clone(),
                    reason: "`weight_scale_2` is the NVFP4 second-level per-tensor scale; this \
                             layer's format has only one scale level",
                });
            }
            // Validated (dtype/shape) together with its weight below.
            continue;
        }

        // A weight (or plain dense) tensor.
        let logical_key = mapping.logical_key(&header.name).ok_or_else(|| {
            LogicalWeightPlanError::UnmappedKey {
                physical_key: header.name.clone(),
            }
        })?;
        if let Some(first) = owners.insert(logical_key.clone(), header.name.clone()) {
            return Err(LogicalWeightPlanError::KeyCollision {
                logical_key,
                first_physical_key: first,
                second_physical_key: header.name.clone(),
            });
        }
        let encoding = WeightEncoding::from_dtype(header.dtype).ok_or_else(|| {
            LogicalWeightPlanError::UnclassifiedDtype {
                physical_key: header.name.clone(),
                dtype: format!("{:?}", header.dtype),
            }
        })?;
        let base = header.name.strip_suffix(".weight");
        let descriptor = base.and_then(|base| descriptors.get(base)).copied();
        // Descriptor ↔ stored-dtype agreement first: "declared e4m3 but stored bf16" is the exact
        // defect, not a missing codec row.
        if let Some(descriptor) = descriptor {
            let expected_dtype = match descriptor.format {
                ComfyQuantFormat::Int8TensorwisePerRow => Dtype::I8,
                ComfyQuantFormat::Float8E4M3Fn | ComfyQuantFormat::Mxfp8 => Dtype::F8_E4M3,
                ComfyQuantFormat::Float8E5M2 => Dtype::F8_E5M2,
                // NVFP4 packs two E2M1 codes per byte; the file has no 4-bit dtype to declare.
                ComfyQuantFormat::Nvfp4 => Dtype::U8,
            };
            if header.dtype != expected_dtype {
                return Err(LogicalWeightPlanError::DescriptorDtypeMismatch {
                    physical_key: header.name.clone(),
                    format: descriptor.format,
                    dtype: format!("{:?}", header.dtype),
                });
            }
        }
        let format = StoredTensorFormat {
            encoding,
            descriptor: descriptor.map(|descriptor| descriptor.format),
        };
        let codec =
            codecs
                .for_format(format)
                .ok_or_else(|| LogicalWeightPlanError::UnsupportedFormat {
                    physical_key: header.name.clone(),
                    format: format.to_string(),
                })?;

        // Shape × dtype byte-length integrity of the stored tensor.
        let expected_bytes = header
            .element_count()
            .ok()
            .and_then(|count| count.checked_mul(encoding.element_bytes()))
            .ok_or_else(|| LogicalWeightPlanError::GeometryMismatch {
                physical_key: header.name.clone(),
                encoding,
                declared_bytes: header.data_bytes,
                expected_bytes: u64::MAX,
            })?;
        if expected_bytes != header.data_bytes {
            return Err(LogicalWeightPlanError::GeometryMismatch {
                physical_key: header.name.clone(),
                encoding,
                declared_bytes: header.data_bytes,
                expected_bytes,
            });
        }

        // Per-format validation of the layer: dtype agreement, rank, companions, block geometry.
        let (codec_spec, logical_shape) = match descriptor {
            // Undescribed fp8 is the plain `weight_dtype=fp8_*` cast: the scalar codec at exactly
            // unit scale (ComfyUI's own reference load is `weight.to(compute_dtype)`). The cast
            // covers *every* tensor of the file — biases and norms included — so no rank
            // constraint applies here. Any companion for this layer refuses above (a scale without
            // a descriptor is the FLUX.2 inline convention, not this format).
            None if matches!(encoding, WeightEncoding::Fp8E4M3 | WeightEncoding::Fp8E5M2) => (
                TensorCodecSpec::ScalarFp8 {
                    scale: ScalarScaleSource::Unit,
                    input_scale: None,
                    full_precision_matrix_mult: false,
                },
                header.shape.clone(),
            ),
            None => (TensorCodecSpec::Dense, header.shape.clone()),
            Some(descriptor) => {
                let base = base.expect("descriptor implies a `.weight` suffix");
                if header.shape.len() != 2 {
                    return Err(LogicalWeightPlanError::QuantizedWeightRank {
                        physical_key: header.name.clone(),
                        format: descriptor.format,
                        shape: header.shape.clone(),
                    });
                }
                let scale_key = format!("{base}{WEIGHT_SCALE_SUFFIX}");
                let scale = *by_name.get(scale_key.as_str()).ok_or_else(|| {
                    LogicalWeightPlanError::MissingCompanion {
                        physical_key: header.name.clone(),
                        companion: scale_key.clone(),
                    }
                })?;
                let input_scale_key = format!("{base}{INPUT_SCALE_SUFFIX}");
                let input_scale = by_name.get(input_scale_key.as_str()).copied();
                if let Some(input_scale) = input_scale {
                    if input_scale.dtype != Dtype::F32 || !scalar_scale_shape_ok(&input_scale.shape)
                    {
                        return Err(LogicalWeightPlanError::CompanionMalformed {
                            physical_key: input_scale.name.clone(),
                            defect: format!(
                                "input_scale must be a scalar F32, got {:?} {:?}",
                                input_scale.dtype, input_scale.shape
                            ),
                        });
                    }
                }
                match descriptor.format {
                    ComfyQuantFormat::Float8E4M3Fn | ComfyQuantFormat::Float8E5M2 => {
                        if scale.dtype != Dtype::F32 || !scalar_scale_shape_ok(&scale.shape) {
                            return Err(LogicalWeightPlanError::CompanionMalformed {
                                physical_key: scale.name.clone(),
                                defect: format!(
                                    "per-tensor fp8 weight_scale must be a scalar F32, got {:?} {:?}",
                                    scale.dtype, scale.shape
                                ),
                            });
                        }
                        (
                            TensorCodecSpec::ScalarFp8 {
                                scale: ScalarScaleSource::Companion {
                                    physical_key: scale.name.clone(),
                                },
                                input_scale: input_scale.map(|header| header.name.clone()),
                                full_precision_matrix_mult: descriptor.full_precision_matrix_mult,
                            },
                            header.shape.clone(),
                        )
                    }
                    ComfyQuantFormat::Mxfp8 => {
                        if !matches!(scale.dtype, Dtype::U8 | Dtype::F8_E8M0) {
                            return Err(LogicalWeightPlanError::CompanionMalformed {
                                physical_key: scale.name.clone(),
                                defect: format!(
                                    "mxfp8 weight_scale must be U8 or F8_E8M0 (E8M0 exponents), got {:?}",
                                    scale.dtype
                                ),
                            });
                        }
                        let declared = mapping.logical_shape(&logical_key);
                        let logical = validate_mxfp8_geometry(
                            &header.shape,
                            &scale.shape,
                            declared.as_deref(),
                        )
                        .map_err(|error| {
                            LogicalWeightPlanError::Mxfp8Geometry {
                                physical_key: header.name.clone(),
                                error,
                            }
                        })?;
                        (
                            TensorCodecSpec::Mxfp8 {
                                scale: scale.name.clone(),
                                stored_shape: [header.shape[0], header.shape[1]],
                                logical_shape_declared: declared.is_some(),
                                full_precision_matrix_mult: descriptor.full_precision_matrix_mult,
                            },
                            logical.to_vec(),
                        )
                    }
                    ComfyQuantFormat::Nvfp4 => {
                        if !matches!(scale.dtype, Dtype::F8_E4M3 | Dtype::U8) {
                            return Err(LogicalWeightPlanError::CompanionMalformed {
                                physical_key: scale.name.clone(),
                                defect: format!(
                                    "nvfp4 weight_scale must be F8_E4M3 or U8 (E4M3 block scales), got {:?}",
                                    scale.dtype
                                ),
                            });
                        }
                        let global_key = format!("{base}{NVFP4_SCALE_2_SUFFIX}");
                        let global = *by_name.get(global_key.as_str()).ok_or_else(|| {
                            LogicalWeightPlanError::MissingCompanion {
                                physical_key: header.name.clone(),
                                companion: global_key.clone(),
                            }
                        })?;
                        if global.dtype != Dtype::F32 || !scalar_scale_shape_ok(&global.shape) {
                            return Err(LogicalWeightPlanError::CompanionMalformed {
                                physical_key: global.name.clone(),
                                defect: format!(
                                    "nvfp4 weight_scale_2 must be a scalar F32, got {:?} {:?}",
                                    global.dtype, global.shape
                                ),
                            });
                        }
                        let declared = mapping.logical_shape(&logical_key);
                        let (stored, logical) = validate_nvfp4_geometry(
                            &header.shape,
                            &scale.shape,
                            declared.as_deref(),
                        )
                        .map_err(|error| {
                            LogicalWeightPlanError::Nvfp4Geometry {
                                physical_key: header.name.clone(),
                                error,
                            }
                        })?;
                        (
                            TensorCodecSpec::Nvfp4 {
                                block_scale: scale.name.clone(),
                                global_scale: global.name.clone(),
                                input_scale: input_scale.map(|header| header.name.clone()),
                                stored_shape: stored,
                                logical_shape: logical,
                                logical_shape_declared: declared.is_some(),
                                full_precision_matrix_mult: descriptor.full_precision_matrix_mult,
                            },
                            logical.to_vec(),
                        )
                    }
                    ComfyQuantFormat::Int8TensorwisePerRow => {
                        let rows = header.shape[0];
                        let scalar_single_row = rows == 1 && scale.shape.is_empty();
                        let per_row_ok = scale.shape == [rows] || scale.shape == [rows, 1];
                        if scale.dtype != Dtype::F32 || !(per_row_ok || scalar_single_row) {
                            return Err(LogicalWeightPlanError::CompanionMalformed {
                                physical_key: scale.name.clone(),
                                defect: format!(
                                    "int8 per-row weight_scale must be F32 [{rows}] or [{rows},1]{}, got {:?} {:?}",
                                    if rows == 1 { " or scalar" } else { "" },
                                    scale.dtype,
                                    scale.shape
                                ),
                            });
                        }
                        (
                            TensorCodecSpec::Int8PerRow {
                                scale: scale.name.clone(),
                                full_precision_matrix_mult: descriptor.full_precision_matrix_mult,
                            },
                            header.shape.clone(),
                        )
                    }
                }
            }
        };

        // Residency: the backend's packed-vs-dense decision, priced per tensor. A layer flagged
        // `full_precision_matrix_mult` never runs packed.
        let mode = if codec_spec.full_precision_matrix_mult() {
            ResidencyMode::Dense
        } else {
            residency.residency(codec, &codec_spec, &header.shape)
        };
        let resident_bytes = match mode {
            ResidencyMode::Packed => header.data_bytes,
            ResidencyMode::Dense => logical_shape
                .iter()
                .try_fold(1_u64, |count, dimension| {
                    count.checked_mul(*dimension as u64)
                })
                .and_then(|count| count.checked_mul(codec.resident_encoding.element_bytes()))
                .ok_or_else(|| LogicalWeightPlanError::GeometryMismatch {
                    physical_key: header.name.clone(),
                    encoding,
                    declared_bytes: header.data_bytes,
                    expected_bytes: u64::MAX,
                })?,
        };

        // Record the layer's companions with the residency the mode implies.
        if let Some(base) = base {
            let descriptor_key = format!("{base}{DESCRIPTOR_SUFFIX}");
            if let Some(descriptor_header) = by_name.get(descriptor_key.as_str()) {
                consume_companion(
                    descriptor_header,
                    CompanionRole::Descriptor,
                    &header.name,
                    0,
                );
            }
            match &codec_spec {
                TensorCodecSpec::Dense => {}
                TensorCodecSpec::ScalarFp8 {
                    scale, input_scale, ..
                } => {
                    if let ScalarScaleSource::Companion { physical_key } = scale {
                        let scale_header = by_name[physical_key.as_str()];
                        let retained = match mode {
                            ResidencyMode::Packed => scale_header.data_bytes,
                            ResidencyMode::Dense => 0,
                        };
                        consume_companion(
                            scale_header,
                            CompanionRole::WeightScale,
                            &header.name,
                            retained,
                        );
                    }
                    if let Some(input_scale) = input_scale {
                        let input_header = by_name[input_scale.as_str()];
                        let retained = match mode {
                            ResidencyMode::Packed => input_header.data_bytes,
                            ResidencyMode::Dense => 0,
                        };
                        consume_companion(
                            input_header,
                            CompanionRole::InputScale,
                            &header.name,
                            retained,
                        );
                    }
                }
                TensorCodecSpec::Nvfp4 {
                    block_scale,
                    global_scale,
                    input_scale,
                    ..
                } => {
                    // NVFP4 keeps *both* scale levels resident on a packed-native load: cuBLASLt
                    // reads the block scales as a device buffer and folds `weight_scale_2` into
                    // `alpha`. A dense decode consumes both.
                    for (key, role) in [
                        (Some(block_scale.as_str()), CompanionRole::WeightScale),
                        (Some(global_scale.as_str()), CompanionRole::GlobalScale),
                        (input_scale.as_deref(), CompanionRole::InputScale),
                    ] {
                        let Some(key) = key else { continue };
                        let companion_header = by_name[key];
                        let retained = match mode {
                            ResidencyMode::Packed => companion_header.data_bytes,
                            ResidencyMode::Dense => 0,
                        };
                        consume_companion(companion_header, role, &header.name, retained);
                    }
                }
                TensorCodecSpec::Mxfp8 { scale, .. }
                | TensorCodecSpec::Int8PerRow { scale, .. } => {
                    let scale_header = by_name[scale.as_str()];
                    let retained = match mode {
                        ResidencyMode::Packed => scale_header.data_bytes,
                        ResidencyMode::Dense => 0,
                    };
                    consume_companion(
                        scale_header,
                        CompanionRole::WeightScale,
                        &header.name,
                        retained,
                    );
                    let input_scale_key = format!("{base}{INPUT_SCALE_SUFFIX}");
                    if let Some(input_header) = by_name.get(input_scale_key.as_str()) {
                        // Same rule as the ScalarFp8 arm: a packed-native load keeps the activation
                        // scale resident, a dense decode consumes it. Pricing it at zero
                        // unconditionally would under-report a packed MXFP8 row. (Only the MXFP8
                        // half of this arm can get here — the companion-surface check above refuses
                        // an `input_scale` on an `int8_tensorwise` layer by format.)
                        let retained = match mode {
                            ResidencyMode::Packed => input_header.data_bytes,
                            ResidencyMode::Dense => 0,
                        };
                        consume_companion(
                            input_header,
                            CompanionRole::InputScale,
                            &header.name,
                            retained,
                        );
                    }
                }
            }
        }

        tensors.push(LogicalTensorPlan {
            logical_key,
            physical_key: header.name.clone(),
            encoding,
            shape: logical_shape,
            source_bytes: header.data_bytes,
            codec_id: codec.codec_id,
            resident_encoding: codec.resident_encoding,
            codec: codec_spec,
            residency: PlannedResidency {
                mode,
                resident_bytes,
            },
        });
    }

    // Every companion in the file must have been claimed by exactly one planned layer.
    for header in by_name.values() {
        let Some((_, role)) = companion_owner(&header.name) else {
            continue;
        };
        if role != CompanionRole::Descriptor && !consumed_companions.contains_key(&header.name) {
            return Err(LogicalWeightPlanError::UnexpectedCompanion {
                physical_key: header.name.clone(),
                reason: "no planned layer consumes it",
            });
        }
    }

    tensors.sort_by(|left, right| left.logical_key.cmp(&right.logical_key));
    Ok(LogicalWeightPlan {
        mapping_id: mapping.mapping_id(),
        tensors,
        companions: consumed_companions.into_values().collect(),
        source_bytes,
    })
}

/// Resolve the single state-dict prefix under which every `_quantization_metadata` layer name has a
/// `{prefix}{layer}.weight` tensor in this file.
///
/// Converters write the layer names the *model* uses, while the file may store them under a wrapper
/// prefix (`model.diffusion_model.`). Candidates are derived from one declared layer, then the whole
/// table must resolve under exactly one of them — a table that resolves under two prefixes, or under
/// none, refuses instead of the compiler picking one.
fn resolve_metadata_prefix(
    by_name: &BTreeMap<&str, &SafetensorsTensorHeader>,
    table: &BTreeMap<String, ComfyQuantDescriptor>,
) -> Result<String, LogicalWeightPlanError> {
    let (first, _) = table
        .iter()
        .next()
        .expect("parse_quantization_metadata refuses an empty table");
    let needle = format!("{first}.weight");
    let candidates: Vec<String> = by_name
        .keys()
        .filter_map(|name| {
            let prefix = name.strip_suffix(needle.as_str())?;
            // A prefix is either empty or ends at a key boundary — `…q_proj` must not match
            // `…attn.q_proj` by raw suffix.
            (prefix.is_empty() || prefix.ends_with('.')).then(|| prefix.to_owned())
        })
        .collect();
    if candidates.is_empty() {
        return Err(LogicalWeightPlanError::QuantizationMetadataLayer {
            layer: first.clone(),
            reason: format!("no tensor in this checkpoint is named {needle:?} under any prefix"),
        });
    }
    let resolved: Vec<&String> = candidates
        .iter()
        .filter(|prefix| {
            table
                .keys()
                .all(|layer| by_name.contains_key(format!("{prefix}{layer}.weight").as_str()))
        })
        .collect();
    match resolved.as_slice() {
        [prefix] => Ok((*prefix).clone()),
        [] => {
            // At least one candidate matched the probe layer; name the first layer that does not
            // resolve under it, which is the actionable fact.
            let prefix = &candidates[0];
            let missing = table
                .keys()
                .find(|layer| !by_name.contains_key(format!("{prefix}{layer}.weight").as_str()))
                .expect("an unresolved candidate has at least one missing layer");
            Err(LogicalWeightPlanError::QuantizationMetadataLayer {
                layer: missing.clone(),
                reason: format!(
                    "no tensor {:?} exists (prefix {prefix:?} was resolved from layer {first:?})",
                    format!("{prefix}{missing}.weight")
                ),
            })
        }
        many => Err(LogicalWeightPlanError::QuantizationMetadataLayer {
            layer: first.clone(),
            reason: format!(
                "its layer names resolve under {} different state-dict prefixes ({:?}); refusing to \
                 guess which tensors the declaration governs",
                many.len(),
                many
            ),
        }),
    }
}

/// What one codec left resident after a read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecResidencyReport {
    pub codec_id: &'static str,
    pub tensor_count: usize,
    /// Bytes read from the source file for these tensors (from the plan; companions included).
    pub source_bytes: u64,
    /// Bytes the decoded tensors occupy resident, measured from the backend arrays after decode
    /// (retained companions included).
    pub resident_bytes: u64,
}

/// Whether a read evaluated its arrays. A deferred read (block-streamed or bounded-quantized
/// loaders) leaves payloads lazy on purpose and must not report invented resident bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalReadMaterialization {
    /// Every planned tensor was evaluated; residency reports are measured.
    Materialized,
    /// Payloads remain lazy; residency reports are absent by construction.
    Deferred,
}

/// The receipt a backend reader returns with the logical weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWeightReceipt {
    pub mapping_id: &'static str,
    pub tensor_count: usize,
    pub source_bytes: u64,
    pub materialization: LogicalReadMaterialization,
    /// One report per codec actually used; empty when deferred.
    pub residency: Vec<CodecResidencyReport>,
}

impl LogicalWeightReceipt {
    pub fn resident_bytes(&self) -> u64 {
        self.residency
            .iter()
            .map(|report| report.resident_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(name: &str, dtype: Dtype, shape: &[usize]) -> SafetensorsTensorHeader {
        let elements: usize = shape.iter().product();
        SafetensorsTensorHeader {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: (elements * dtype.size()) as u64,
        }
    }

    fn no_descriptors() -> BTreeMap<String, Vec<u8>> {
        BTreeMap::new()
    }

    fn compile(
        headers: &[SafetensorsTensorHeader],
        descriptors: &BTreeMap<String, Vec<u8>>,
        mapping: &dyn LogicalKeyMapping,
        codecs: &CheckpointCodecRegistry,
    ) -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
        compile_logical_weight_plan(headers, descriptors, mapping, codecs, &DenseResidencyPolicy)
    }

    struct StripPrefix;

    impl LogicalKeyMapping for StripPrefix {
        fn mapping_id(&self) -> &'static str {
            "strip-prefix-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
    }

    struct Collapse;

    impl LogicalKeyMapping for Collapse {
        fn mapping_id(&self) -> &'static str {
            "collapse-test"
        }
        fn logical_key(&self, _physical_key: &str) -> Option<String> {
            Some("same".to_owned())
        }
    }

    fn baseline() -> CheckpointCodecRegistry {
        CheckpointCodecRegistry::new([DENSE_BF16_CODEC]).unwrap()
    }

    fn full() -> CheckpointCodecRegistry {
        CheckpointCodecRegistry::new(
            DENSE_CODECS
                .iter()
                .chain(COMFY_QUANT_CODECS.iter())
                .copied(),
        )
        .unwrap()
    }

    /// sc-20641. A checkpoint whose quantization is declared file-wide plans exactly as one declared
    /// by per-layer `.comfy_quant` tensors, including when the metadata's layer names are relative
    /// to the tensors' state-dict prefix — and the prefix is never guessed.
    #[test]
    fn header_declared_quantization_resolves_one_prefix_or_refuses() {
        let nvfp4_layer = |base: &str| {
            [
                header(&format!("{base}.weight"), Dtype::U8, &[32, 32]),
                header(&format!("{base}.weight_scale"), Dtype::F8_E4M3, &[128, 4]),
                header(&format!("{base}.weight_scale_2"), Dtype::F32, &[]),
            ]
        };
        let metadata = r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#;
        let compile_meta = |headers: &[SafetensorsTensorHeader],
                            metadata: Option<&str>|
         -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
            compile_logical_weight_plan_with_metadata(
                headers,
                &no_descriptors(),
                metadata,
                &StripPrefix,
                &full(),
                &DenseResidencyPolicy,
            )
        };

        // Relative layer name `q` under the file's `model.` prefix.
        let headers = nvfp4_layer("model.q");
        let plan = compile_meta(&headers, Some(metadata)).expect("header-declared nvfp4 plans");
        assert_eq!(plan.codec_ids(), vec!["nvfp4-v1"]);
        assert_eq!(
            plan.tensors[0].shape,
            vec![32, 64],
            "U8 [32,32] → 64 codes/row"
        );
        assert!(matches!(
            plan.tensors[0].codec,
            TensorCodecSpec::Nvfp4 {
                stored_shape: [32, 64],
                ..
            }
        ));
        // Without the declaration the same file has no codec for a bare U8 weight: the metadata is
        // load-bearing, not decoration.
        assert!(matches!(
            compile_meta(&headers, None),
            Err(LogicalWeightPlanError::UnsupportedFormat { .. })
        ));

        // A layer name matching nothing refuses instead of being ignored.
        let error = compile_meta(
            &headers,
            Some(r#"{"format_version": "1.0", "layers": {"absent": {"format": "nvfp4"}}}"#),
        )
        .expect_err("an unmatched layer must refuse");
        assert!(
            matches!(&error, LogicalWeightPlanError::QuantizationMetadataLayer { layer, .. } if layer == "absent"),
            "{error}"
        );

        // The same relative name under TWO prefixes is ambiguous: refuse rather than pick one.
        let mut ambiguous: Vec<SafetensorsTensorHeader> = nvfp4_layer("model.a.q").into();
        ambiguous.extend(nvfp4_layer("model.b.q"));
        let error = compile_meta(&ambiguous, Some(metadata))
            .expect_err("two candidate prefixes must refuse");
        assert!(
            matches!(&error, LogicalWeightPlanError::QuantizationMetadataLayer { reason, .. }
                if reason.contains("2 different state-dict prefixes")),
            "{error}"
        );

        // A prefix that resolves the probe layer but not every declared layer names the layer that
        // does not resolve.
        let mut partial: Vec<SafetensorsTensorHeader> = nvfp4_layer("model.q").into();
        partial.extend(nvfp4_layer("model.k"));
        let error = compile_meta(
            &partial,
            Some(r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}, "deep.k": {"format": "nvfp4"}}}"#),
        )
        .expect_err("a partially-resolving prefix must refuse");
        assert!(
            matches!(&error, LogicalWeightPlanError::QuantizationMetadataLayer { layer, .. } if layer == "deep.k"),
            "{error}"
        );

        // A malformed table is a typed refusal, not a silently-unquantized plan.
        assert!(matches!(
            compile_meta(&headers, Some(r#"{"format_version": "9.9", "layers": {}}"#)),
            Err(LogicalWeightPlanError::QuantizationMetadata { .. })
        ));
    }

    /// sc-20641. A layer declared by BOTH routes must agree; a disagreement refuses rather than one
    /// route silently winning.
    #[test]
    fn a_layer_declared_twice_must_agree() {
        let descriptor = br#"{"format": "nvfp4"}"#;
        let headers = [
            header("model.q.weight", Dtype::U8, &[32, 32]),
            header("model.q.weight_scale", Dtype::F8_E4M3, &[128, 4]),
            header("model.q.weight_scale_2", Dtype::F32, &[]),
            header("model.q.comfy_quant", Dtype::U8, &[descriptor.len()]),
        ];
        let descriptors: BTreeMap<String, Vec<u8>> =
            [("model.q.comfy_quant".to_owned(), descriptor.to_vec())].into();
        let compile_meta = |payload: &str| {
            compile_logical_weight_plan_with_metadata(
                &headers,
                &descriptors,
                Some(payload),
                &StripPrefix,
                &full(),
                &DenseResidencyPolicy,
            )
        };
        // Agreeing declarations plan.
        assert!(
            compile_meta(r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#)
                .is_ok()
        );
        // Disagreeing formats refuse, naming the weight, both descriptors and the field.
        let error = compile_meta(
            r#"{"format_version": "1.0", "layers": {"q": {"format": "float8_e4m3fn"}}}"#,
        )
        .expect_err("two declarations that disagree must refuse");
        assert_eq!(
            error,
            LogicalWeightPlanError::DescriptorConflict {
                physical_key: "model.q.weight".to_owned(),
                tensor: ComfyQuantDescriptor {
                    format: ComfyQuantFormat::Nvfp4,
                    full_precision_matrix_mult: false,
                },
                metadata: ComfyQuantDescriptor {
                    format: ComfyQuantFormat::Float8E4M3Fn,
                    full_precision_matrix_mult: false,
                },
                disagreement: vec!["format"],
            }
        );
        assert!(error.to_string().contains("model.q.weight"), "{error}");

        // The conflict check is over the WHOLE descriptor, not just `format`: agreeing on `nvfp4`
        // while disagreeing on `full_precision_matrix_mult` decides packed-vs-dense residency, so it
        // must refuse too rather than resolve in the tensor declaration's favour.
        let error = compile_meta(
            r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4", "full_precision_matrix_mult": true}}}"#,
        )
        .expect_err("a `full_precision_matrix_mult` disagreement must refuse");
        assert_eq!(
            error,
            LogicalWeightPlanError::DescriptorConflict {
                physical_key: "model.q.weight".to_owned(),
                tensor: ComfyQuantDescriptor {
                    format: ComfyQuantFormat::Nvfp4,
                    full_precision_matrix_mult: false,
                },
                metadata: ComfyQuantDescriptor {
                    format: ComfyQuantFormat::Nvfp4,
                    full_precision_matrix_mult: true,
                },
                disagreement: vec!["full_precision_matrix_mult"],
            }
        );
        assert!(
            error.to_string().contains("full_precision_matrix_mult"),
            "{error}"
        );
    }

    /// sc-20641. Packed NVFP4 residency prices the stored nibbles and retains BOTH scale levels; the
    /// dense fallback prices the logical bf16 grid and retains neither.
    #[test]
    fn nvfp4_prices_packed_and_dense_residency_independently() {
        struct PackEverything;
        impl CodecResidencyPolicy for PackEverything {
            fn residency(
                &self,
                _codec: &CheckpointCodecRegistration,
                spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                if spec.full_precision_matrix_mult() {
                    ResidencyMode::Dense
                } else {
                    ResidencyMode::Packed
                }
            }
        }
        let descriptor = br#"{"format": "nvfp4"}"#;
        let headers = [
            header("model.q.weight", Dtype::U8, &[32, 32]),
            header("model.q.weight_scale", Dtype::F8_E4M3, &[128, 4]),
            header("model.q.weight_scale_2", Dtype::F32, &[]),
            header("model.q.input_scale", Dtype::F32, &[]),
            header("model.q.comfy_quant", Dtype::U8, &[descriptor.len()]),
        ];
        let descriptors: BTreeMap<String, Vec<u8>> =
            [("model.q.comfy_quant".to_owned(), descriptor.to_vec())].into();
        let plan = compile_logical_weight_plan(
            &headers,
            &descriptors,
            &StripPrefix,
            &full(),
            &PackEverything,
        )
        .expect("packed plan");
        assert_eq!(plan.tensors[0].residency.mode, ResidencyMode::Packed);
        assert_eq!(
            plan.tensors[0].residency.resident_bytes,
            32 * 32,
            "packed holds the stored 4-bit byte matrix"
        );
        let retained: BTreeMap<&str, u64> = plan
            .companions
            .iter()
            .map(|companion| (companion.physical_key.as_str(), companion.resident_bytes))
            .collect();
        assert_eq!(
            retained["model.q.weight_scale"],
            128 * 4,
            "block scales stay"
        );
        assert_eq!(
            retained["model.q.weight_scale_2"], 4,
            "the global scale stays"
        );
        assert_eq!(
            retained["model.q.input_scale"], 4,
            "the activation scale stays"
        );
        assert_eq!(
            retained["model.q.comfy_quant"], 0,
            "descriptors never reside"
        );
        assert_eq!(plan.resident_bytes(), 32 * 32 + 128 * 4 + 4 + 4);

        // The dense fallback of the same file: bf16 over the logical grid, no companion retained.
        let dense = compile_logical_weight_plan(
            &headers,
            &descriptors,
            &StripPrefix,
            &full(),
            &DenseResidencyPolicy,
        )
        .expect("dense plan");
        assert_eq!(dense.tensors[0].residency.resident_bytes, 32 * 64 * 2);
        assert!(dense
            .companions
            .iter()
            .all(|companion| companion.resident_bytes == 0));
        assert_eq!(dense.resident_bytes(), 32 * 64 * 2);
        // The packed form of a 4-bit weight is a quarter of its bf16 dense form, plus scales.
        assert!(plan.resident_bytes() < dense.resident_bytes());

        // The packed projection presents the stored byte matrix, the dense one the logical grid.
        assert_eq!(plan.resident_tensor_headers()[0].shape, vec![32, 32]);
        assert_eq!(dense.resident_tensor_headers()[0].shape, vec![32, 64]);
    }

    #[test]
    fn plan_is_a_sorted_bijection_with_source_bytes_and_codec_ids() {
        let headers = [
            header("model.b.weight", Dtype::BF16, &[2, 3]),
            header("model.a.weight", Dtype::BF16, &[4]),
        ];
        let plan = compile(&headers, &no_descriptors(), &StripPrefix, &baseline()).unwrap();
        assert_eq!(plan.mapping_id, "strip-prefix-test");
        assert_eq!(
            plan.logical_keys().collect::<Vec<_>>(),
            ["a.weight", "b.weight"]
        );
        assert_eq!(
            plan.physical_keys().collect::<Vec<_>>(),
            ["model.a.weight", "model.b.weight"]
        );
        assert_eq!(plan.source_bytes, 8 + 12);
        assert_eq!(plan.codec_ids(), ["dense-bf16-v1"]);
        assert_eq!(plan.tensors[1].shape, vec![2, 3]);
        assert_eq!(plan.tensors[1].encoding, WeightEncoding::DenseBf16);
        assert_eq!(plan.tensors[1].codec, TensorCodecSpec::Dense);
        assert_eq!(
            plan.tensors[1].residency,
            PlannedResidency {
                mode: ResidencyMode::Dense,
                resident_bytes: 12
            }
        );
        assert!(plan.companions.is_empty());
        assert_eq!(plan.resident_bytes(), 20);
        // Deterministic regardless of header order.
        let reversed = [headers[1].clone(), headers[0].clone()];
        assert_eq!(
            compile(&reversed, &no_descriptors(), &StripPrefix, &baseline()).unwrap(),
            plan
        );
    }

    #[test]
    fn plan_refuses_unmapped_keys_collisions_and_unregistered_formats_by_tensor() {
        let unmapped = [header("foreign.weight", Dtype::BF16, &[1])];
        assert_eq!(
            compile(&unmapped, &no_descriptors(), &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::UnmappedKey {
                physical_key: "foreign.weight".to_owned()
            })
        );
        let colliding = [
            header("model.x", Dtype::BF16, &[1]),
            header("model.y", Dtype::BF16, &[1]),
        ];
        assert_eq!(
            compile(&colliding, &no_descriptors(), &Collapse, &baseline()),
            Err(LogicalWeightPlanError::KeyCollision {
                logical_key: "same".to_owned(),
                first_physical_key: "model.x".to_owned(),
                second_physical_key: "model.y".to_owned(),
            })
        );
        // fp8 against a registry with no fp8 codec still refuses, naming the format.
        let fp8 = [
            header("model.ok", Dtype::BF16, &[1]),
            header("model.packed", Dtype::F8_E4M3, &[4]),
        ];
        assert_eq!(
            compile(&fp8, &no_descriptors(), &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::UnsupportedFormat {
                physical_key: "model.packed".to_owned(),
                format: "fp8-e4m3".to_owned(),
            })
        );
        assert_eq!(
            compile(&[], &no_descriptors(), &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::EmptyCheckpoint)
        );
    }

    #[test]
    fn plan_refuses_declared_bytes_that_disagree_with_shape_and_encoding() {
        let mut short = header("model.w", Dtype::BF16, &[3]);
        short.data_bytes = 5;
        assert_eq!(
            compile(&[short], &no_descriptors(), &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::GeometryMismatch {
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                declared_bytes: 5,
                expected_bytes: 6,
            })
        );
    }

    #[test]
    fn codec_registry_rejects_duplicate_ids_and_double_claimed_formats() {
        let duplicate = CheckpointCodecRegistry::new([DENSE_BF16_CODEC, DENSE_BF16_CODEC]);
        assert!(
            duplicate
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("duplicate checkpoint codec id")),
            "{duplicate:?}"
        );
        let double_claim = CheckpointCodecRegistry::new([
            DENSE_BF16_CODEC,
            CheckpointCodecRegistration {
                codec_id: "dense-bf16-shadow",
                ..DENSE_BF16_CODEC
            },
        ]);
        assert!(
            double_claim
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("both claim stored format")),
            "{double_claim:?}"
        );
        let malformed = CheckpointCodecRegistry::new([CheckpointCodecRegistration {
            codec_id: "Dense BF16",
            ..DENSE_BF16_CODEC
        }]);
        assert!(malformed.is_err());
        let empty_claims = CheckpointCodecRegistry::new([CheckpointCodecRegistration {
            codec_id: "claims-nothing",
            stored: &[],
            resident_encoding: WeightEncoding::DenseBf16,
        }]);
        assert!(empty_claims.is_err());
        let registry = baseline();
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.for_encoding(WeightEncoding::DenseBf16),
            Some(&DENSE_BF16_CODEC)
        );
        assert_eq!(registry.for_encoding(WeightEncoding::Fp8E4M3), None);
        assert_eq!(registry.by_id("dense-bf16-v1"), Some(&DENSE_BF16_CODEC));
    }

    #[test]
    fn comfy_codec_table_separates_the_two_fp8_e4m3_rows_by_descriptor() {
        let registry = full();
        assert_eq!(registry.len(), 8);
        // NVFP4 is the one `U8`-stored codec, separated from an undescribed `U8` tensor (which has
        // no codec at all) by its descriptor half.
        assert_eq!(
            registry
                .for_format(StoredTensorFormat::described(
                    WeightEncoding::UInt8,
                    ComfyQuantFormat::Nvfp4
                ))
                .map(|codec| codec.codec_id),
            Some("nvfp4-v1")
        );
        assert_eq!(registry.for_encoding(WeightEncoding::UInt8), None);
        assert_eq!(
            registry
                .for_format(StoredTensorFormat::described(
                    WeightEncoding::Fp8E4M3,
                    ComfyQuantFormat::Float8E4M3Fn
                ))
                .map(|codec| codec.codec_id),
            Some("fp8-e4m3-scalar-v1")
        );
        assert_eq!(
            registry
                .for_format(StoredTensorFormat::described(
                    WeightEncoding::Fp8E4M3,
                    ComfyQuantFormat::Mxfp8
                ))
                .map(|codec| codec.codec_id),
            Some("mxfp8-v1")
        );
        // Undescribed fp8 (the plain UNETLoader cast) is the scalar row at unit scale.
        assert_eq!(
            registry
                .for_encoding(WeightEncoding::Fp8E4M3)
                .map(|codec| codec.codec_id),
            Some("fp8-e4m3-scalar-v1")
        );
        assert_eq!(
            registry
                .for_format(StoredTensorFormat::described(
                    WeightEncoding::Int8,
                    ComfyQuantFormat::Int8TensorwisePerRow
                ))
                .map(|codec| codec.codec_id),
            Some("int8-per-row-v1")
        );
        // Described int8 without its codec registered is not silently the dense row.
        assert_eq!(registry.for_encoding(WeightEncoding::Int8), None);
    }

    #[test]
    fn every_safetensors_dtype_classifies_to_exactly_one_encoding_with_its_width() {
        // The per-variant expectation table. Iterating `Dtype::ALL` below (rather than this table)
        // is what makes a newly added `Dtype` variant fail here instead of silently classifying to
        // `None`: an unlisted variant has no expectation and the lookup panics by name.
        let expected: BTreeMap<Dtype, (WeightEncoding, u64)> = [
            (Dtype::BOOL, WeightEncoding::Bool, 1),
            (Dtype::U8, WeightEncoding::UInt8, 1),
            (Dtype::I8, WeightEncoding::Int8, 1),
            (Dtype::F8_E5M2, WeightEncoding::Fp8E5M2, 1),
            (Dtype::F8_E4M3, WeightEncoding::Fp8E4M3, 1),
            (Dtype::I16, WeightEncoding::Int16, 2),
            (Dtype::U16, WeightEncoding::UInt16, 2),
            (Dtype::F16, WeightEncoding::DenseF16, 2),
            (Dtype::BF16, WeightEncoding::DenseBf16, 2),
            (Dtype::I32, WeightEncoding::Int32, 4),
            (Dtype::U32, WeightEncoding::UInt32, 4),
            (Dtype::F32, WeightEncoding::DenseF32, 4),
            (Dtype::F64, WeightEncoding::DenseF64, 8),
            (Dtype::I64, WeightEncoding::Int64, 8),
            (Dtype::U64, WeightEncoding::UInt64, 8),
        ]
        .into_iter()
        .map(|(dtype, encoding, width)| (dtype, (encoding, width)))
        .collect();

        for &dtype in Dtype::ALL {
            // E8M0 is a companion-only dtype: never a weight element encoding. It is the one
            // deliberate `None`, and it is named here rather than left as an unlisted gap.
            if dtype == Dtype::F8_E8M0 {
                assert_eq!(WeightEncoding::from_dtype(dtype), None, "{dtype:?}");
                continue;
            }
            let &(encoding, width) = expected.get(&dtype).unwrap_or_else(|| {
                panic!(
                    "{dtype:?} is a `Dtype` variant with no expectation in this test: classify it \
                     to a `WeightEncoding` (or add it to the E8M0 companion-only exemption) \
                     instead of letting it fall to `None` unnoticed"
                )
            });
            assert_eq!(
                WeightEncoding::from_dtype(dtype),
                Some(encoding),
                "{dtype:?}"
            );
            assert_eq!(encoding.element_bytes(), width, "{encoding:?}");
        }
        // Every expectation was reached: the table has no rows `Dtype::ALL` cannot name.
        assert_eq!(expected.len() + 1, Dtype::ALL.len());
    }

    #[test]
    fn identity_mapping_keeps_keys_and_receipt_sums_codec_residency() {
        assert_eq!(IdentityKeyMapping.mapping_id(), "identity-v1");
        assert_eq!(
            IdentityKeyMapping.logical_key("transformer_blocks.0.attn.to_q.weight"),
            Some("transformer_blocks.0.attn.to_q.weight".to_owned())
        );
        assert_eq!(IdentityKeyMapping.logical_shape("anything"), None);
        let receipt = LogicalWeightReceipt {
            mapping_id: "identity-v1",
            tensor_count: 2,
            source_bytes: 10,
            materialization: LogicalReadMaterialization::Materialized,
            residency: vec![
                CodecResidencyReport {
                    codec_id: "a",
                    tensor_count: 1,
                    source_bytes: 4,
                    resident_bytes: 4,
                },
                CodecResidencyReport {
                    codec_id: "b",
                    tensor_count: 1,
                    source_bytes: 6,
                    resident_bytes: 12,
                },
            ],
        };
        assert_eq!(receipt.resident_bytes(), 16);
    }

    // ---- sc-20385: descriptor-gated per-layer plans ---------------------------------------------

    fn descriptor_blob(json: &str) -> (SafetensorsTensorHeader, Vec<u8>) {
        (
            header_raw("", Dtype::U8, &[json.len()]),
            json.as_bytes().to_vec(),
        )
    }

    fn header_raw(name: &str, dtype: Dtype, shape: &[usize]) -> SafetensorsTensorHeader {
        header(name, dtype, shape)
    }

    /// A mixed checkpoint: dense bf16 + dense f32 + scalar e4m3 (with input_scale) + scalar e5m2
    /// (full-precision-matmul) + mxfp8 + int8-per-row + a plain undescribed fp8 cast — one file,
    /// seven decode routes, dispatched per layer.
    fn mixed_fixture() -> (Vec<SafetensorsTensorHeader>, BTreeMap<String, Vec<u8>>) {
        let mut headers = vec![
            header("model.dense.weight", Dtype::BF16, &[2, 4]),
            header("model.norm.bias", Dtype::F32, &[4]),
            // scalar e4m3
            header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
            header("model.q.weight_scale", Dtype::F32, &[]),
            header("model.q.input_scale", Dtype::F32, &[1]),
            header("model.q.comfy_quant", Dtype::U8, &[27]),
            // scalar e5m2, fpmm
            header("model.k.weight", Dtype::F8_E5M2, &[4, 8]),
            header("model.k.weight_scale", Dtype::F32, &[1]),
            header("model.k.comfy_quant", Dtype::U8, &[63]),
            // mxfp8: stored [32, 64], scales swizzled [128, 4]... scale shape below
            header("model.v.weight", Dtype::F8_E4M3, &[32, 64]),
            header("model.v.weight_scale", Dtype::U8, &[128, 4]),
            header("model.v.comfy_quant", Dtype::U8, &[19]),
            // int8 per-row
            header("model.o.weight", Dtype::I8, &[4, 8]),
            header("model.o.weight_scale", Dtype::F32, &[4, 1]),
            header("model.o.comfy_quant", Dtype::U8, &[46]),
            // plain fp8 cast, no descriptor, no companions
            header("model.p.weight", Dtype::F8_E4M3, &[2, 2]),
        ];
        // Fix descriptor payload lengths to match the JSON below.
        let mut descriptors = BTreeMap::new();
        for (key, json) in [
            ("model.q.comfy_quant", r#"{"format": "float8_e4m3fn"}"#),
            (
                "model.k.comfy_quant",
                r#"{"format": "float8_e5m2", "full_precision_matrix_mult": true}"#,
            ),
            ("model.v.comfy_quant", r#"{"format": "mxfp8"}"#),
            (
                "model.o.comfy_quant",
                r#"{"format": "int8_tensorwise", "per_row": true}"#,
            ),
        ] {
            let (mut blob_header, payload) = descriptor_blob(json);
            blob_header.name = key.to_owned();
            let slot = headers
                .iter_mut()
                .find(|header| header.name == key)
                .unwrap();
            *slot = blob_header;
            descriptors.insert(key.to_owned(), payload);
        }
        (headers, descriptors)
    }

    #[test]
    fn mixed_checkpoint_dispatches_per_layer_and_accounts_every_companion() {
        let (headers, descriptors) = mixed_fixture();
        let plan = compile(&headers, &descriptors, &StripPrefix, &full()).unwrap();
        assert_eq!(plan.tensor_count(), 7);
        assert_eq!(
            plan.codec_ids(),
            [
                "dense-bf16-v1",
                "dense-f32-v1",
                "fp8-e4m3-scalar-v1",
                "fp8-e5m2-scalar-v1",
                "int8-per-row-v1",
                "mxfp8-v1",
            ]
        );
        let by_logical: BTreeMap<&str, &LogicalTensorPlan> = plan
            .tensors
            .iter()
            .map(|tensor| (tensor.logical_key.as_str(), tensor))
            .collect();
        assert_eq!(by_logical["dense.weight"].codec, TensorCodecSpec::Dense);
        assert_eq!(
            by_logical["q.weight"].codec,
            TensorCodecSpec::ScalarFp8 {
                scale: ScalarScaleSource::Companion {
                    physical_key: "model.q.weight_scale".to_owned()
                },
                input_scale: Some("model.q.input_scale".to_owned()),
                full_precision_matrix_mult: false,
            }
        );
        assert!(matches!(
            &by_logical["k.weight"].codec,
            TensorCodecSpec::ScalarFp8 {
                full_precision_matrix_mult: true,
                ..
            }
        ));
        assert_eq!(
            by_logical["v.weight"].codec,
            TensorCodecSpec::Mxfp8 {
                scale: "model.v.weight_scale".to_owned(),
                stored_shape: [32, 64],
                logical_shape_declared: false,
                full_precision_matrix_mult: false,
            }
        );
        assert!(matches!(
            &by_logical["o.weight"].codec,
            TensorCodecSpec::Int8PerRow { scale, .. } if scale == "model.o.weight_scale"
        ));
        assert_eq!(
            by_logical["p.weight"].codec,
            TensorCodecSpec::ScalarFp8 {
                scale: ScalarScaleSource::Unit,
                input_scale: None,
                full_precision_matrix_mult: false,
            }
        );
        // Dense residency prices every quantized layer at bf16 of its logical shape.
        assert_eq!(by_logical["q.weight"].residency.resident_bytes, 4 * 8 * 2);
        assert_eq!(by_logical["v.weight"].residency.resident_bytes, 32 * 64 * 2);
        assert_eq!(by_logical["o.weight"].residency.resident_bytes, 4 * 8 * 2);
        assert_eq!(
            by_logical["dense.weight"].residency.resident_bytes,
            2 * 4 * 2
        );
        // Companions: 4 descriptors + 4 weight scales + 1 input scale, all consumed (0 resident).
        assert_eq!(plan.companions.len(), 9);
        assert!(plan
            .companions
            .iter()
            .all(|companion| companion.resident_bytes == 0));
        // Every byte of the file is accounted for.
        let declared: u64 = headers.iter().map(|header| header.data_bytes).sum();
        assert_eq!(plan.source_bytes, declared);
        let weights: u64 = plan.tensors.iter().map(|tensor| tensor.source_bytes).sum();
        let companions: u64 = plan
            .companions
            .iter()
            .map(|companion| companion.source_bytes)
            .sum();
        assert_eq!(weights + companions, declared);
        // The physical key surface is exact: weights + companions = the file's key set.
        let mut all: Vec<&str> = plan.all_physical_keys().collect();
        all.sort_unstable();
        let mut file: Vec<&str> = headers.iter().map(|header| header.name.as_str()).collect();
        file.sort_unstable();
        assert_eq!(all, file);
    }

    /// sc-20385 review: the shared MXFP8 / int8-per-row companion arm recorded `input_scale` at
    /// **zero** resident bytes unconditionally, while the ScalarFp8 arm branched on the residency
    /// mode, so a packed MXFP8 row under-priced its retained activation scale. Both arms now use
    /// the same `match mode` rule.
    ///
    /// Only the MXFP8 half of that arm can actually carry an `input_scale`: the companion-surface
    /// check refuses one on an `int8_tensorwise` layer by format (asserted below), so the int8 row
    /// here contributes its `weight_scale` only.
    #[test]
    fn packed_mxfp8_rows_retain_their_input_scale_companion() {
        struct PackEverything;
        impl CodecResidencyPolicy for PackEverything {
            fn residency(
                &self,
                _codec: &CheckpointCodecRegistration,
                _spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                ResidencyMode::Packed
            }
        }
        // Both codec rows that share the arm; the MXFP8 one carries the activation scale.
        let mut headers = vec![
            header("model.o.weight", Dtype::I8, &[4, 8]),
            header("model.o.weight_scale", Dtype::F32, &[4]),
            header("model.o.comfy_quant", Dtype::U8, &[1]),
            header("model.v.weight", Dtype::F8_E4M3, &[32, 64]),
            header("model.v.weight_scale", Dtype::U8, &[128, 4]),
            header("model.v.input_scale", Dtype::F32, &[1]),
            header("model.v.comfy_quant", Dtype::U8, &[1]),
        ];
        let mut descriptors = BTreeMap::new();
        for (key, json) in [
            (
                "model.o.comfy_quant",
                r#"{"format": "int8_tensorwise", "per_row": true}"#,
            ),
            ("model.v.comfy_quant", r#"{"format": "mxfp8"}"#),
        ] {
            let (mut blob_header, payload) = descriptor_blob(json);
            blob_header.name = key.to_owned();
            *headers
                .iter_mut()
                .find(|header| header.name == key)
                .unwrap() = blob_header;
            descriptors.insert(key.to_owned(), payload);
        }

        let packed = compile_logical_weight_plan(
            &headers,
            &descriptors,
            &StripPrefix,
            &full(),
            &PackEverything,
        )
        .unwrap();
        let companion = |plan: &LogicalWeightPlan, key: &str| {
            plan.companions
                .iter()
                .find(|companion| companion.physical_key == key)
                .unwrap()
                .resident_bytes
        };
        // Retained at its stored width — 4 bytes, not zero.
        assert_eq!(companion(&packed, "model.v.input_scale"), 4);
        // And the weight scales alongside it, so the input-scale row is the only thing this
        // fixture could be getting wrong.
        assert_eq!(companion(&packed, "model.o.weight_scale"), 16);
        assert_eq!(companion(&packed, "model.v.weight_scale"), 512);
        // The plan total accounts for every retained byte.
        assert_eq!(packed.resident_bytes(), 32 + 16 + 32 * 64 + 512 + 4);

        // Under the dense policy the same companion is consumed, so the assertion above is about
        // the packed branch, not about the fixture always pricing scales.
        let dense = compile(&headers, &descriptors, &StripPrefix, &full()).unwrap();
        assert_eq!(companion(&dense, "model.v.input_scale"), 0);

        // The int8 half of the shared arm cannot reach the branch at all: an `input_scale` on an
        // `int8_tensorwise` layer is refused by format before residency is decided.
        let mut with_int8_input_scale = headers.clone();
        with_int8_input_scale.push(header("model.o.input_scale", Dtype::F32, &[]));
        assert!(matches!(
            compile(
                &with_int8_input_scale,
                &descriptors,
                &StripPrefix,
                &full()
            ),
            Err(LogicalWeightPlanError::UnexpectedCompanion { physical_key, .. })
                if physical_key == "model.o.input_scale"
        ));
    }

    /// A packed-selecting policy prices packed layers at stored + retained scales, EXCEPT layers
    /// flagged `full_precision_matrix_mult`, which stay dense by contract. Mutation partner:
    /// `residency-pricing` — flipping packed pricing to dense bytes (or ignoring the fpmm flag)
    /// turns this red.
    #[test]
    fn packed_policy_prices_stored_bytes_plus_scales_and_honors_full_precision_layers() {
        struct PackFp8;
        impl CodecResidencyPolicy for PackFp8 {
            fn residency(
                &self,
                codec: &CheckpointCodecRegistration,
                _spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                if codec.codec_id == "fp8-e4m3-scalar-v1" {
                    ResidencyMode::Packed
                } else {
                    ResidencyMode::Dense
                }
            }
        }
        let (headers, descriptors) = mixed_fixture();
        let plan =
            compile_logical_weight_plan(&headers, &descriptors, &StripPrefix, &full(), &PackFp8)
                .unwrap();
        let by_logical: BTreeMap<&str, &LogicalTensorPlan> = plan
            .tensors
            .iter()
            .map(|tensor| (tensor.logical_key.as_str(), tensor))
            .collect();
        // Packed e4m3: stored bytes stay resident; its scale companions are retained.
        assert_eq!(
            by_logical["q.weight"].residency,
            PlannedResidency {
                mode: ResidencyMode::Packed,
                resident_bytes: 4 * 8
            }
        );
        let companion = |key: &str| {
            plan.companions
                .iter()
                .find(|companion| companion.physical_key == key)
                .unwrap()
        };
        assert_eq!(companion("model.q.weight_scale").resident_bytes, 4);
        assert_eq!(companion("model.q.input_scale").resident_bytes, 4);
        // The plain cast has no companions; packed keeps its 4 stored bytes.
        assert_eq!(
            by_logical["p.weight"].residency,
            PlannedResidency {
                mode: ResidencyMode::Packed,
                resident_bytes: 4
            }
        );
        // e5m2 is not this policy's packed codec → dense; and even if it were, its
        // full_precision_matrix_mult flag forces dense.
        assert_eq!(by_logical["k.weight"].residency.mode, ResidencyMode::Dense);
        struct PackEverything;
        impl CodecResidencyPolicy for PackEverything {
            fn residency(
                &self,
                _codec: &CheckpointCodecRegistration,
                _spec: &TensorCodecSpec,
                _stored_shape: &[usize],
            ) -> ResidencyMode {
                ResidencyMode::Packed
            }
        }
        let plan = compile_logical_weight_plan(
            &headers,
            &descriptors,
            &StripPrefix,
            &full(),
            &PackEverything,
        )
        .unwrap();
        let k = plan
            .tensors
            .iter()
            .find(|tensor| tensor.logical_key == "k.weight")
            .unwrap();
        assert_eq!(
            k.residency.mode,
            ResidencyMode::Dense,
            "a full_precision_matrix_mult layer never runs packed, whatever the policy answers"
        );
        assert_eq!(k.residency.resident_bytes, 4 * 8 * 2);
    }

    #[test]
    fn malformed_descriptors_name_the_exact_layer_and_defect() {
        let make = |descriptor_json: &str| {
            let mut headers = vec![
                header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
                header("model.q.weight_scale", Dtype::F32, &[]),
                header("model.q.comfy_quant", Dtype::U8, &[descriptor_json.len()]),
            ];
            headers.rotate_left(1);
            let descriptors: BTreeMap<String, Vec<u8>> = [(
                "model.q.comfy_quant".to_owned(),
                descriptor_json.as_bytes().to_vec(),
            )]
            .into();
            compile(&headers, &descriptors, &StripPrefix, &full())
        };
        let error = make(r#"{"format": "int4_awq"}"#).unwrap_err();
        assert_eq!(
            error,
            LogicalWeightPlanError::Descriptor {
                layer: "model.q".to_owned(),
                physical_key: "model.q.comfy_quant".to_owned(),
                defect: ComfyQuantDescriptorError::UnsupportedFormat {
                    format: "int4_awq".to_owned()
                },
            }
        );
        assert!(error.to_string().contains("model.q"), "{error}");
        let error = make(r#"{"format": "float8_e4m3fn", "group_size": 64}"#).unwrap_err();
        assert!(
            matches!(
                &error,
                LogicalWeightPlanError::Descriptor {
                    layer,
                    defect: ComfyQuantDescriptorError::UnknownField { field },
                    ..
                } if layer == "model.q" && field == "group_size"
            ),
            "{error:?}"
        );
        let error = make("not json").unwrap_err();
        assert!(
            matches!(
                &error,
                LogicalWeightPlanError::Descriptor {
                    defect: ComfyQuantDescriptorError::NotJson { .. },
                    ..
                }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn descriptor_and_companion_surface_defects_refuse_by_tensor() {
        let full = full();
        // Descriptor tensor that is not rank-1 U8.
        let bad_blob = [
            header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
            header("model.q.comfy_quant", Dtype::F32, &[4]),
        ];
        assert_eq!(
            compile(&bad_blob, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::DescriptorTensor {
                physical_key: "model.q.comfy_quant".to_owned()
            })
        );
        // Orphan descriptor (no weight).
        let orphan = [header("model.q.comfy_quant", Dtype::U8, &[2])];
        assert_eq!(
            compile(&orphan, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::OrphanDescriptor {
                physical_key: "model.q.comfy_quant".to_owned()
            })
        );
        // Missing payload for a well-formed descriptor tensor.
        let missing_payload = [
            header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
            header("model.q.comfy_quant", Dtype::U8, &[2]),
        ];
        assert_eq!(
            compile(&missing_payload, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::DescriptorPayloadUnavailable {
                physical_key: "model.q.comfy_quant".to_owned()
            })
        );
        // Scale companion with no descriptor = the FLUX.2 inline convention → refuse.
        let inline = [
            header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
            header("model.q.weight_scale", Dtype::F32, &[]),
        ];
        assert!(matches!(
            compile(&inline, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::UnexpectedCompanion { physical_key, .. })
                if physical_key == "model.q.weight_scale"
        ));
        // Legacy scaled_fp8 convention → its own refusal.
        let legacy = [
            header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
            header("model.q.scale_weight", Dtype::F32, &[]),
        ];
        assert_eq!(
            compile(&legacy, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::LegacyScaledFp8 {
                physical_key: "model.q.scale_weight".to_owned()
            })
        );
        let marker = [
            header("model.q.weight", Dtype::BF16, &[4, 8]),
            header("scaled_fp8", Dtype::F8_E4M3, &[2]),
        ];
        assert!(matches!(
            compile(&marker, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::LegacyScaledFp8 { physical_key }) if physical_key == "scaled_fp8"
        ));
        // `weight_scale_2` is NVFP4's second scale level (sc-20641): without a descriptor it is an
        // undescribed companion, and its `U8` weight has no codec either way.
        let nvfp4 = [
            header("model.q.weight", Dtype::U8, &[4, 4]),
            header("model.q.weight_scale_2", Dtype::F32, &[]),
        ];
        assert!(matches!(
            compile(&nvfp4, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::UnsupportedFormat { physical_key, .. })
                if physical_key == "model.q.weight"
        ));
        // On a layer whose format is *not* nvfp4, the second scale level is refused by name.
        let stray = [
            header("model.q.weight", Dtype::BF16, &[4, 4]),
            header("model.q.weight_scale_2", Dtype::F32, &[]),
        ];
        assert!(matches!(
            compile(&stray, &no_descriptors(), &StripPrefix, &full),
            Err(LogicalWeightPlanError::UnexpectedCompanion { physical_key, .. })
                if physical_key == "model.q.weight_scale_2"
        ));

        let with_descriptor = |json: &str,
                               extra: &[SafetensorsTensorHeader]|
         -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
            let mut headers = vec![header("model.q.comfy_quant", Dtype::U8, &[json.len()])];
            headers.extend(extra.iter().cloned());
            let descriptors: BTreeMap<String, Vec<u8>> =
                [("model.q.comfy_quant".to_owned(), json.as_bytes().to_vec())].into();
            compile(&headers, &descriptors, &StripPrefix, &full)
        };
        // Declared e4m3 but stored bf16.
        assert_eq!(
            with_descriptor(
                r#"{"format": "float8_e4m3fn"}"#,
                &[
                    header("model.q.weight", Dtype::BF16, &[4, 8]),
                    header("model.q.weight_scale", Dtype::F32, &[]),
                ]
            ),
            Err(LogicalWeightPlanError::DescriptorDtypeMismatch {
                physical_key: "model.q.weight".to_owned(),
                format: ComfyQuantFormat::Float8E4M3Fn,
                dtype: "BF16".to_owned(),
            })
        );
        // Rank-1 quantized weight.
        assert_eq!(
            with_descriptor(
                r#"{"format": "float8_e4m3fn"}"#,
                &[
                    header("model.q.weight", Dtype::F8_E4M3, &[8]),
                    header("model.q.weight_scale", Dtype::F32, &[]),
                ]
            ),
            Err(LogicalWeightPlanError::QuantizedWeightRank {
                physical_key: "model.q.weight".to_owned(),
                format: ComfyQuantFormat::Float8E4M3Fn,
                shape: vec![8],
            })
        );
        // Missing weight_scale.
        assert_eq!(
            with_descriptor(
                r#"{"format": "float8_e4m3fn"}"#,
                &[header("model.q.weight", Dtype::F8_E4M3, &[4, 8])]
            ),
            Err(LogicalWeightPlanError::MissingCompanion {
                physical_key: "model.q.weight".to_owned(),
                companion: "model.q.weight_scale".to_owned(),
            })
        );
        // Vector scale on a scalar-fp8 layer.
        assert!(matches!(
            with_descriptor(
                r#"{"format": "float8_e4m3fn"}"#,
                &[
                    header("model.q.weight", Dtype::F8_E4M3, &[4, 8]),
                    header("model.q.weight_scale", Dtype::F32, &[4]),
                ]
            ),
            Err(LogicalWeightPlanError::CompanionMalformed { physical_key, .. })
                if physical_key == "model.q.weight_scale"
        ));
        // int8 scale wrong shape.
        assert!(matches!(
            with_descriptor(
                r#"{"format": "int8_tensorwise", "per_row": true}"#,
                &[
                    header("model.q.weight", Dtype::I8, &[4, 8]),
                    header("model.q.weight_scale", Dtype::F32, &[8]),
                ]
            ),
            Err(LogicalWeightPlanError::CompanionMalformed { .. })
        ));
        // int8 with an input_scale → not part of that format.
        assert!(matches!(
            with_descriptor(
                r#"{"format": "int8_tensorwise", "per_row": true}"#,
                &[
                    header("model.q.weight", Dtype::I8, &[4, 8]),
                    header("model.q.weight_scale", Dtype::F32, &[4, 1]),
                    header("model.q.input_scale", Dtype::F32, &[]),
                ]
            ),
            Err(LogicalWeightPlanError::UnexpectedCompanion { physical_key, .. })
                if physical_key == "model.q.input_scale"
        ));
        // MXFP8 geometry: unpadded stored shape / wrong scale shape / wrong scale dtype.
        assert!(matches!(
            with_descriptor(
                r#"{"format": "mxfp8"}"#,
                &[
                    header("model.q.weight", Dtype::F8_E4M3, &[40, 70]),
                    header("model.q.weight_scale", Dtype::U8, &[128, 4]),
                ]
            ),
            Err(LogicalWeightPlanError::Mxfp8Geometry {
                error: Mxfp8GeometryError::StoredNotPadded { .. },
                ..
            })
        ));
        assert!(matches!(
            with_descriptor(
                r#"{"format": "mxfp8"}"#,
                &[
                    header("model.q.weight", Dtype::F8_E4M3, &[32, 64]),
                    header("model.q.weight_scale", Dtype::U8, &[32, 2]),
                ]
            ),
            Err(LogicalWeightPlanError::Mxfp8Geometry {
                error: Mxfp8GeometryError::ScaleShape { .. },
                ..
            })
        ));
        assert!(matches!(
            with_descriptor(
                r#"{"format": "mxfp8"}"#,
                &[
                    header("model.q.weight", Dtype::F8_E4M3, &[32, 64]),
                    header("model.q.weight_scale", Dtype::F32, &[128, 4]),
                ]
            ),
            Err(LogicalWeightPlanError::CompanionMalformed { physical_key, .. })
                if physical_key == "model.q.weight_scale"
        ));
    }

    struct DeclaredShape;

    impl LogicalKeyMapping for DeclaredShape {
        fn mapping_id(&self) -> &'static str {
            "declared-shape-test"
        }
        fn logical_key(&self, physical_key: &str) -> Option<String> {
            physical_key.strip_prefix("model.").map(str::to_owned)
        }
        fn logical_shape(&self, logical_key: &str) -> Option<Vec<usize>> {
            (logical_key == "q.weight").then(|| vec![5, 40])
        }
    }

    #[test]
    fn mxfp8_unpads_to_the_adapter_declared_logical_shape_and_checks_it_pads_back() {
        let headers = [
            header("model.q.weight", Dtype::F8_E4M3, &[32, 64]),
            header("model.q.weight_scale", Dtype::U8, &[128, 4]),
            header("model.q.comfy_quant", Dtype::U8, &[19]),
        ];
        let descriptors: BTreeMap<String, Vec<u8>> = [(
            "model.q.comfy_quant".to_owned(),
            br#"{"format": "mxfp8"}"#.to_vec(),
        )]
        .into();
        let plan = compile(&headers, &descriptors, &DeclaredShape, &full()).unwrap();
        let q = &plan.tensors[0];
        assert_eq!(q.shape, vec![5, 40]);
        assert!(matches!(
            &q.codec,
            TensorCodecSpec::Mxfp8 {
                stored_shape: [32, 64],
                logical_shape_declared: true,
                ..
            }
        ));
        // Dense residency prices the LOGICAL (unpadded) shape, not storage.
        assert_eq!(q.residency.resident_bytes, 5 * 40 * 2);

        // A declared shape that does not pad to the stored shape refuses.
        struct WrongShape;
        impl LogicalKeyMapping for WrongShape {
            fn mapping_id(&self) -> &'static str {
                "wrong-shape-test"
            }
            fn logical_key(&self, physical_key: &str) -> Option<String> {
                physical_key.strip_prefix("model.").map(str::to_owned)
            }
            fn logical_shape(&self, _logical_key: &str) -> Option<Vec<usize>> {
                Some(vec![5, 100])
            }
        }
        assert!(matches!(
            compile(&headers, &descriptors, &WrongShape, &full()),
            Err(LogicalWeightPlanError::Mxfp8Geometry {
                error: Mxfp8GeometryError::LogicalDoesNotPadToStored { .. },
                ..
            })
        ));
    }

    #[test]
    fn resident_tensor_headers_present_the_dense_fallback_form_for_pricing() {
        let (headers, descriptors) = mixed_fixture();
        let plan = compile(&headers, &descriptors, &StripPrefix, &full()).unwrap();
        let resident = plan.resident_tensor_headers();
        let by_name: BTreeMap<&str, &SafetensorsTensorHeader> = resident
            .iter()
            .map(|header| (header.name.as_str(), header))
            .collect();
        // fp8 → bf16 at the logical shape.
        assert_eq!(by_name["q.weight"].dtype, Dtype::BF16);
        assert_eq!(by_name["q.weight"].shape, vec![4, 8]);
        assert_eq!(by_name["q.weight"].data_bytes, 4 * 8 * 2);
        // dense f32 stays f32.
        assert_eq!(by_name["norm.bias"].dtype, Dtype::F32);
        assert_eq!(by_name["norm.bias"].data_bytes, 16);
        // The header sum is the plan's resident bytes (all companions consumed under dense).
        let sum: u64 = resident.iter().map(|header| header.data_bytes).sum();
        assert_eq!(sum, plan.resident_bytes());
    }

    /// sc-20385 review: `resident_tensor_headers` used to re-find the codec row by scanning the two
    /// hardcoded const tables and, when the scan missed, fell back to the tensor's **stored**
    /// encoding. A codec registered outside those tables therefore synthesized a pricing header
    /// whose dtype contradicted its own `data_bytes` — and the dtype is what the Q4/Q8 projections
    /// read. The plan now carries `resident_encoding` from the registry row it compiled against.
    #[test]
    fn a_codec_outside_the_const_tables_prices_from_its_own_resident_encoding() {
        // Stored int8, dense fallback f32: divergent from the stored encoding (Int8) *and* from
        // every built-in row (`int8-per-row-v1` leaves bf16), so neither the const scan nor the
        // stored-encoding fallback can accidentally produce the right answer.
        const SYNTHETIC: CheckpointCodecRegistration = CheckpointCodecRegistration {
            codec_id: "synthetic-int8-to-f32-v1",
            stored: &[StoredTensorFormat::undescribed(WeightEncoding::Int8)],
            resident_encoding: WeightEncoding::DenseF32,
        };
        let registry =
            CheckpointCodecRegistry::new(DENSE_CODECS.iter().copied().chain([SYNTHETIC])).unwrap();
        let headers = [header("model.w.weight", Dtype::I8, &[4, 8])];
        let plan = compile(&headers, &no_descriptors(), &StripPrefix, &registry).unwrap();
        let tensor = &plan.tensors[0];
        assert_eq!(tensor.codec_id, SYNTHETIC.codec_id);
        assert_eq!(tensor.resident_encoding, WeightEncoding::DenseF32);
        assert_eq!(tensor.residency.resident_bytes, 4 * 8 * 4);

        let resident = plan.resident_tensor_headers();
        assert_eq!(resident.len(), 1);
        assert_eq!(resident[0].dtype, Dtype::F32);
        // The synthesized header must be internally consistent: dtype width × logical elements is
        // the byte count it reports. The old const scan produced I8 here against 128 bytes.
        let elements: usize = resident[0].shape.iter().product();
        assert_eq!(
            (elements * resident[0].dtype.size()) as u64,
            resident[0].data_bytes
        );
        assert_eq!(resident[0].data_bytes, 4 * 8 * 4);
    }
}

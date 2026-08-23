//! Backend-neutral checkpoint **codec registry** and **mapped logical-weight plan** (epic 20398,
//! sc-20634).
//!
//! Two tensor-free halves that every backend reader consumes the same way:
//!
//! 1. A [`CheckpointCodecRegistration`] declares how one stored tensor encoding becomes resident
//!    weights. Codecs register **once**, at engine/catalog level, and are keyed by
//!    [`WeightEncoding`]; a family adapter never carries its own codec table (that was the
//!    duplicate-row defect of the withdrawn sc-20638 draft), so adding a codec never touches an
//!    adapter and adding an adapter never re-registers a codec.
//! 2. A [`LogicalWeightPlan`] is compiled from a checkpoint's **header only**
//!    ([`SafetensorsTensorHeader`]) plus the family adapter's [`LogicalKeyMapping`] and the codec
//!    registry. It fails closed before any backend array exists: an unmapped physical key, two
//!    physical keys mapping onto one logical key, or a stored encoding with no registered codec are
//!    typed [`LogicalWeightPlanError`]s. The backend reader then materializes exactly the planned
//!    tensors and returns a [`LogicalWeightReceipt`] whose resident bytes are measured from the
//!    decoded arrays, never copied from the header.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::weightsmeta::{Dtype, SafetensorsTensorHeader};

/// How a tensor is stored on disk (or, for [`CheckpointCodecRegistration::resident_encoding`],
/// how a codec leaves it resident). Classified from the safetensors header dtype; descriptor-gated
/// packings (ComfyUI `.comfy_quant`, ConvRot) are deliberately **not** modelled here — they need
/// the per-layer descriptor plans of sc-20385 and keep their existing bespoke loader arms until
/// those codecs register.
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
    /// Classify one stored safetensors dtype. Every dtype the `safetensors` crate can name maps to
    /// exactly one encoding; a future dtype the pinned crate does not know cannot reach here because
    /// the header parser rejects it first.
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

/// One registered codec: the stored encoding it claims and the encoding it leaves resident.
///
/// The portable row is the declaration; the backend engine that registers it owns the matching
/// implementation, and the engine's conformance test proves every registered row has one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointCodecRegistration {
    pub codec_id: &'static str,
    pub encoding: WeightEncoding,
    pub resident_encoding: WeightEncoding,
}

/// The baseline dense bf16 codec: stored bf16 stays bf16, byte-for-byte (no cast, no remap).
pub const DENSE_BF16_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-bf16-v1",
    encoding: WeightEncoding::DenseBf16,
    resident_encoding: WeightEncoding::DenseBf16,
};

/// Dense f16 pass-through (stored f16 stays f16).
pub const DENSE_F16_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-f16-v1",
    encoding: WeightEncoding::DenseF16,
    resident_encoding: WeightEncoding::DenseF16,
};

/// Dense f32 pass-through (stored f32 stays f32). A "dense bf16" community DiT routinely keeps a
/// handful of embedding/projection biases and norms in f32 (e.g. `kreamania_variant5`: 415 bf16 +
/// 15 f32 tensors); those tensors decode through this row and report their own resident bytes
/// rather than being cast or silently folded into the bf16 report.
pub const DENSE_F32_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "dense-f32-v1",
    encoding: WeightEncoding::DenseF32,
    resident_encoding: WeightEncoding::DenseF32,
};

/// The portable dense pass-through codec rows every backend can implement without a kernel.
pub const DENSE_CODECS: &[CheckpointCodecRegistration] =
    &[DENSE_BF16_CODEC, DENSE_F16_CODEC, DENSE_F32_CODEC];

/// Validated, immutable set of codecs keyed by stored encoding. Built by
/// [`CheckpointCodecRegistry::new`] (and by `ProviderRegistryBuilder::build` from the rows a catalog
/// registers); duplicate ids or two codecs claiming one encoding fail closed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointCodecRegistry {
    by_encoding: BTreeMap<WeightEncoding, CheckpointCodecRegistration>,
}

impl CheckpointCodecRegistry {
    pub fn new(
        codecs: impl IntoIterator<Item = CheckpointCodecRegistration>,
    ) -> crate::Result<Self> {
        let mut ids = BTreeSet::new();
        let mut by_encoding = BTreeMap::new();
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
            if let Some(previous) = by_encoding.insert(codec.encoding, codec) {
                return Err(crate::Error::Msg(format!(
                    "checkpoint codecs '{}' and '{}' both claim stored encoding {}",
                    previous.codec_id,
                    codec.codec_id,
                    codec.encoding.label()
                )));
            }
        }
        Ok(Self { by_encoding })
    }

    pub fn codecs(&self) -> impl ExactSizeIterator<Item = &CheckpointCodecRegistration> {
        self.by_encoding.values()
    }

    pub fn is_empty(&self) -> bool {
        self.by_encoding.is_empty()
    }

    pub fn len(&self) -> usize {
        self.by_encoding.len()
    }

    /// The codec registered for a stored encoding, or `None` (the caller refuses).
    pub fn for_encoding(&self, encoding: WeightEncoding) -> Option<&CheckpointCodecRegistration> {
        self.by_encoding.get(&encoding)
    }

    pub fn by_id(&self, codec_id: &str) -> Option<&CheckpointCodecRegistration> {
        self.by_encoding
            .values()
            .find(|codec| codec.codec_id == codec_id)
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
}

/// The identity mapping for checkpoints already stored under canonical keys.
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

/// One planned tensor: where it lives on disk, what it is called logically, how it is decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTensorPlan {
    pub logical_key: String,
    pub physical_key: String,
    pub encoding: WeightEncoding,
    pub shape: Vec<usize>,
    /// Bytes the tensor occupies in the source file.
    pub source_bytes: u64,
    pub codec_id: &'static str,
}

/// The complete mapped read plan for one safetensors file, sorted by logical key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalWeightPlan {
    pub mapping_id: &'static str,
    pub tensors: Vec<LogicalTensorPlan>,
    pub source_bytes: u64,
}

impl LogicalWeightPlan {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn physical_keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.physical_key.as_str())
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
    /// The stored dtype is not one this contract can classify.
    UnclassifiedDtype { physical_key: String, dtype: String },
    /// The stored encoding has no registered codec on this backend.
    UnsupportedEncoding {
        physical_key: String,
        encoding: WeightEncoding,
    },
    /// The tensor's declared shape and byte length disagree with its encoding.
    GeometryMismatch {
        physical_key: String,
        encoding: WeightEncoding,
        declared_bytes: u64,
        expected_bytes: u64,
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
                "on-disk tensor {physical_key:?} uses dtype {dtype} that this contract cannot classify"
            ),
            Self::UnsupportedEncoding {
                physical_key,
                encoding,
            } => write!(
                f,
                "on-disk tensor {physical_key:?} is stored as {} and no checkpoint codec is registered for that encoding on this backend",
                encoding.label()
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
        }
    }
}

impl std::error::Error for LogicalWeightPlanError {}

impl From<LogicalWeightPlanError> for crate::Error {
    fn from(error: LogicalWeightPlanError) -> Self {
        crate::Error::Msg(format!("logical weight plan: {error}"))
    }
}

/// Compile a header into a [`LogicalWeightPlan`]. Pure and deterministic: the same header, mapping,
/// and codec registry always produce the same plan, in logical-key order. Fails on the first tensor
/// (in header order, sorted by physical key) that cannot be planned.
pub fn compile_logical_weight_plan(
    headers: &[SafetensorsTensorHeader],
    mapping: &dyn LogicalKeyMapping,
    codecs: &CheckpointCodecRegistry,
) -> Result<LogicalWeightPlan, LogicalWeightPlanError> {
    if headers.is_empty() {
        return Err(LogicalWeightPlanError::EmptyCheckpoint);
    }
    let mut ordered: Vec<&SafetensorsTensorHeader> = headers.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));

    let mut owners: BTreeMap<String, String> = BTreeMap::new();
    let mut tensors = Vec::with_capacity(ordered.len());
    let mut source_bytes = 0_u64;
    for header in ordered {
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
        let codec =
            codecs
                .for_encoding(encoding)
                .ok_or(LogicalWeightPlanError::UnsupportedEncoding {
                    physical_key: header.name.clone(),
                    encoding,
                })?;
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
        source_bytes = source_bytes.saturating_add(header.data_bytes);
        tensors.push(LogicalTensorPlan {
            logical_key,
            physical_key: header.name.clone(),
            encoding,
            shape: header.shape.clone(),
            source_bytes: header.data_bytes,
            codec_id: codec.codec_id,
        });
    }
    tensors.sort_by(|left, right| left.logical_key.cmp(&right.logical_key));
    Ok(LogicalWeightPlan {
        mapping_id: mapping.mapping_id(),
        tensors,
        source_bytes,
    })
}

/// What one codec left resident after a read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodecResidencyReport {
    pub codec_id: &'static str,
    pub tensor_count: usize,
    /// Bytes read from the source file for these tensors (from the plan).
    pub source_bytes: u64,
    /// Bytes the decoded tensors occupy resident, measured from the backend arrays after decode.
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
        let width = WeightEncoding::from_dtype(dtype).unwrap().element_bytes() as usize;
        SafetensorsTensorHeader {
            name: name.to_owned(),
            dtype,
            shape: shape.to_vec(),
            data_bytes: (elements * width) as u64,
        }
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

    #[test]
    fn plan_is_a_sorted_bijection_with_source_bytes_and_codec_ids() {
        let headers = [
            header("model.b.weight", Dtype::BF16, &[2, 3]),
            header("model.a.weight", Dtype::BF16, &[4]),
        ];
        let plan = compile_logical_weight_plan(&headers, &StripPrefix, &baseline()).unwrap();
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
        // Deterministic regardless of header order.
        let reversed = [headers[1].clone(), headers[0].clone()];
        assert_eq!(
            compile_logical_weight_plan(&reversed, &StripPrefix, &baseline()).unwrap(),
            plan
        );
    }

    #[test]
    fn plan_refuses_unmapped_keys_collisions_and_unregistered_encodings_by_tensor() {
        let unmapped = [header("foreign.weight", Dtype::BF16, &[1])];
        assert_eq!(
            compile_logical_weight_plan(&unmapped, &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::UnmappedKey {
                physical_key: "foreign.weight".to_owned()
            })
        );
        let colliding = [
            header("model.x", Dtype::BF16, &[1]),
            header("model.y", Dtype::BF16, &[1]),
        ];
        assert_eq!(
            compile_logical_weight_plan(&colliding, &Collapse, &baseline()),
            Err(LogicalWeightPlanError::KeyCollision {
                logical_key: "same".to_owned(),
                first_physical_key: "model.x".to_owned(),
                second_physical_key: "model.y".to_owned(),
            })
        );
        let fp8 = [
            header("model.ok", Dtype::BF16, &[1]),
            header("model.packed", Dtype::F8_E4M3, &[4]),
        ];
        assert_eq!(
            compile_logical_weight_plan(&fp8, &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::UnsupportedEncoding {
                physical_key: "model.packed".to_owned(),
                encoding: WeightEncoding::Fp8E4M3,
            })
        );
        assert_eq!(
            compile_logical_weight_plan(&[], &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::EmptyCheckpoint)
        );
    }

    #[test]
    fn plan_refuses_declared_bytes_that_disagree_with_shape_and_encoding() {
        let mut short = header("model.w", Dtype::BF16, &[3]);
        short.data_bytes = 5;
        assert_eq!(
            compile_logical_weight_plan(&[short], &StripPrefix, &baseline()),
            Err(LogicalWeightPlanError::GeometryMismatch {
                physical_key: "model.w".to_owned(),
                encoding: WeightEncoding::DenseBf16,
                declared_bytes: 5,
                expected_bytes: 6,
            })
        );
    }

    #[test]
    fn codec_registry_rejects_duplicate_ids_and_double_claimed_encodings() {
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
                encoding: WeightEncoding::DenseBf16,
                resident_encoding: WeightEncoding::DenseF32,
            },
        ]);
        assert!(
            double_claim
                .as_ref()
                .err()
                .is_some_and(|error| error.to_string().contains("both claim stored encoding")),
            "{double_claim:?}"
        );
        let malformed = CheckpointCodecRegistry::new([CheckpointCodecRegistration {
            codec_id: "Dense BF16",
            ..DENSE_BF16_CODEC
        }]);
        assert!(malformed.is_err());
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
    fn every_safetensors_dtype_classifies_to_exactly_one_encoding_with_its_width() {
        for (dtype, encoding, width) in [
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
        ] {
            assert_eq!(
                WeightEncoding::from_dtype(dtype),
                Some(encoding),
                "{dtype:?}"
            );
            assert_eq!(encoding.element_bytes(), width, "{encoding:?}");
        }
    }

    #[test]
    fn identity_mapping_keeps_keys_and_receipt_sums_codec_residency() {
        assert_eq!(IdentityKeyMapping.mapping_id(), "identity-v1");
        assert_eq!(
            IdentityKeyMapping.logical_key("transformer_blocks.0.attn.to_q.weight"),
            Some("transformer_blocks.0.attn.to_q.weight".to_owned())
        );
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
}

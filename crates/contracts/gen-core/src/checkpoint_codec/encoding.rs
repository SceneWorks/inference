//! Weight encodings, stored tensor formats, and the codec registrations and registry.

use super::*;

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
    /// A **GGUF container** payload: one ggml block-quantized (or dense F16/F32/BF16) tensor as it
    /// sits in a `.gguf` file (epic 20398, sc-20649).
    ///
    /// Deliberately unreachable from safetensors: [`WeightEncoding::from_dtype`] never returns it,
    /// so no safetensors header can route to the GGUF codec row, and
    /// [`element_bytes`](WeightEncoding::element_bytes) reports `0` because a block quant has no
    /// integral per-element width — a caller that tried to size a GGUF tensor the safetensors way
    /// gets `0` and fails closed on the byte-integrity check instead of a plausible-looking number.
    /// A GGUF plan producer looks the row up directly (`for_encoding(GgufContainer)`) and measures
    /// every byte from the container's own ggml block/type sizes — see [`GGUF_CONTAINER_CODEC`].
    GgufContainer,
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

    /// Bytes one element occupies when resident in this encoding. `0` for
    /// [`WeightEncoding::GgufContainer`], which is block-quantized and has no integral per-element
    /// width — see that variant's documentation.
    pub fn element_bytes(self) -> u64 {
        match self {
            Self::GgufContainer => 0,
            Self::Bool | Self::Int8 | Self::UInt8 | Self::Fp8E4M3 | Self::Fp8E5M2 => 1,
            Self::DenseBf16 | Self::DenseF16 | Self::Int16 | Self::UInt16 => 2,
            Self::DenseF32 | Self::Int32 | Self::UInt32 => 4,
            Self::DenseF64 | Self::Int64 | Self::UInt64 => 8,
        }
    }

    /// The safetensors dtype an element of this encoding is stored as — the exact inverse of
    /// [`Self::from_dtype`] on its `Some` domain.
    ///
    /// Exhaustive on purpose. The two byte-pricing projections in
    /// [`LogicalWeightPlan::resident_tensor_headers`] each need this map, and the packed one used to
    /// spell it as three arms plus `_ => Dtype::U8`. That wildcard is only correct because the
    /// packed rows happen to be fp8/int8/NVFP4-in-`U8` today; a codec that grows a packed row for
    /// any other encoding would have been silently re-labelled `U8` — right byte count, wrong dtype
    /// — in the headers admission prices from. Routing both arms through one total function makes
    /// that a compile error instead.
    pub fn to_dtype(self) -> Dtype {
        match self {
            Self::DenseBf16 => Dtype::BF16,
            Self::DenseF16 => Dtype::F16,
            Self::DenseF32 => Dtype::F32,
            Self::DenseF64 => Dtype::F64,
            Self::Fp8E4M3 => Dtype::F8_E4M3,
            Self::Fp8E5M2 => Dtype::F8_E5M2,
            Self::Int8 => Dtype::I8,
            Self::UInt8 => Dtype::U8,
            Self::Int16 => Dtype::I16,
            Self::Int32 => Dtype::I32,
            Self::Int64 => Dtype::I64,
            Self::UInt16 => Dtype::U16,
            Self::UInt32 => Dtype::U32,
            Self::UInt64 => Dtype::U64,
            Self::Bool => Dtype::BOOL,
            // GGUF is a different *container*: its tensors carry a ggml block type, not a
            // safetensors dtype, and a block quant has no integral bytes-per-element. `from_dtype`
            // can never produce this variant, so this arm exists only so the map stays total; the
            // opaque byte view is the honest projection of ggml blocks, and the `data_bytes` a
            // pricing consumer reads alongside it carries the measured size regardless.
            Self::GgufContainer => Dtype::U8,
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
            Self::GgufContainer => "gguf-container",
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

/// The **baseline engine codec table**: [`DENSE_CODECS`] followed by [`COMFY_QUANT_CODECS`], the
/// rows every checkpoint-importing engine registers.
///
/// Declared once here rather than twice in the backends. A *codec* is a backend-portable
/// declaration of a stored format — the same eight rows mean the same eight things on Metal and on
/// CUDA — so two hand-maintained copies of the list could drift into a checkpoint one backend
/// plans and the other refuses as `UnsupportedFormat`, with nothing comparing them (mlx-gen builds
/// only on macOS, candle-gen's quantized legs only under `cuda`, so no single compilation sees both
/// lists). What stays per-backend is `CODEC_IMPLEMENTATION_IDS` — the claim "I have a decode arm
/// for this row" — which each engine's catalog conformance test pins against this table.
pub const BASELINE_CHECKPOINT_CODECS: &[CheckpointCodecRegistration] = &[
    DENSE_BF16_CODEC,
    DENSE_F16_CODEC,
    DENSE_F32_CODEC,
    FP8_E4M3_SCALAR_CODEC,
    FP8_E5M2_SCALAR_CODEC,
    MXFP8_CODEC,
    INT8_PER_ROW_CODEC,
    NVFP4_CODEC,
];

/// The **GGUF container** codec (epic 20398, sc-20649): one ggml-quantized tensor read out of a
/// `.gguf` file and held **quantized-resident**, dequantized per matmul rather than at load.
///
/// # Why this row is not a safetensors format
///
/// Every other row claims a [`StoredTensorFormat`] built from a safetensors dtype plus an optional
/// `.comfy_quant` descriptor. GGUF is a different *container*: its tensors carry a ggml block type
/// (`Q4_K`, `Q6_K`, `F16`, …), not a safetensors dtype, and a block quant has no integral
/// bytes-per-element. So this row claims [`WeightEncoding::GgufContainer`], which
/// [`WeightEncoding::from_dtype`] can never produce — a safetensors header cannot reach this codec,
/// and this codec never shadows one of the safetensors rows.
///
/// # Residency
///
/// The engine that implements this row keeps the ggml blocks resident (the whole point of a GGUF
/// tier: a Q4_K DiT that fits where a bf16 one does not) and reports **measured** container bytes;
/// [`resident_encoding`](CheckpointCodecRegistration::resident_encoding) is the bf16 the dense
/// fallback would leave, i.e. what the per-matmul dequant produces.
pub const GGUF_CONTAINER_CODEC: CheckpointCodecRegistration = CheckpointCodecRegistration {
    codec_id: "gguf-container-v1",
    stored: &[StoredTensorFormat::undescribed(
        WeightEncoding::GgufContainer,
    )],
    resident_encoding: WeightEncoding::DenseBf16,
};

/// Validated, immutable set of codecs keyed by stored format. Built by
/// [`CheckpointCodecRegistry::new`] (and by `ProviderRegistryBuilder::build` from the rows a catalog
/// registers); duplicate ids or two codecs claiming one stored format fail closed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CheckpointCodecRegistry {
    pub(super) rows: Vec<CheckpointCodecRegistration>,
    pub(super) by_format: BTreeMap<StoredTensorFormat, usize>,
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

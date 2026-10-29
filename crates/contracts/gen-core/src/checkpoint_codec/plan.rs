//! Logical tensor/weight plans and their error types.

use super::*;

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
    /// The adapter-declared transform this logical weight applies to its physical source (sc-21547),
    /// or `None` for the ordinary one-to-one rename.
    ///
    /// When it is `Some`, several plan entries share one `physical_key`. Two consequences the
    /// readers depend on: the file's physical-key surface is the **deduplicated** set of
    /// [`LogicalWeightPlan::all_physical_keys`], and [`Self::source_bytes`] is carried by exactly
    /// one output per physical tensor.
    pub transform: Option<LogicalTensorTransform>,
}

impl LogicalTensorPlan {
    /// The logical shape the codec decodes **before** this entry's transform: the physical tensor's
    /// own geometry. Equal to [`Self::shape`] for an untransformed entry.
    ///
    /// Every backend decode arm that indexes the logical shape positionally must read it from here,
    /// not from `shape` — `shape` is the post-slice shape the model sees, and handing it to a
    /// whole-tensor dequantizer would decode the wrong geometry.
    pub fn source_shape(&self) -> &[usize] {
        match &self.transform {
            Some(transform) => &transform.source_shape,
            None => &self.shape,
        }
    }
    /// `Some(stored_shape)` when this entry is a **block-padded** codec row (MXFP8 or NVFP4) whose
    /// [`Self::shape`] is the padded *stored* grid only because the adapter's
    /// [`LogicalKeyMapping::logical_shape`] declared nothing — i.e. the plan does not know the
    /// layer's true geometry and merely assumed the padding away.
    ///
    /// # Why this is a read-time refusal and not a plan-time one
    ///
    /// MXFP8 storage is 32-padded on both axes and NVFP4 storage 16-padded, and neither file format
    /// records the true `[out_features, in_features]`. With no declaration the compiler can only
    /// carry the stored grid forward — so a `[3072, 60]` layer stored as `[3072, 64]` plans, prices
    /// and decodes as `[3072, 64]`, and four columns of **padding become four columns of weights**
    /// in the tensor handed to the model. That is silent corruption, not a shape error: the values
    /// are real bytes and every downstream shape check that trusts the plan agrees with it.
    ///
    /// Planning is still allowed, because a plan is also a *pricing* artifact: the padded grid is a
    /// conservative over-estimate of resident bytes, and the memory-strategy paths that compile a
    /// plan for admission never materialize a tensor. It is **materialization** that must refuse,
    /// which is what [`Self::undeclared_padded_storage_refusal`] says.
    ///
    /// Dense, scalar-fp8 and int8 rows are never padded (their stored grid *is* the logical one), so
    /// they answer `None` regardless of what the adapter declares.
    pub fn undeclared_padded_storage(&self) -> Option<[usize; 2]> {
        match &self.codec {
            TensorCodecSpec::Mxfp8 {
                stored_shape,
                logical_shape_declared,
                ..
            }
            | TensorCodecSpec::Nvfp4 {
                stored_shape,
                logical_shape_declared,
                ..
            } if !*logical_shape_declared => Some(*stored_shape),
            _ => None,
        }
    }

    /// The one refusal message both engines emit for [`Self::undeclared_padded_storage`], so the
    /// two backends refuse the same checkpoint with the same diagnosis rather than diverging.
    pub fn undeclared_padded_storage_refusal(&self) -> Option<String> {
        let stored = self.undeclared_padded_storage()?;
        Some(format!(
            "codec {}: tensor {:?} stores a block-padded grid {stored:?}, but the plan's mapping \
             declares no logical shape for {:?}; the plan can only assume the padding away, which \
             would materialize the pad rows/columns as real weights. Declare the architecture's \
             logical shape on the adapter's LogicalKeyMapping (or plan this file with a mapping \
             that does) — refusing rather than decoding a padded grid as the layer",
            self.codec_id, self.physical_key, self.logical_key
        ))
    }

    /// The one refusal message both engines emit when a **matrix codec** row (`mxfp8-v1`,
    /// `nvfp4-v1`, `int8-per-row-v1`) carries a logical [`Self::shape`] that is not rank 2.
    ///
    /// Those three codecs decode an `[out, in]` grid: every backend arm indexes the shape
    /// positionally and hands the pair to this crate's `[rows, cols]` reference decoders
    /// ([`crate::decode_mxfp8`], [`crate::decode_nvfp4`], [`crate::decode_int8_per_row`]).
    ///
    /// **Both ranks are checked.** The arms index [`Self::source_shape`] — the *pre-transform*
    /// geometry the codec decodes (sc-21547) — while [`Self::shape`] is the post-transform shape the
    /// model sees and the one downstream shape checks trust. A plan carrying a rank-2 `shape` over a
    /// rank-&lt;2 `transform.source_shape` would pass a `shape`-only guard and then panic out of
    /// bounds on `source_shape[1]`, so this refuses whenever *either* is not rank 2.
    ///
    /// [`compile_logical_weight_plan_with_metadata`] already refuses any descriptor-bearing layer
    /// whose stored header is not rank 2, and the MXFP8/NVFP4 geometry validators refuse a declared
    /// logical shape that is not rank 2 either — so a plan that came from the compiler cannot reach
    /// those arms at another rank. This is the **local** restatement of that contract at the sites
    /// that depend on it: each engine's logical-weight read is a public entry taking a
    /// caller-supplied [`LogicalWeightPlan`] whose fields are public, so without this guard a plan
    /// that did not come from the compiler panics on an out-of-bounds index instead of refusing by
    /// name. It lives here, beside [`Self::undeclared_padded_storage_refusal`], so both engines
    /// refuse the same plan with the same diagnosis rather than keeping two copies that can drift
    /// (nothing compiles both backends at once: mlx-gen is macOS-only and candle-gen's quantized
    /// legs are `cuda`-gated).
    ///
    /// Dense and scalar-fp8 rows are rank-agnostic — they carry the logical shape through whole
    /// (ComfyUI casts biases and modulation vectors to fp8 too) — so they answer `None`: imposing a
    /// floor on them would refuse real checkpoints.
    pub fn matrix_rank_refusal(&self) -> Option<String> {
        const MATRIX_RANK: usize = 2;
        match &self.codec {
            TensorCodecSpec::Mxfp8 { .. }
            | TensorCodecSpec::Nvfp4 { .. }
            | TensorCodecSpec::Int8PerRow { .. } => {}
            TensorCodecSpec::Dense | TensorCodecSpec::ScalarFp8 { .. } => return None,
        }
        // The decode arms index `source_shape()`; `shape` is what the model is handed. Refuse if
        // either is not an [out, in] matrix — see the doc above.
        let offending = if self.source_shape().len() != MATRIX_RANK {
            self.source_shape()
        } else if self.shape.len() != MATRIX_RANK {
            self.shape.as_slice()
        } else {
            return None;
        };
        Some(format!(
            "codec {}: tensor {:?} decodes as an [out, in] matrix — expected rank {MATRIX_RANK}, \
             observed rank {} (planned logical shape {offending:?}); this plan did not come from \
             the checkpoint plan compiler, which enforces rank {MATRIX_RANK} for this codec",
            self.codec_id,
            self.physical_key,
            offending.len(),
        ))
    }
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
    ///
    /// **Not a set.** A tensor the adapter split with a [`LogicalTensorTransform`] contributes one
    /// entry per logical output, all naming the same physical key, so a caller that wants the
    /// file's tensor *surface* must deduplicate (see [`Self::all_physical_keys`]).
    pub fn physical_keys(&self) -> impl Iterator<Item = &str> {
        self.tensors
            .iter()
            .map(|tensor| tensor.physical_key.as_str())
    }

    /// Physical keys of everything the plan accounts for: weights and companions.
    ///
    /// May repeat a weight's key once per transformed logical output — deduplicate before
    /// comparing it against a file's tensor set.
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
    ///
    /// # Why this refuses a GGUF-backed plan (sc-20651)
    ///
    /// The whole point of a synthesized header is that `dtype × shape` **is** the tensor's byte
    /// count — that is the invariant every consumer of these headers reads them under.
    /// `mlx_gen::asset_facts::projected_tensor_headers_bytes` re-prices from `shape` for both the
    /// dense-width and the `GroupQuantized` projection, and only falls back to `data_bytes` for a
    /// tensor it cannot shape.
    ///
    /// A [`WeightEncoding::GgufContainer`] entry breaks that invariant: it is ggml block-quantized,
    /// so it has no integral per-element width ([`WeightEncoding::element_bytes`] reports `0`) and
    /// its `to_dtype` is the opaque `U8` byte view, while `shape` stays the *logical element* grid.
    /// A `[256, 256]` Q4_K weight would present as `U8 [256, 256]` — 65 536 bytes — against a real
    /// container size of 36 864. Rather than emit that header and rely on every present and future
    /// consumer preferring `data_bytes`, the whole view refuses and names the tensor. A GGUF plan's
    /// byte accounting is already available, correct, from [`Self::resident_bytes`] and each
    /// tensor's `residency.resident_bytes`, both measured from ggml's own block/type sizes.
    pub fn resident_tensor_headers(
        &self,
    ) -> Result<Vec<SafetensorsTensorHeader>, ResidentTensorHeadersError> {
        self.tensors
            .iter()
            .map(|tensor| {
                // The encoding whose `to_dtype` this header would present. A zero per-element width
                // means the `dtype × shape` re-pricing the header promises cannot be honoured.
                let presented = match tensor.residency.mode {
                    ResidencyMode::Dense => tensor.resident_encoding,
                    ResidencyMode::Packed => tensor.encoding,
                };
                if presented.element_bytes() == 0 {
                    return Err(ResidentTensorHeadersError::NoPerElementWidth {
                        logical_key: tensor.logical_key.clone(),
                        codec_id: tensor.codec_id,
                        encoding: presented,
                        resident_bytes: tensor.residency.resident_bytes,
                    });
                }
                let (dtype, shape) = match tensor.residency.mode {
                    ResidencyMode::Dense => {
                        (tensor.resident_encoding.to_dtype(), tensor.shape.clone())
                    }
                    ResidencyMode::Packed => (
                        // The stored encoding, exhaustively — NVFP4's nibbles already classify as
                        // `UInt8`, so no wildcard is needed to reach `U8`.
                        tensor.encoding.to_dtype(),
                        {
                            let mut stored = match &tensor.codec {
                                TensorCodecSpec::Mxfp8 { stored_shape, .. } => {
                                    stored_shape.to_vec()
                                }
                                // NVFP4's resident packed form is the on-disk `U8` byte matrix
                                // `[rows, cols / 2]`, not the logical element grid.
                                TensorCodecSpec::Nvfp4 { stored_shape, .. } => {
                                    vec![stored_shape[0], stored_shape[1] / 2]
                                }
                                _ => tensor.shape.clone(),
                            };
                            // A transformed entry is resident as its *slice* of the stored grid, so
                            // the header's `dtype x shape` keeps matching its `data_bytes`
                            // (sc-21547). A packed slice never changes the trailing axes.
                            if let (Some(transform), Some(rows)) =
                                (tensor.transform.as_ref(), stored.first_mut())
                            {
                                *rows = transform.rows.len;
                            }
                            stored
                        },
                    ),
                };
                Ok(SafetensorsTensorHeader {
                    name: tensor.logical_key.clone(),
                    dtype,
                    shape,
                    data_bytes: tensor.residency.resident_bytes,
                })
            })
            .collect()
    }
}

/// Why a plan cannot present a synthesized resident tensor-header view.
///
/// See [`LogicalWeightPlan::resident_tensor_headers`]; the one case today is the GGUF container
/// row, whose ggml blocks have no per-element width to re-price from.
///
/// # Deliberately NOT `#[non_exhaustive]` (sc-20651 feature-end review, minor 13)
///
/// Marking it `#[non_exhaustive]` would let a future variant be added without touching any
/// consumer. That is the wrong trade here, twice over:
///
/// * gen-core is a **workspace-internal contract crate** with no downstream users outside this
///   repo, so there is no compatibility to buy — the only thing `#[non_exhaustive]` would purchase
///   is that a new refusal reason slips past the code that has to react to it.
/// * Every other error enum on this seam ([`LogicalWeightPlanError`] here,
///   `candle_gen_wan::gguf::GgufPlanError`) is exhaustive for the same reason, and
///   `mlx-gen-minimax-h3`'s residency test states the rule outright: list every variant so a new
///   one **reds the code that must handle it**.
///
/// The consequence is intended and is a *compile* error, never a silent one: a second variant here
/// stops `candle_gen_wan::gguf`'s `resident_tensor_headers_refuse_a_gguf_backed_plan` from
/// compiling (it destructures `NoPerElementWidth` irrefutably), which is exactly the prompt to
/// decide what that test should now assert. Whoever adds the variant owns that edit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentTensorHeadersError {
    /// The tensor's resident encoding has no integral bytes-per-element, so a synthesized
    /// `dtype × shape` header would contradict its own byte count.
    NoPerElementWidth {
        logical_key: String,
        codec_id: &'static str,
        encoding: WeightEncoding,
        /// The correct resident size, so the diagnostic carries the number the caller wanted.
        resident_bytes: u64,
    },
}

impl fmt::Display for ResidentTensorHeadersError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoPerElementWidth {
                logical_key,
                codec_id,
                encoding,
                resident_bytes,
            } => write!(
                f,
                "codec {codec_id}: tensor {logical_key:?} is resident as `{}`, which has no \
                 bytes-per-element, so a synthesized `dtype x shape` tensor header would misstate \
                 its size; this plan cannot be priced through tensor headers — read its measured \
                 residency ({resident_bytes} bytes for this tensor) instead",
                encoding.label()
            ),
        }
    }
}

impl std::error::Error for ResidentTensorHeadersError {}

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
    /// One layer is declared twice — by a `.comfy_quant` tensor and by the file-level metadata —
    /// and the two declarations **contradict** each other. Every modelled field is compared, not
    /// just `format`: `full_precision_matrix_mult` decides packed-vs-dense residency, so a silent
    /// resolution in either declaration's favour would change how the layer is priced and run.
    ///
    /// # Field-presence semantics (sc-20651)
    ///
    /// The comparison is presence-aware, because the two routes do not carry the same objects on
    /// real ComfyUI output. `kreamania_variant1.safetensors` declares each int8 projection as
    /// `{"format": "int8_tensorwise", "per_row": true}` in its per-tensor blob and as a bare
    /// `{"format": "int8_tensorwise"}` in the file-level table. The blob is authoritative (it sits
    /// with the weight); the table entry is corroborating. A **strict subset** — keys the entry does
    /// not declare — is not a conflict. A declared value that differs still is, so the sc-20641
    /// conflict rule stands wherever both routes actually say something.
    DescriptorConflict {
        physical_key: String,
        /// The authoritative per-tensor `.comfy_quant` declaration.
        tensor: ComfyQuantDescriptor,
        /// The corroborating file-level entry, as declared (absent fields stay absent).
        metadata: PartialComfyQuantDescriptor,
        /// The descriptor field names that disagree, in declaration order.
        disagreement: Vec<&'static str>,
    },
    /// A block-scaled descriptor declared an `orig_shape` (the Kitchen exporter's pre-quantization
    /// geometry) that is not the logical shape this plan derived from the stored grid plus the
    /// adapter's declaration (sc-21485 review).
    ///
    /// The declaration is never an input — geometry authority stays with the stored shapes — so the
    /// only two possibilities are "it agrees, and corroborates the packed geometry" and "it
    /// disagrees, and one of the two is wrong about the layer". The second refuses here, by name,
    /// rather than being type-checked and dropped.
    DescriptorOrigShape {
        physical_key: String,
        /// The descriptor's declared pre-quantization shape.
        declared: Vec<usize>,
        /// The logical shape the plan derived independently.
        logical: Vec<usize>,
    },
    /// The adapter's declarative logical transform for this tensor is not compilable (sc-21547).
    Transform {
        physical_key: String,
        error: LogicalTransformError,
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
                 authoritative `.comfy_quant` tensor and {} by \
                 `__metadata__._quantization_metadata`; the two declarations disagree on {}",
                tensor.format,
                tensor.full_precision_matrix_mult,
                metadata,
                disagreement.join(", ")
            ),
            Self::DescriptorOrigShape {
                physical_key,
                declared,
                logical,
            } => write!(
                f,
                "weight {physical_key:?} declares `orig_shape` {declared:?} but its stored geometry \
                 and the adapter's declaration give the logical shape {logical:?}; the descriptor \
                 and the checkpoint disagree about this layer"
            ),
            Self::Transform {
                physical_key,
                error,
            } => write!(
                f,
                "on-disk tensor {physical_key:?}: the family adapter's declared logical transform \
                 is invalid: {error}"
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

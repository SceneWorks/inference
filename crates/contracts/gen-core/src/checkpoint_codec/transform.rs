//! Logical key mappings and the logical tensor transforms they declare.

use super::*;

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

    /// The adapter's **declarative logical transform** for one on-disk tensor (sc-21547), when the
    /// checkpoint's physical layout is not one-to-one with the architecture's logical weights.
    ///
    /// Fused checkpoint layouts are the reason this exists: a fused-QKV projection stores one
    /// `[3·d, d]` matrix where the model wants three `[d, d]` weights, and a diffusers-vs-ComfyUI
    /// AdaLN modulation stores `shift` and `scale` in the opposite order from the one the block
    /// reads. Both are pure, deterministic **re-labellings of rows** — no arithmetic, no codec
    /// knowledge — so the adapter declares them and the shared plan compiler consumes them (epic
    /// requirement E1: no codec knowledge moves into a provider).
    ///
    /// `None` (the default) means the ordinary one-to-one route: [`Self::logical_key`] renames the
    /// tensor and [`Self::logical_shape`] declares its geometry. A `Some` declaration **replaces**
    /// both for that physical key — the compiler does not consult `logical_key` or `logical_shape`
    /// for a transformed tensor, because a one-to-many tensor has no single logical key to key them
    /// on; the declaration carries its own
    /// [`LogicalTransformDeclaration::source_logical_shape`] instead.
    ///
    /// Every declaration is validated at plan time against the tensor's real geometry, its codec and
    /// its planned residency ([`LogicalTransformError`]) — before any tensor payload is read.
    fn logical_transform(&self, _physical_key: &str) -> Option<LogicalTransformDeclaration> {
        None
    }
}

/// A half-open row range `[start, start + len)` of a tensor's leading (output/row) axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowRange {
    pub start: usize,
    pub len: usize,
}

impl RowRange {
    pub const fn new(start: usize, len: usize) -> Self {
        Self { start, len }
    }

    /// One past the last row, saturating (the compiler refuses an out-of-bounds range by name, so
    /// saturation here only keeps the *diagnostic* arithmetic total).
    pub const fn end(&self) -> usize {
        self.start.saturating_add(self.len)
    }
}

impl fmt::Display for RowRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rows {}..{}", self.start, self.end())
    }
}

/// One logical weight an adapter derives from a physical tensor.
///
/// The two primitives compose, and every combination the story needs is a point in this space:
/// a plain **rename** is `rows: None, half_swap: false`; a fused-QKV **row slice** is
/// `rows: Some(..)`; an AdaLN **half swap** is `half_swap: true`; and a fused modulation that needs
/// both is both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTransformOutput {
    /// The canonical logical key this output is published under.
    pub logical_key: String,
    /// The contiguous slice of the source's leading axis this output takes. `None` is the whole
    /// axis — the rename case, and the only shape a rank-0 source could accept.
    pub rows: Option<RowRange>,
    /// Exchange the two halves of *this output's* leading axis (after slicing).
    ///
    /// Supported for outputs whose tensor plans a **dense** residency only; a packed-native row
    /// permutation would have to re-derive the codec's scale surface, which is exactly the codec
    /// knowledge this seam keeps out of providers, so a half-swap on a packed row refuses at plan
    /// time ([`LogicalTransformError::HalfSwapOnPackedResidency`]).
    pub half_swap: bool,
}

impl LogicalTransformOutput {
    /// A whole-tensor rename.
    pub fn rename(logical_key: impl Into<String>) -> Self {
        Self {
            logical_key: logical_key.into(),
            rows: None,
            half_swap: false,
        }
    }

    /// A contiguous row slice.
    pub fn row_slice(logical_key: impl Into<String>, start: usize, len: usize) -> Self {
        Self {
            logical_key: logical_key.into(),
            rows: Some(RowRange::new(start, len)),
            half_swap: false,
        }
    }

    /// A whole-tensor leading-axis half swap.
    pub fn half_swap(logical_key: impl Into<String>) -> Self {
        Self {
            logical_key: logical_key.into(),
            rows: None,
            half_swap: true,
        }
    }

    /// This output with its half-swap flag set (composes with [`Self::row_slice`]).
    pub fn with_half_swap(mut self) -> Self {
        self.half_swap = true;
        self
    }
}

/// The adapter's complete declaration for one physical tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTransformDeclaration {
    /// The logical weights this tensor produces, in declaration order. Their row ranges must
    /// **exactly partition** the source's leading axis — see
    /// [`LogicalTransformError::SliceOverlap`] and [`LogicalTransformError::SliceGap`].
    pub outputs: Vec<LogicalTransformOutput>,
    /// The **physical** tensor's true (unpadded) logical shape, when the adapter knows it. This is
    /// the transform's *input* geometry and plays exactly the role
    /// [`LogicalKeyMapping::logical_shape`] plays for an untransformed key: without it a
    /// block-padded codec row (MXFP8/NVFP4) cannot be materialized at all
    /// ([`LogicalTensorPlan::undeclared_padded_storage`]).
    pub source_logical_shape: Option<Vec<usize>>,
}

impl LogicalTransformDeclaration {
    /// A declaration with no source-shape statement.
    pub fn new(outputs: Vec<LogicalTransformOutput>) -> Self {
        Self {
            outputs,
            source_logical_shape: None,
        }
    }

    /// This declaration with the physical tensor's true logical shape attached.
    pub fn with_source_logical_shape(mut self, shape: Vec<usize>) -> Self {
        self.source_logical_shape = Some(shape);
        self
    }
}

/// The **resolved** transform one planned logical tensor applies to its physical source.
///
/// Present on a [`LogicalTensorPlan`] only when the transform actually does something: an adapter
/// declaration that resolves to "the whole tensor, unswapped" is an ordinary rename and plans as
/// `None`, so nothing downstream has to special-case the identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogicalTensorTransform {
    /// The physical tensor's **pre-transform** logical shape — what the codec decodes to before the
    /// slice. Every decode arm reads its geometry from here (via
    /// [`LogicalTensorPlan::source_shape`]); [`LogicalTensorPlan::shape`] is the post-transform
    /// shape the model sees.
    pub source_shape: Vec<usize>,
    /// The rows of `source_shape[0]` this output takes.
    pub rows: RowRange,
    /// Exchange the two halves of the sliced output's leading axis.
    pub half_swap: bool,
}

/// Why an adapter's [`LogicalTransformDeclaration`] cannot be compiled. Every variant names the
/// logical key at fault, and every one is raised **before any tensor payload is materialized**.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogicalTransformError {
    /// A declaration that produces no logical weights would silently drop the tensor.
    NoOutputs,
    /// Two outputs of the same declaration claim one logical key.
    DuplicateLogicalKey { logical_key: String },
    /// A zero-length slice produces an empty tensor.
    EmptySlice { logical_key: String },
    /// The source has no leading axis to slice (a rank-0 tensor).
    SourceNotSliceable {
        logical_key: String,
        source_shape: Vec<usize>,
    },
    /// The slice runs past the end of the source's leading axis.
    SliceOutOfBounds {
        logical_key: String,
        rows: RowRange,
        source_rows: usize,
    },
    /// Two outputs claim overlapping rows.
    SliceOverlap {
        first_logical_key: String,
        first: RowRange,
        second_logical_key: String,
        second: RowRange,
    },
    /// The outputs leave rows of the source unclaimed. Uncovered rows are a **silent drop** — the
    /// most likely shape of an off-by-one in a fused-layout declaration — so the partition must be
    /// exact rather than merely non-overlapping.
    SliceGap {
        /// The first uncovered row.
        first_uncovered_row: usize,
        /// The output that claims rows immediately after the gap, if any.
        next_logical_key: Option<String>,
        source_rows: usize,
    },
    /// A half swap needs an even number of rows to have two halves.
    HalfSwapOddRows { logical_key: String, rows: usize },
    /// A half swap was declared on a tensor this backend plans as packed-native. See
    /// [`LogicalTransformOutput::half_swap`].
    HalfSwapOnPackedResidency {
        logical_key: String,
        codec_id: &'static str,
    },
    /// A packed-native block-scaled row (NVFP4/MXFP8) was sliced at a boundary that does not fall on
    /// the cuBLAS block-scale row tile ([`crate::comfy_quant::MXFP8_SCALE_ROW_TILE`]).
    ///
    /// The packed container keeps its block scales in the 128×4 swizzle, so a slice that cuts an
    /// atom in half would have to *re-derive* the scale surface, and its padded scale tensor would
    /// no longer sum to the source's — the plan's retained-companion pricing and the reader's
    /// measurement would disagree by the padding. A tile-aligned slice takes whole atoms, which is
    /// both sound and exactly priced; anything else refuses.
    PackedSliceAlignment {
        logical_key: String,
        codec_id: &'static str,
        rows: RowRange,
        align: usize,
    },
    /// A packed-native slice of a tensor whose stored grid is **wider than the layer** (ComfyUI
    /// block padding). The slice's row indices are the layer's; the stored payload's are the padded
    /// grid's, and nothing in the packed container records the unpad, so the two cannot be
    /// reconciled. Plan such a file with a dense residency policy.
    PackedSliceOnPaddedGrid {
        logical_key: String,
        codec_id: &'static str,
        source_rows: usize,
        stored_rows: usize,
    },
    /// A packed-native slice whose source bytes do not divide evenly by its rows, so the output's
    /// share of the stored payload is not an integral byte count. (Unreachable for the row-major
    /// codecs in this table; refused rather than rounded.)
    PackedSliceNotByteAligned {
        logical_key: String,
        codec_id: &'static str,
        source_bytes: u64,
        source_rows: usize,
    },
}

impl fmt::Display for LogicalTransformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoOutputs => write!(
                f,
                "the adapter declares a logical transform with no outputs, which would drop the \
                 tensor silently"
            ),
            Self::DuplicateLogicalKey { logical_key } => write!(
                f,
                "two outputs of this transform both publish logical key {logical_key:?}"
            ),
            Self::EmptySlice { logical_key } => write!(
                f,
                "output {logical_key:?} claims zero rows; an empty logical weight is never intended"
            ),
            Self::SourceNotSliceable {
                logical_key,
                source_shape,
            } => write!(
                f,
                "output {logical_key:?} slices the leading axis of a tensor whose logical shape is \
                 {source_shape:?}, which has no leading axis"
            ),
            Self::SliceOutOfBounds {
                logical_key,
                rows,
                source_rows,
            } => write!(
                f,
                "output {logical_key:?} claims {rows} but the tensor has only {source_rows} rows"
            ),
            Self::SliceOverlap {
                first_logical_key,
                first,
                second_logical_key,
                second,
            } => write!(
                f,
                "outputs {first_logical_key:?} ({first}) and {second_logical_key:?} ({second}) \
                 claim overlapping rows"
            ),
            Self::SliceGap {
                first_uncovered_row,
                next_logical_key,
                source_rows,
            } => match next_logical_key {
                Some(next) => write!(
                    f,
                    "the transform leaves row {first_uncovered_row} of {source_rows} unclaimed \
                     (the next output is {next:?}); the outputs must partition the tensor exactly, \
                     or those rows are dropped silently"
                ),
                None => write!(
                    f,
                    "the transform claims only rows 0..{first_uncovered_row} of {source_rows}; the \
                     outputs must partition the tensor exactly, or the remainder is dropped silently"
                ),
            },
            Self::HalfSwapOddRows { logical_key, rows } => write!(
                f,
                "output {logical_key:?} declares a half swap over an odd row count ({rows}); a \
                 half swap needs two equal halves"
            ),
            Self::HalfSwapOnPackedResidency {
                logical_key,
                codec_id,
            } => write!(
                f,
                "output {logical_key:?} declares a half swap but codec {codec_id} plans a \
                 packed-native residency for this tensor; permuting packed rows would require \
                 re-deriving the codec's scale surface. Plan this file with a dense residency \
                 policy, or declare the swap on a dense layer"
            ),
            Self::PackedSliceAlignment {
                logical_key,
                codec_id,
                rows,
                align,
            } => write!(
                f,
                "output {logical_key:?} slices a packed-native {codec_id} tensor at {rows}, but a \
                 packed block-scaled slice must start and end on a multiple of {align} (the \
                 cuBLAS block-scale row tile) so it takes whole scale-factor atoms"
            ),
            Self::PackedSliceOnPaddedGrid {
                logical_key,
                codec_id,
                source_rows,
                stored_rows,
            } => write!(
                f,
                "output {logical_key:?} slices a packed-native {codec_id} tensor whose stored grid \
                 has {stored_rows} rows for a {source_rows}-row layer; the packed container cannot \
                 express the unpad, so its rows cannot be sliced by the layer's indices"
            ),
            Self::PackedSliceNotByteAligned {
                logical_key,
                codec_id,
                source_bytes,
                source_rows,
            } => write!(
                f,
                "output {logical_key:?} slices a packed-native {codec_id} tensor whose \
                 {source_bytes} stored bytes do not divide evenly across its {source_rows} rows"
            ),
        }
    }
}

impl std::error::Error for LogicalTransformError {}

/// One validated output of a [`LogicalTransformDeclaration`], resolved against the tensor's real
/// geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ResolvedTransformOutput {
    pub(super) logical_key: String,
    pub(super) rows: RowRange,
    pub(super) half_swap: bool,
}

impl ResolvedTransformOutput {
    /// Whether this output is the whole tensor, unpermuted — the ordinary rename, which plans with
    /// no [`LogicalTensorTransform`] at all.
    pub(super) fn is_identity(&self, source_rows: usize) -> bool {
        self.rows == RowRange::new(0, source_rows) && !self.half_swap
    }
}

/// Validate one adapter declaration against the tensor it governs and resolve every output's row
/// range — **fail-closed, before any tensor payload is read** (epic requirement E2).
///
/// The declaration is checked against four independent facts, and the order of the checks is the
/// order in which a defect is most usefully named: the declaration's own shape (outputs exist, keys
/// are distinct), each output's slice against the real row count, the whole output set against the
/// axis it must partition, and finally each output against the codec + residency the compiler
/// picked for the tensor.
pub(super) fn resolve_logical_transform(
    outputs: &[LogicalTransformOutput],
    source_shape: &[usize],
    stored_rows: usize,
    codec_id: &'static str,
    codec: &TensorCodecSpec,
    mode: ResidencyMode,
    source_bytes: u64,
) -> Result<Vec<ResolvedTransformOutput>, LogicalTransformError> {
    if outputs.is_empty() {
        return Err(LogicalTransformError::NoOutputs);
    }
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for output in outputs {
        if !seen.insert(output.logical_key.as_str()) {
            return Err(LogicalTransformError::DuplicateLogicalKey {
                logical_key: output.logical_key.clone(),
            });
        }
    }

    // A rank-0 tensor has no leading axis: it can only be renamed, once.
    let Some(source_rows) = source_shape.first().copied() else {
        if let Some(output) = outputs
            .iter()
            .find(|output| output.rows.is_some() || output.half_swap)
            .or_else(|| (outputs.len() > 1).then(|| &outputs[1]))
        {
            return Err(LogicalTransformError::SourceNotSliceable {
                logical_key: output.logical_key.clone(),
                source_shape: source_shape.to_vec(),
            });
        }
        return Ok(vec![ResolvedTransformOutput {
            logical_key: outputs[0].logical_key.clone(),
            rows: RowRange::new(0, 0),
            half_swap: false,
        }]);
    };

    // Per-output geometry.
    let mut resolved = Vec::with_capacity(outputs.len());
    for output in outputs {
        let rows = output.rows.unwrap_or(RowRange::new(0, source_rows));
        if rows.len == 0 {
            return Err(LogicalTransformError::EmptySlice {
                logical_key: output.logical_key.clone(),
            });
        }
        let end = rows.start.checked_add(rows.len).ok_or_else(|| {
            LogicalTransformError::SliceOutOfBounds {
                logical_key: output.logical_key.clone(),
                rows,
                source_rows,
            }
        })?;
        if end > source_rows {
            return Err(LogicalTransformError::SliceOutOfBounds {
                logical_key: output.logical_key.clone(),
                rows,
                source_rows,
            });
        }
        if output.half_swap && !rows.len.is_multiple_of(2) {
            return Err(LogicalTransformError::HalfSwapOddRows {
                logical_key: output.logical_key.clone(),
                rows: rows.len,
            });
        }
        resolved.push(ResolvedTransformOutput {
            logical_key: output.logical_key.clone(),
            rows,
            half_swap: output.half_swap,
        });
    }

    // The output set must partition the leading axis exactly: an overlap double-counts rows, a gap
    // drops them.
    let mut order: Vec<usize> = (0..resolved.len()).collect();
    order.sort_by_key(|index| (resolved[*index].rows.start, resolved[*index].rows.len));
    let mut covered = 0_usize;
    for (position, index) in order.iter().enumerate() {
        let output = &resolved[*index];
        match output.rows.start.cmp(&covered) {
            std::cmp::Ordering::Less => {
                let previous = &resolved[order[position.saturating_sub(1)]];
                return Err(LogicalTransformError::SliceOverlap {
                    first_logical_key: previous.logical_key.clone(),
                    first: previous.rows,
                    second_logical_key: output.logical_key.clone(),
                    second: output.rows,
                });
            }
            std::cmp::Ordering::Greater => {
                return Err(LogicalTransformError::SliceGap {
                    first_uncovered_row: covered,
                    next_logical_key: Some(output.logical_key.clone()),
                    source_rows,
                })
            }
            std::cmp::Ordering::Equal => {}
        }
        covered = output.rows.end();
    }
    if covered != source_rows {
        return Err(LogicalTransformError::SliceGap {
            first_uncovered_row: covered,
            next_logical_key: None,
            source_rows,
        });
    }

    // Codec + residency facts. Nothing below applies to a dense-fallback load: the codec has already
    // produced an ordinary dense tensor by the time the transform runs.
    if mode == ResidencyMode::Packed {
        let block_scaled = matches!(
            codec,
            TensorCodecSpec::Mxfp8 { .. } | TensorCodecSpec::Nvfp4 { .. }
        );
        for output in &resolved {
            if output.half_swap {
                return Err(LogicalTransformError::HalfSwapOnPackedResidency {
                    logical_key: output.logical_key.clone(),
                    codec_id,
                });
            }
            if output.is_identity(source_rows) {
                continue;
            }
            if source_rows != stored_rows {
                return Err(LogicalTransformError::PackedSliceOnPaddedGrid {
                    logical_key: output.logical_key.clone(),
                    codec_id,
                    source_rows,
                    stored_rows,
                });
            }
            if block_scaled
                && !(output
                    .rows
                    .start
                    .is_multiple_of(crate::comfy_quant::MXFP8_SCALE_ROW_TILE)
                    && output
                        .rows
                        .len
                        .is_multiple_of(crate::comfy_quant::MXFP8_SCALE_ROW_TILE))
            {
                return Err(LogicalTransformError::PackedSliceAlignment {
                    logical_key: output.logical_key.clone(),
                    codec_id,
                    rows: output.rows,
                    align: crate::comfy_quant::MXFP8_SCALE_ROW_TILE,
                });
            }
            if stored_rows == 0 || !source_bytes.is_multiple_of(stored_rows as u64) {
                return Err(LogicalTransformError::PackedSliceNotByteAligned {
                    logical_key: output.logical_key.clone(),
                    codec_id,
                    source_bytes,
                    source_rows: stored_rows,
                });
            }
        }
    }

    Ok(resolved)
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

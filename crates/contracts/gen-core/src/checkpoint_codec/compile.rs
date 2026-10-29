//! Compiling a logical weight plan from checkpoint headers and metadata.

use super::*;

/// The companion suffixes of the `.comfy_quant` convention plus the refused legacy names.
pub(super) const DESCRIPTOR_SUFFIX: &str = ".comfy_quant";
pub(super) const WEIGHT_SCALE_SUFFIX: &str = ".weight_scale";
pub(super) const INPUT_SCALE_SUFFIX: &str = ".input_scale";
pub(super) const NVFP4_SCALE_2_SUFFIX: &str = ".weight_scale_2";
pub(super) const LEGACY_SCALE_SUFFIXES: &[&str] = &[".scale_weight", ".scale_input"];
pub(super) const LEGACY_MARKER_KEY: &str = "scaled_fp8";

pub(super) fn scalar_scale_shape_ok(shape: &[usize]) -> bool {
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
///
/// # Precedence when a layer is declared twice (sc-20651)
///
/// The two routes do **not** carry the same per-layer objects on real ComfyUI output, so they are
/// not symmetric. The published KreaMania int8 artifacts write each of their 264 projections as
/// `{"format": "int8_tensorwise", "per_row": true}` in the per-tensor blob and as a bare
/// `{"format": "int8_tensorwise"}` in this table; parsing the file-level entry under the standalone
/// rules refused the entire checkpoint on `Int8NotPerRow`.
///
/// * The per-tensor `.comfy_quant` blob is **authoritative** — it sits with the weight it describes.
///   A file-level entry for the same layer is **corroborating**.
/// * A corroborating entry that is a strict **subset** of the blob's fields does not refuse.
/// * A corroborating entry whose declared **value** contradicts the blob still refuses with
///   [`LogicalWeightPlanError::DescriptorConflict`] — the sc-20641 conflict rule stands.
/// * A layer this table declares **alone** must stand on its own, and refuses exactly as before.
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
    //
    // Parsed but deliberately NOT completed here (sc-20651). These entries are *corroborating*
    // wherever a per-tensor `.comfy_quant` blob describes the same layer, and only a layer this
    // table declares alone has to satisfy the standalone rules. Completing them up front is exactly
    // what refused the real KreaMania int8 artifacts, whose file-level entries are a strict subset
    // of their blobs — see [`PartialComfyQuantDescriptor`].
    let mut metadata_descriptors: BTreeMap<String, (String, PartialComfyQuantDescriptor)> =
        BTreeMap::new(); // physical layer base -> (name as declared in the table, entry)
    if let Some(payload) = quantization_metadata {
        let table = crate::comfy_quant::parse_quantization_metadata(payload)
            .map_err(|error| LogicalWeightPlanError::QuantizationMetadata { error })?;
        let prefix = resolve_metadata_prefix(&by_name, &table)?;
        for (layer, descriptor) in table {
            metadata_descriptors.insert(format!("{prefix}{layer}"), (layer, descriptor));
        }
    }

    // ---- classify companions and validate descriptors, per layer ----------------------------
    let mut descriptors: BTreeMap<String, ComfyQuantDescriptor> = BTreeMap::new(); // by layer base
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
        // The blob is authoritative. A file-level entry for the same layer only has to be
        // CONSISTENT with it: fields the entry omits are not a disagreement (the real-artifact
        // precedent), a value that contradicts the blob still is.
        if let Some((_, metadata)) = metadata_descriptors.remove(base) {
            let disagreement = metadata.disagreement_with(&descriptor);
            if !disagreement.is_empty() {
                return Err(LogicalWeightPlanError::DescriptorConflict {
                    physical_key: weight_key,
                    tensor: descriptor,
                    metadata,
                    disagreement,
                });
            }
        }
        descriptors.insert(base.to_owned(), descriptor);
    }

    // Whatever is left in the table describes a layer with no `.comfy_quant` blob, so it is the only
    // declaration that layer has and must stand on its own — the pre-sc-20651 behaviour, refusals
    // included (the metadata-only NVFP4 form, sc-20641).
    for (base, (declared_layer, metadata)) in metadata_descriptors {
        let descriptor = metadata.into_complete().map_err(|defect| {
            LogicalWeightPlanError::QuantizationMetadata {
                error: QuantizationMetadataError::Layer {
                    layer: declared_layer,
                    defect,
                },
            }
        })?;
        descriptors.insert(base, descriptor);
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

        // A weight (or plain dense) tensor. The adapter either renames it one-to-one
        // (`logical_key`) or declares a one-to-many transform (`logical_transform`, sc-21547); the
        // two are alternatives, and a declaration carries its own source-shape statement.
        let declaration = mapping.logical_transform(&header.name);
        let declared_outputs: Vec<LogicalTransformOutput> = match &declaration {
            Some(declaration) => declaration.outputs.clone(),
            None => vec![LogicalTransformOutput::rename(
                mapping.logical_key(&header.name).ok_or_else(|| {
                    LogicalWeightPlanError::UnmappedKey {
                        physical_key: header.name.clone(),
                    }
                })?,
            )],
        };
        // Declaration-internal duplicates are named as such before the cross-tensor collision map
        // sees them (where they would read as a tensor colliding with itself).
        {
            let mut seen: BTreeSet<&str> = BTreeSet::new();
            for output in &declared_outputs {
                if !seen.insert(output.logical_key.as_str()) {
                    return Err(LogicalWeightPlanError::Transform {
                        physical_key: header.name.clone(),
                        error: LogicalTransformError::DuplicateLogicalKey {
                            logical_key: output.logical_key.clone(),
                        },
                    });
                }
            }
        }
        for output in &declared_outputs {
            if let Some(first) = owners.insert(output.logical_key.clone(), header.name.clone()) {
                return Err(LogicalWeightPlanError::KeyCollision {
                    logical_key: output.logical_key.clone(),
                    first_physical_key: first,
                    second_physical_key: header.name.clone(),
                });
            }
        }
        // The declared geometry of the tensor the codec decodes. For a transformed tensor this is
        // the declaration's own statement — the outputs have no single logical key to key
        // `logical_shape` on.
        let declared_source_shape: Option<Vec<usize>> = match &declaration {
            Some(declaration) => declaration.source_logical_shape.clone(),
            None => mapping.logical_shape(&declared_outputs[0].logical_key),
        };
        let encoding = WeightEncoding::from_dtype(header.dtype).ok_or_else(|| {
            LogicalWeightPlanError::UnclassifiedDtype {
                physical_key: header.name.clone(),
                dtype: format!("{:?}", header.dtype),
            }
        })?;
        let base = header.name.strip_suffix(".weight");
        let descriptor = base.and_then(|base| descriptors.get(base));
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
                        let declared = declared_source_shape.clone();
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
                        let declared = declared_source_shape.clone();
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

        // The block-scaled provenance `orig_shape`, checked rather than merely type-checked
        // (sc-21485 review). `logical_shape` was derived independently — from the stored grid's
        // geometry plus, where the adapter declared one, its logical shape — so an `orig_shape`
        // that agrees is free corroboration of the packed geometry and one that differs means the
        // exporter's declaration and the checkpoint disagree about the layer. Only block-scaled
        // descriptors can carry the key at all (`partial_descriptor_from_json` refuses it elsewhere
        // as an unknown field), so this covers exactly NVFP4 and MXFP8.
        if let Some(declared_orig) =
            descriptor.and_then(|descriptor| descriptor.orig_shape.as_deref())
        {
            if declared_orig != logical_shape.as_slice() {
                return Err(LogicalWeightPlanError::DescriptorOrigShape {
                    physical_key: header.name.clone(),
                    declared: declared_orig.to_vec(),
                    logical: logical_shape.clone(),
                });
            }
        }

        // Residency: the backend's packed-vs-dense decision, priced per tensor. A layer flagged
        // `full_precision_matrix_mult` never runs packed.
        let mode = if codec_spec.full_precision_matrix_mult() {
            ResidencyMode::Dense
        } else {
            residency.residency(codec, &codec_spec, &header.shape)
        };
        // The adapter's declarative transform, resolved against this tensor's real geometry, its
        // codec and the residency just chosen (sc-21547) — the last plan-time gate before the
        // entries are emitted, and still before any tensor payload is read.
        let outputs = resolve_logical_transform(
            &declared_outputs,
            &logical_shape,
            header.shape.first().copied().unwrap_or(0),
            codec.codec_id,
            &codec_spec,
            mode,
            header.data_bytes,
        )
        .map_err(|error| LogicalWeightPlanError::Transform {
            physical_key: header.name.clone(),
            error,
        })?;
        let source_rows = logical_shape.first().copied().unwrap_or(0);
        let stored_rows = header.shape.first().copied().unwrap_or(0);
        // Every packed output owns its own copy of each retained **scalar** scale — those survive
        // the decode as values on the materialized tensor, not as file rows — so a one-to-many
        // transform retains one per output and the plan must price all of them (epic requirement
        // E4's other half: no *under*-counting either). The block-scale surface does not multiply:
        // a tile-aligned packed slice takes whole scale-factor atoms, so the per-output scale
        // tensors partition the stored one exactly.
        let scalar_scale_copies = outputs.len() as u64;

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
                            ResidencyMode::Packed => {
                                scale_header.data_bytes.saturating_mul(scalar_scale_copies)
                            }
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
                            ResidencyMode::Packed => {
                                input_header.data_bytes.saturating_mul(scalar_scale_copies)
                            }
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
                        // The two scalar levels are retained once per logical output; the swizzled
                        // block-scale surface partitions across them instead (sc-21547).
                        let copies = match role {
                            CompanionRole::WeightScale => 1,
                            _ => scalar_scale_copies,
                        };
                        let retained = match mode {
                            ResidencyMode::Packed => {
                                companion_header.data_bytes.saturating_mul(copies)
                            }
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
                            ResidencyMode::Packed => {
                                input_header.data_bytes.saturating_mul(scalar_scale_copies)
                            }
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

        // One plan entry per logical output. The physical tensor's **source bytes are counted
        // once** (epic requirement E4): the first output in declaration order carries them and its
        // siblings carry zero, so a receipt that totals `source_bytes` over materialized tensors
        // still totals the file exactly once. Resident bytes are attributed per output, and --
        // because the outputs partition the leading axis -- they sum to the whole tensor's pricing.
        for (position, output) in outputs.into_iter().enumerate() {
            let mut shape = logical_shape.clone();
            if let Some(rows) = shape.first_mut() {
                *rows = output.rows.len;
            }
            let resident_bytes = match mode {
                // A packed output is resident as its share of the stored payload; the identity
                // takes the stored byte count verbatim rather than re-deriving it.
                ResidencyMode::Packed => {
                    if output.is_identity(source_rows) || stored_rows == 0 {
                        header.data_bytes
                    } else {
                        header.data_bytes / stored_rows as u64 * output.rows.len as u64
                    }
                }
                ResidencyMode::Dense => shape
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
            let transform = (!output.is_identity(source_rows)).then(|| LogicalTensorTransform {
                source_shape: logical_shape.clone(),
                rows: output.rows,
                half_swap: output.half_swap,
            });
            tensors.push(LogicalTensorPlan {
                logical_key: output.logical_key,
                physical_key: header.name.clone(),
                encoding,
                shape,
                source_bytes: if position == 0 { header.data_bytes } else { 0 },
                codec_id: codec.codec_id,
                resident_encoding: codec.resident_encoding,
                codec: codec_spec.clone(),
                residency: PlannedResidency {
                    mode,
                    resident_bytes,
                },
                transform,
            });
        }
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
pub(super) fn resolve_metadata_prefix(
    by_name: &BTreeMap<&str, &SafetensorsTensorHeader>,
    table: &BTreeMap<String, PartialComfyQuantDescriptor>,
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

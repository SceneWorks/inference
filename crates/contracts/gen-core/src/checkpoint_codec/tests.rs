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
    let error =
        compile_meta(&ambiguous, Some(metadata)).expect_err("two candidate prefixes must refuse");
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

/// sc-21485 review. The Kitchen provenance `orig_shape` is **checked** against the logical
/// shape the plan derived from the stored grid, not type-checked and dropped: an agreeing
/// declaration corroborates the packed geometry, a differing one refuses by name.
///
/// Mutation witness: delete the `DescriptorOrigShape` equality check in
/// `compile_logical_weight_plan_with_metadata` and the mismatch arm below goes green — which is
/// exactly the "type-checked then discarded" behaviour this test exists to forbid.
#[test]
fn a_declared_orig_shape_must_equal_the_derived_logical_shape() {
    // Stored U8 [32, 32] ⇒ 64 four-bit codes per row ⇒ logical [32, 64].
    let headers = [
        header("model.q.weight", Dtype::U8, &[32, 32]),
        header("model.q.weight_scale", Dtype::F8_E4M3, &[128, 4]),
        header("model.q.weight_scale_2", Dtype::F32, &[]),
    ];
    let compile_meta = |payload: &str| {
        compile_logical_weight_plan_with_metadata(
            &headers,
            &no_descriptors(),
            Some(payload),
            &StripPrefix,
            &full(),
            &DenseResidencyPolicy,
        )
    };

    // The true pre-quantization shape: accepted, and the plan is unchanged by its presence.
    let corroborated = compile_meta(
            r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4", "group_size": 16, "orig_dtype": "torch.bfloat16", "orig_shape": [32, 64]}}}"#,
        )
        .expect("an orig_shape equal to the derived logical shape corroborates");
    assert_eq!(corroborated.tensors[0].shape, vec![32, 64]);

    // The STORED byte grid is not the logical shape; declaring it refuses by name.
    let error = compile_meta(
            r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4", "orig_shape": [32, 32]}}}"#,
        )
        .expect_err("an orig_shape that is not the logical shape must refuse");
    assert_eq!(
        error,
        LogicalWeightPlanError::DescriptorOrigShape {
            physical_key: "model.q.weight".to_owned(),
            declared: vec![32, 32],
            logical: vec![32, 64],
        },
        "{error}"
    );
    assert!(error.to_string().contains("model.q.weight"), "{error}");

    // A transposed declaration is the same class of disagreement — the check is equality, not
    // an element-count comparison that a transpose would slip through.
    assert!(
        matches!(
            compile_meta(
                r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4", "orig_shape": [64, 32]}}}"#,
            ),
            Err(LogicalWeightPlanError::DescriptorOrigShape { .. })
        ),
        "a transposed orig_shape must refuse"
    );
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
        compile_meta(r#"{"format_version": "1.0", "layers": {"q": {"format": "nvfp4"}}}"#).is_ok()
    );
    // Disagreeing formats refuse, naming the weight, both descriptors and the field.
    let error =
        compile_meta(r#"{"format_version": "1.0", "layers": {"q": {"format": "float8_e4m3fn"}}}"#)
            .expect_err("two declarations that disagree must refuse");
    assert_eq!(
        error,
        LogicalWeightPlanError::DescriptorConflict {
            physical_key: "model.q.weight".to_owned(),
            tensor: ComfyQuantDescriptor {
                format: ComfyQuantFormat::Nvfp4,
                full_precision_matrix_mult: false,
                orig_shape: None,
            },
            metadata: PartialComfyQuantDescriptor {
                format: ComfyQuantFormat::Float8E4M3Fn,
                full_precision_matrix_mult: None,
                per_row: None,
                orig_shape: None,
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
                orig_shape: None,
            },
            metadata: PartialComfyQuantDescriptor {
                format: ComfyQuantFormat::Nvfp4,
                full_precision_matrix_mult: Some(true),
                per_row: None,
                orig_shape: None,
            },
            disagreement: vec!["full_precision_matrix_mult"],
        }
    );
    assert!(
        error.to_string().contains("full_precision_matrix_mult"),
        "{error}"
    );
}

/// sc-20651, the real-artifact precedent. `kreamania_variant1.safetensors` declares each of its
/// 264 int8 projections TWICE and the two declarations are **not** the same object: the
/// authoritative per-tensor blob says `{"format": "int8_tensorwise", "per_row": true}` while the
/// file-level table says a bare `{"format": "int8_tensorwise"}`. Parsing the file-level entry
/// under the standalone rules refused the whole checkpoint on `Int8NotPerRow`.
///
/// The rule this pins, in both directions:
/// * a file-level entry that is a strict SUBSET of the blob (missing keys) is corroborating and
///   must not refuse — in either field, and whichever route omits;
/// * a file-level entry whose declared VALUE contradicts the blob still refuses;
/// * a layer the table declares ALONE is unchanged: it must stand on its own.
///
/// Mutation: restore blind file-level-first parsing (complete the table's entries before the
/// blobs are read) and the first assertion here refuses with `Int8NotPerRow`.
#[test]
fn a_file_level_entry_that_is_a_subset_of_its_tensor_blob_corroborates_it() {
    let int8_headers = |descriptor_len: usize| {
        [
            header("model.q.weight", Dtype::I8, &[4, 8]),
            header("model.q.weight_scale", Dtype::F32, &[4]),
            header("model.q.comfy_quant", Dtype::U8, &[descriptor_len]),
        ]
    };
    let compile = |blob: &[u8], payload: Option<&str>| {
        let headers = int8_headers(blob.len());
        let descriptors: BTreeMap<String, Vec<u8>> =
            [("model.q.comfy_quant".to_owned(), blob.to_vec())].into();
        compile_logical_weight_plan_with_metadata(
            &headers,
            &descriptors,
            payload,
            &StripPrefix,
            &full(),
            &DenseResidencyPolicy,
        )
    };
    const PER_ROW: &[u8] = br#"{"format": "int8_tensorwise", "per_row": true}"#;
    const BARE_TABLE: &str =
        r#"{"format_version": "1.0", "layers": {"q": {"format": "int8_tensorwise"}}}"#;

    // THE REGRESSION: the real artifact's exact pair of declarations must plan.
    let plan = compile(PER_ROW, Some(BARE_TABLE))
        .expect("a bare file-level entry corroborates a per_row blob; it must not refuse");
    assert_eq!(plan.codec_ids(), ["int8-per-row-v1"]);
    assert_eq!(plan.tensors.len(), 1);
    // Same plan as the blob alone: the corroborating entry adds nothing and takes nothing away.
    assert_eq!(plan.tensors, compile(PER_ROW, None).unwrap().tensors);

    // Subset in the other field, and in the other direction: the blob declares
    // `full_precision_matrix_mult`, the table does not. Still corroborating, and the
    // authoritative flag survives.
    const FPMM: &[u8] =
        br#"{"format": "int8_tensorwise", "per_row": true, "full_precision_matrix_mult": true}"#;
    let plan = compile(FPMM, Some(BARE_TABLE)).expect("a subset entry must not refuse");
    assert!(plan.tensors[0].codec.full_precision_matrix_mult());

    // A declared value that CONTRADICTS the blob still refuses — sc-20641's rule stands.
    let error = compile(
            PER_ROW,
            Some(
                r#"{"format_version": "1.0", "layers": {"q": {"format": "int8_tensorwise", "per_row": false}}}"#,
            ),
        )
        .expect_err("a declared `per_row: false` contradicts the authoritative blob");
    assert_eq!(
        error,
        LogicalWeightPlanError::DescriptorConflict {
            physical_key: "model.q.weight".to_owned(),
            tensor: ComfyQuantDescriptor {
                format: ComfyQuantFormat::Int8TensorwisePerRow,
                full_precision_matrix_mult: false,
                orig_shape: None,
            },
            metadata: PartialComfyQuantDescriptor {
                format: ComfyQuantFormat::Int8TensorwisePerRow,
                full_precision_matrix_mult: None,
                per_row: Some(false),
                orig_shape: None,
            },
            disagreement: vec!["per_row"],
        }
    );
    assert!(error.to_string().contains("per_row"), "{error}");
    // The diagnostic renders the file-level entry as DECLARED — `(per_row=false)` alone, never
    // the `full_precision_matrix_mult=false` default the producer never wrote. (The tensor half
    // of the message does carry that field, because the blob's descriptor really is complete.)
    assert!(
        error
            .to_string()
            .contains("`int8_tensorwise` (per_row=false)"),
        "{error}"
    );

    // A layer the table declares ALONE is unchanged: no blob corroborates it, so it must stand
    // on its own and refuses with the same layer-named message as before.
    let headers = [
        header("model.q.weight", Dtype::I8, &[4, 8]),
        header("model.q.weight_scale", Dtype::F32, &[4]),
    ];
    let error = compile_logical_weight_plan_with_metadata(
        &headers,
        &BTreeMap::new(),
        Some(BARE_TABLE),
        &StripPrefix,
        &full(),
        &DenseResidencyPolicy,
    )
    .expect_err("a metadata-only int8 layer has no authority to defer to");
    assert_eq!(
        error,
        LogicalWeightPlanError::QuantizationMetadata {
            error: QuantizationMetadataError::Layer {
                layer: "q".to_owned(),
                defect: ComfyQuantDescriptorError::Int8NotPerRow,
            },
        }
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
    assert_eq!(
        plan.resident_tensor_headers().expect("priceable")[0].shape,
        vec![32, 32]
    );
    assert_eq!(
        dense.resident_tensor_headers().expect("priceable")[0].shape,
        vec![32, 64]
    );
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
        demotions: Vec::new(),
        residency: vec![
            CodecResidencyReport {
                codec_id: "a",
                representation: crate::checkpoint_facts::ExecutionRepresentation::DenseFallback,
                tensor_count: 1,
                source_bytes: 4,
                resident_bytes: 4,
            },
            CodecResidencyReport {
                codec_id: "b",
                representation: crate::checkpoint_facts::ExecutionRepresentation::NativePacked,
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
    let plan = compile_logical_weight_plan(&headers, &descriptors, &StripPrefix, &full(), &PackFp8)
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
    // Scalar fp8 has no block axis, so a declared `group_size` refuses as the layout
    // redefinition it is (sc-21485 refined the refusal from UnknownField).
    let error = make(r#"{"format": "float8_e4m3fn", "group_size": 64}"#).unwrap_err();
    assert!(
        matches!(
            &error,
            LogicalWeightPlanError::Descriptor {
                layer,
                defect: ComfyQuantDescriptorError::GroupSizeMismatch { expected: None, .. },
                ..
            } if layer == "model.q"
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
    let resident = plan.resident_tensor_headers().expect("priceable");
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

    let resident = plan.resident_tensor_headers().expect("priceable");
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

// ---- adapter-declared logical transforms (sc-21547) ---------------------------------------
//
// Fused checkpoint layouts (a fused-QKV projection; a diffusers-vs-ComfyUI AdaLN modulation)
// are one physical tensor and several logical weights. The adapter declares the re-labelling;
// the shared compiler validates it against the tensor's real geometry, codec and residency and
// prices every output. Nothing below reads a byte of tensor payload — that is the point.

/// A mapping that answers with whatever declaration the test installs for a physical key.
struct DeclaredTransforms {
    transforms: BTreeMap<String, LogicalTransformDeclaration>,
}

impl DeclaredTransforms {
    fn new(
        transforms: impl IntoIterator<Item = (&'static str, LogicalTransformDeclaration)>,
    ) -> Self {
        Self {
            transforms: transforms
                .into_iter()
                .map(|(key, declaration)| (key.to_owned(), declaration))
                .collect(),
        }
    }
}

impl LogicalKeyMapping for DeclaredTransforms {
    fn mapping_id(&self) -> &'static str {
        "declared-transforms-test"
    }
    fn logical_key(&self, physical_key: &str) -> Option<String> {
        physical_key.strip_prefix("model.").map(str::to_owned)
    }
    fn logical_transform(&self, physical_key: &str) -> Option<LogicalTransformDeclaration> {
        self.transforms.get(physical_key).cloned()
    }
}

/// A policy that keeps every quantized row packed — the residency half of the transform rules
/// only has teeth against a backend that actually plans `Packed`.
struct AlwaysPacked;

impl CodecResidencyPolicy for AlwaysPacked {
    fn residency(
        &self,
        _codec: &CheckpointCodecRegistration,
        spec: &TensorCodecSpec,
        _stored_shape: &[usize],
    ) -> ResidencyMode {
        if spec.full_precision_matrix_mult() || matches!(spec, TensorCodecSpec::Dense) {
            ResidencyMode::Dense
        } else {
            ResidencyMode::Packed
        }
    }
}

/// The transform error a declaration produces for `model.w`, or a panic naming what it did
/// instead. Every one of these must be raised at *plan* time.
fn transform_defect(
    headers: &[SafetensorsTensorHeader],
    declaration: LogicalTransformDeclaration,
) -> LogicalTransformError {
    let mapping = DeclaredTransforms::new([("model.w", declaration)]);
    match compile(headers, &no_descriptors(), &mapping, &baseline()) {
        Err(LogicalWeightPlanError::Transform {
            physical_key,
            error,
        }) => {
            assert_eq!(physical_key, "model.w");
            error
        }
        other => panic!("expected a transform refusal, got {other:?}"),
    }
}

/// A fused-QKV row split: one `[96, 8]` matrix becomes three `[32, 8]` logical weights whose
/// keys and shapes are exactly the compiled plan's, whose resident pricing sums to the whole
/// tensor's, and whose **source bytes are counted once**.
#[test]
fn a_declared_row_split_publishes_priced_logical_outputs_and_counts_source_bytes_once() {
    let headers = [header("model.w", Dtype::BF16, &[96, 8])];
    let mapping = DeclaredTransforms::new([(
        "model.w",
        LogicalTransformDeclaration::new(vec![
            LogicalTransformOutput::row_slice("attn.q", 0, 32),
            LogicalTransformOutput::row_slice("attn.k", 32, 32),
            LogicalTransformOutput::row_slice("attn.v", 64, 32),
        ]),
    )]);
    let plan = compile(&headers, &no_descriptors(), &mapping, &baseline()).expect("plan");

    assert_eq!(
        plan.logical_keys().collect::<Vec<_>>(),
        ["attn.k", "attn.q", "attn.v"],
        "the plan is sorted by logical key"
    );
    for tensor in &plan.tensors {
        assert_eq!(tensor.physical_key, "model.w");
        assert_eq!(tensor.shape, vec![32, 8]);
        assert_eq!(tensor.source_shape(), [96, 8]);
        assert_eq!(tensor.residency.resident_bytes, 32 * 8 * 2);
    }
    let by_key: BTreeMap<&str, &LogicalTensorPlan> = plan
        .tensors
        .iter()
        .map(|tensor| (tensor.logical_key.as_str(), tensor))
        .collect();
    assert_eq!(
        by_key["attn.k"].transform.as_ref().map(|t| t.rows),
        Some(RowRange::new(32, 32))
    );
    assert!(by_key["attn.v"]
        .transform
        .as_ref()
        .is_some_and(|t| !t.half_swap));

    // E4: the physical tensor's bytes appear exactly once across the three entries, and the
    // plan's own file total is untouched by the fan-out.
    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.source_bytes)
            .sum::<u64>(),
        96 * 8 * 2
    );
    assert_eq!(plan.source_bytes, 96 * 8 * 2);
    assert_eq!(plan.resident_bytes(), 96 * 8 * 2);
    // Three logical weights, one physical tensor.
    assert_eq!(plan.tensor_count(), 3);
    assert_eq!(
        plan.all_physical_keys().collect::<BTreeSet<_>>(),
        BTreeSet::from(["model.w"])
    );
}

/// The AdaLN case: the whole tensor under one key, with its two halves exchanged. The shape is
/// unchanged, so only the transform records that the rows moved.
#[test]
fn a_declared_half_swap_keeps_the_shape_and_records_the_permutation() {
    let headers = [header("model.w", Dtype::BF16, &[8, 4])];
    let mapping = DeclaredTransforms::new([(
        "model.w",
        LogicalTransformDeclaration::new(vec![LogicalTransformOutput::half_swap(
            "norm.modulation",
        )]),
    )]);
    let plan = compile(&headers, &no_descriptors(), &mapping, &baseline()).expect("plan");
    let tensor = &plan.tensors[0];
    assert_eq!(tensor.logical_key, "norm.modulation");
    assert_eq!(tensor.shape, vec![8, 4]);
    let transform = tensor
        .transform
        .as_ref()
        .expect("a swap is not the identity");
    assert!(transform.half_swap);
    assert_eq!(transform.rows, RowRange::new(0, 8));
    assert_eq!(plan.resident_bytes(), 8 * 4 * 2);
}

/// A declaration that resolves to "the whole tensor, unpermuted" is an ordinary rename: it must
/// plan with **no** transform, so every existing consumer of an untransformed plan is untouched.
#[test]
fn a_declared_whole_tensor_rename_plans_as_the_identity() {
    let headers = [header("model.w", Dtype::BF16, &[8, 4])];
    for declaration in [
        LogicalTransformDeclaration::new(vec![LogicalTransformOutput::rename("w")]),
        LogicalTransformDeclaration::new(vec![LogicalTransformOutput::row_slice("w", 0, 8)]),
    ] {
        let mapping = DeclaredTransforms::new([("model.w", declaration)]);
        let plan = compile(&headers, &no_descriptors(), &mapping, &baseline()).expect("plan");
        assert_eq!(plan.tensors[0].logical_key, "w");
        assert_eq!(plan.tensors[0].transform, None);
        assert_eq!(plan.tensors[0].source_shape(), [8, 4]);
    }
}

/// Every malformed declaration refuses, by name, before any tensor payload could be read. These
/// are the mutations of the fused-QKV split above: each one is a plausible off-by-one in an
/// adapter and each one would otherwise corrupt or silently drop weights.
#[test]
fn malformed_transform_declarations_refuse_at_plan_time() {
    let headers = [header("model.w", Dtype::BF16, &[96, 8])];
    let split = |q: (usize, usize), k: (usize, usize), v: (usize, usize)| {
        LogicalTransformDeclaration::new(vec![
            LogicalTransformOutput::row_slice("attn.q", q.0, q.1),
            LogicalTransformOutput::row_slice("attn.k", k.0, k.1),
            LogicalTransformOutput::row_slice("attn.v", v.0, v.1),
        ])
    };

    // Overlap: `k` starts one row early, so row 31 would be published twice.
    assert_eq!(
        transform_defect(&headers, split((0, 32), (31, 33), (64, 32))),
        LogicalTransformError::SliceOverlap {
            first_logical_key: "attn.q".to_owned(),
            first: RowRange::new(0, 32),
            second_logical_key: "attn.k".to_owned(),
            second: RowRange::new(31, 33),
        }
    );
    // Gap: a short `q` leaves row 31 unclaimed — a silent drop, not a shape error.
    assert_eq!(
        transform_defect(&headers, split((0, 31), (32, 32), (64, 32))),
        LogicalTransformError::SliceGap {
            first_uncovered_row: 31,
            next_logical_key: Some("attn.k".to_owned()),
            source_rows: 96,
        }
    );
    // Short by a whole slice at the end.
    assert_eq!(
        transform_defect(
            &headers,
            LogicalTransformDeclaration::new(vec![
                LogicalTransformOutput::row_slice("attn.q", 0, 32),
                LogicalTransformOutput::row_slice("attn.k", 32, 32),
            ])
        ),
        LogicalTransformError::SliceGap {
            first_uncovered_row: 64,
            next_logical_key: None,
            source_rows: 96,
        }
    );
    // Out of bounds.
    assert_eq!(
        transform_defect(&headers, split((0, 32), (32, 32), (64, 40))),
        LogicalTransformError::SliceOutOfBounds {
            logical_key: "attn.v".to_owned(),
            rows: RowRange::new(64, 40),
            source_rows: 96,
        }
    );
    // Zero-length.
    assert_eq!(
        transform_defect(&headers, split((0, 0), (0, 32), (32, 64))),
        LogicalTransformError::EmptySlice {
            logical_key: "attn.q".to_owned()
        }
    );
    // Two outputs of one declaration under one key.
    assert_eq!(
        transform_defect(
            &headers,
            LogicalTransformDeclaration::new(vec![
                LogicalTransformOutput::row_slice("attn.q", 0, 48),
                LogicalTransformOutput::row_slice("attn.q", 48, 48),
            ])
        ),
        LogicalTransformError::DuplicateLogicalKey {
            logical_key: "attn.q".to_owned()
        }
    );
    // A half swap needs two halves.
    assert_eq!(
        transform_defect(
            &headers,
            LogicalTransformDeclaration::new(vec![
                LogicalTransformOutput::row_slice("attn.q", 0, 31).with_half_swap(),
                LogicalTransformOutput::row_slice("attn.k", 31, 65),
            ])
        ),
        LogicalTransformError::HalfSwapOddRows {
            logical_key: "attn.q".to_owned(),
            rows: 31,
        }
    );
    // A declaration that publishes nothing.
    assert_eq!(
        transform_defect(&headers, LogicalTransformDeclaration::new(Vec::new())),
        LogicalTransformError::NoOutputs
    );
    // A rank-0 tensor has no rows to slice.
    assert_eq!(
        transform_defect(
            &[header("model.w", Dtype::BF16, &[])],
            LogicalTransformDeclaration::new(vec![LogicalTransformOutput::row_slice(
                "scalar", 0, 1
            )])
        ),
        LogicalTransformError::SourceNotSliceable {
            logical_key: "scalar".to_owned(),
            source_shape: Vec::new(),
        }
    );
}

/// A transform output that collides with **another** tensor's logical key is the ordinary
/// collision, reported as such (the mapping is not injective over the file).
#[test]
fn a_transform_output_that_collides_with_another_tensor_refuses_as_a_collision() {
    let headers = [
        header("model.w", Dtype::BF16, &[4, 8]),
        header("model.attn.q", Dtype::BF16, &[2, 8]),
    ];
    let mapping = DeclaredTransforms::new([(
        "model.w",
        LogicalTransformDeclaration::new(vec![
            LogicalTransformOutput::row_slice("attn.q", 0, 2),
            LogicalTransformOutput::row_slice("attn.k", 2, 2),
        ]),
    )]);
    assert!(matches!(
        compile(&headers, &no_descriptors(), &mapping, &baseline()),
        Err(LogicalWeightPlanError::KeyCollision { ref logical_key, .. })
            if logical_key == "attn.q"
    ));
}

/// An NVFP4 layer this backend keeps packed, split into two logical operands.
///
/// The split is tile-aligned, so each output keeps its own share of the nibble payload and its
/// own whole scale-factor atoms: the tensor rows partition the stored bytes exactly, the
/// swizzled block-scale surface partitions rather than multiplies, and the two **scalar** scale
/// levels are retained once per output because each packed operand owns a copy.
#[test]
fn a_packed_nvfp4_row_split_prices_every_scale_level_exactly_once_per_owner() {
    let headers = [
        header("model.qkv.weight", Dtype::U8, &[256, 128]),
        header("model.qkv.weight_scale", Dtype::F8_E4M3, &[256, 16]),
        header("model.qkv.weight_scale_2", Dtype::F32, &[]),
    ];
    let metadata = r#"{"format_version": "1.0", "layers": {"qkv": {"format": "nvfp4"}}}"#;
    let mapping = DeclaredTransforms::new([(
        "model.qkv.weight",
        LogicalTransformDeclaration::new(vec![
            LogicalTransformOutput::row_slice("attn.q", 0, 128),
            LogicalTransformOutput::row_slice("attn.k", 128, 128),
        ])
        .with_source_logical_shape(vec![256, 256]),
    )]);
    let plan = compile_logical_weight_plan_with_metadata(
        &headers,
        &no_descriptors(),
        Some(metadata),
        &mapping,
        &full(),
        &AlwaysPacked,
    )
    .expect("plan");

    assert_eq!(plan.tensor_count(), 2);
    for tensor in &plan.tensors {
        assert_eq!(tensor.codec_id, NVFP4_CODEC.codec_id);
        assert_eq!(tensor.residency.mode, ResidencyMode::Packed);
        assert_eq!(tensor.shape, vec![128, 256]);
        assert_eq!(tensor.source_shape(), [256, 256]);
        // Half of the stored nibble payload each.
        assert_eq!(tensor.residency.resident_bytes, 256 * 128 / 2);
    }
    let retained: BTreeMap<&str, u64> = plan
        .companions
        .iter()
        .map(|companion| (companion.physical_key.as_str(), companion.resident_bytes))
        .collect();
    assert_eq!(
        retained["model.qkv.weight_scale"],
        256 * 16,
        "the swizzled block scales partition across the outputs, they do not multiply"
    );
    assert_eq!(
        retained["model.qkv.weight_scale_2"],
        2 * 4,
        "each packed operand owns its own copy of the per-tensor scale"
    );
    assert_eq!(plan.resident_bytes(), 256 * 128 + 256 * 16 + 2 * 4);
    // Source bytes are still the file's, counted once.
    assert_eq!(plan.source_bytes, 256 * 128 + 256 * 16 + 4);
    assert_eq!(
        plan.tensors
            .iter()
            .map(|tensor| tensor.source_bytes)
            .sum::<u64>(),
        256 * 128
    );
}

/// The block-alignment rule the story names: a packed NVFP4 slice whose boundary cuts a
/// scale-factor atom refuses, and a half swap of packed rows refuses, because either would have
/// to re-derive the codec's scale surface. The same file plans fine under a dense policy — the
/// rule is about the packed representation, not the declaration.
#[test]
fn a_packed_block_scaled_slice_must_respect_the_scale_row_tile() {
    let headers = [
        header("model.qkv.weight", Dtype::U8, &[256, 128]),
        header("model.qkv.weight_scale", Dtype::F8_E4M3, &[256, 16]),
        header("model.qkv.weight_scale_2", Dtype::F32, &[]),
    ];
    let metadata = r#"{"format_version": "1.0", "layers": {"qkv": {"format": "nvfp4"}}}"#;
    let misaligned = LogicalTransformDeclaration::new(vec![
        LogicalTransformOutput::row_slice("attn.q", 0, 64),
        LogicalTransformOutput::row_slice("attn.k", 64, 192),
    ])
    .with_source_logical_shape(vec![256, 256]);
    let compile_with = |declaration: LogicalTransformDeclaration,
                        policy: &dyn CodecResidencyPolicy| {
        compile_logical_weight_plan_with_metadata(
            &headers,
            &no_descriptors(),
            Some(metadata),
            &DeclaredTransforms::new([("model.qkv.weight", declaration)]),
            &full(),
            policy,
        )
    };

    match compile_with(misaligned.clone(), &AlwaysPacked) {
        Err(LogicalWeightPlanError::Transform { error, .. }) => assert_eq!(
            error,
            LogicalTransformError::PackedSliceAlignment {
                logical_key: "attn.q".to_owned(),
                codec_id: NVFP4_CODEC.codec_id,
                rows: RowRange::new(0, 64),
                align: crate::comfy_quant::MXFP8_SCALE_ROW_TILE,
            }
        ),
        other => panic!("expected an alignment refusal, got {other:?}"),
    }
    // Dequantized first, the same split is an ordinary row slice.
    let dense = compile_with(misaligned, &DenseResidencyPolicy).expect("dense plan");
    assert_eq!(dense.tensors.len(), 2);

    let swapped =
        LogicalTransformDeclaration::new(vec![LogicalTransformOutput::half_swap("attn.qkv")])
            .with_source_logical_shape(vec![256, 256]);
    match compile_with(swapped, &AlwaysPacked) {
        Err(LogicalWeightPlanError::Transform { error, .. }) => assert_eq!(
            error,
            LogicalTransformError::HalfSwapOnPackedResidency {
                logical_key: "attn.qkv".to_owned(),
                codec_id: NVFP4_CODEC.codec_id,
            }
        ),
        other => panic!("expected a packed half-swap refusal, got {other:?}"),
    }
}

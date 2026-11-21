//! Select encoded operands from a component view of a file or a shard set. Metadata names
//! resolve against the complete source before selection, including companions on a different shard.
use super::*;
use std::path::PathBuf;

struct ConstructorMapping;
impl LogicalKeyMapping for ConstructorMapping {
    fn mapping_id(&self) -> &'static str {
        "source-projection-v1"
    }
    fn logical_key(&self, key: &str) -> Option<String> {
        Some(key.to_owned())
    }
    fn accepts_descriptor_shape(&self) -> bool {
        true
    }
    fn inline_fp8_scales(&self) -> bool {
        true
    }
}

/// Physical companion names that travel with a selected projection, including incompatible
/// conventions the codec validator needs to see in order to reject a contradictory declaration.
pub fn projection_companion_base(name: &str) -> Option<&str> {
    let (base, suffix) = name.rsplit_once('.')?;
    matches!(
        suffix,
        "comfy_quant"
            | "weight_scale"
            | "weight_scale_2"
            | "input_scale"
            | "scales"
            | "biases"
            | "scale_weight"
            | "scale_input"
            | "weight_s_rel"
            | "weight_s_channel"
            | "weight_codebook"
    )
    .then_some(base)
}

/// Layer-owned matrix spellings, including fused OpenCLIP parameters. The physical key is
/// retained in the plan; only companion ownership uses the normalized layer base.
pub(super) fn projection_weight_base(name: &str) -> Option<&str> {
    name.strip_suffix(".weight")
        .or_else(|| name.strip_suffix("_weight"))
}

/// Compile the selected encoded rows without imposing checkpoint-codec constraints on other
/// providers' dense, affine, codebook or integer-control fields in the same file.
pub fn plan_projection_sources(
    paths: &[PathBuf],
    selected: impl Fn(&str) -> bool,
    tables: &[String],
    registry: &CheckpointCodecRegistry,
    residency: &dyn CodecResidencyPolicy,
) -> crate::Result<Option<LogicalWeightPlan>> {
    let Some(input) = ProjectionSourceHeaders::read(paths, selected, tables)? else {
        return Ok(None);
    };
    input
        .compile(&ConstructorMapping, registry, residency)
        .map(Some)
}

/// Retained bytes of the encoded portion of a component, keyed by physical name. A consumed
/// descriptor or scale is present with zero bytes. A zero dense width omits the entire owner group.
/// Ordinary floats and unrelated integer/control buffers remain the caller's responsibility. Read only bounded declarations from the same
/// files whose complete physical header inventory was supplied by the caller.
pub fn projected_checkpoint_bytes(
    paths: &[PathBuf],
    headers: &[SafetensorsTensorHeader],
    dense_width: impl Fn(&SafetensorsTensorHeader) -> u64,
    policy: &dyn CodecResidencyPolicy,
    native_matrix: impl Fn(&str, &[usize]) -> bool,
) -> crate::Result<BTreeMap<String, u64>> {
    let Some(plan) = projected_checkpoint_plan(
        paths,
        headers,
        |_| true,
        policy,
        |key, shape| native_matrix(key, shape).then(|| shape.to_vec()),
    )?
    else {
        return Ok(BTreeMap::new());
    };
    projected_bytes_from_plan(&plan, headers, dense_width)
}

/// Price the same compiled source under a consumer's actual coefficient widths.
pub fn projected_bytes_from_plan(
    plan: &LogicalWeightPlan,
    headers: &[SafetensorsTensorHeader],
    dense_width: impl Fn(&SafetensorsTensorHeader) -> u64,
) -> crate::Result<BTreeMap<String, u64>> {
    let physical: BTreeMap<_, _> = headers.iter().map(|h| (h.name.as_str(), h)).collect();
    let mut bytes = BTreeMap::new();
    for row in &plan.tensors {
        let header = physical.get(row.physical_key.as_str()).ok_or_else(|| {
            crate::Error::Msg(format!(
                "missing projected source header {}",
                row.physical_key
            ))
        })?;
        let width = dense_width(header);
        let resident = match row.residency.mode {
            _ if width == 0 => 0,
            ResidencyMode::Packed => row.residency.resident_bytes,
            ResidencyMode::Dense => row
                .shape
                .iter()
                .try_fold(width, |count, &extent| count.checked_mul(extent as u64))
                .ok_or_else(|| {
                    crate::Error::Msg(format!(
                        "{}: dense checkpoint resident byte count overflow",
                        row.physical_key
                    ))
                })?,
        };
        bytes.insert(row.physical_key.clone(), resident);
    }
    // A bias is also independently addressable. Apply the native projection's retention rule
    // after individual rows so an encoded bias's coefficient width is charged exactly once.
    for row in &plan.tensors {
        if bytes[&row.physical_key] == 0 {
            continue;
        }
        let Some(key) = row.retained_convrot_bias_key() else {
            continue;
        };
        if let Some(bias) = physical.get(key.as_str()) {
            let width = match bias.dtype {
                Dtype::F32 | Dtype::F8_E4M3 | Dtype::F8_E5M2 => 4,
                Dtype::BF16 | Dtype::F16 => 2,
                _ => continue,
            };
            let retained = if dense_width(bias) == 0 {
                0
            } else {
                bias.materialized_bytes(width)?
            };
            bytes.insert(key, retained);
        }
    }
    for c in &plan.companions {
        let retained = if bytes[&c.owner_physical_key] == 0 {
            0
        } else {
            c.resident_bytes
        };
        bytes.insert(c.physical_key.clone(), retained);
    }
    Ok(bytes)
}

/// Compile projection rows selected by their actual operator. A component can separately
/// own other encodings (e.g. H3 W4A8) or consume an otherwise eligible matrix as coefficients.
pub fn projected_checkpoint_plan(
    paths: &[PathBuf],
    headers: &[SafetensorsTensorHeader],
    selected: impl Fn(&str) -> bool,
    policy: &dyn CodecResidencyPolicy,
    matrix_shape: impl Fn(&str, &[usize]) -> Option<Vec<usize>>,
) -> crate::Result<Option<LogicalWeightPlan>> {
    if !headers.iter().any(|h| {
        matches!(h.dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2 | Dtype::I8)
            || h.name.ends_with(".comfy_quant")
    }) {
        return Ok(None);
    }
    struct Consumer<'a, F>(&'a dyn CodecResidencyPolicy, F);
    impl<F: Fn(&str, &[usize]) -> Option<Vec<usize>>> CodecResidencyPolicy for Consumer<'_, F> {
        fn residency(
            &self,
            codec: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            shape: &[usize],
        ) -> ResidencyMode {
            self.0.residency(codec, spec, shape)
        }
        fn residency_for_key(
            &self,
            key: &str,
            codec: &CheckpointCodecRegistration,
            spec: &TensorCodecSpec,
            shape: &[usize],
        ) -> ResidencyMode {
            if let Some(matrix) = (self.1)(key, shape) {
                self.0.residency_for_key(key, codec, spec, &matrix)
            } else {
                ResidencyMode::Dense
            }
        }
    }
    let mut tables = Vec::new();
    let mut descriptors = BTreeMap::new();
    let has_descriptors = headers.iter().any(|h| h.name.ends_with(".comfy_quant"));
    for path in paths {
        if let Some(table) = crate::safetensors_path_quantization_metadata(path)? {
            tables.push(table);
        }
        if has_descriptors {
            descriptors.extend(crate::read_safetensors_tensor_payloads(
                path,
                |h| h.name.ends_with(".comfy_quant"),
                65_536,
            )?);
        }
    }
    let registry = CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied())
        .expect("valid baseline checkpoint codecs");
    plan_projection_source_headers(
        headers,
        &descriptors,
        selected,
        &tables,
        &registry,
        &Consumer(policy, matrix_shape),
    )
}

/// Inspect the encoded operands of a complete source before a model importer chooses a dialect.
/// Dense parameters and integer control buffers are left to the family. Every field of an encoded
/// leaf must have known semantics; a foreign scale spelling can never become an implicit unit
/// scale. The returned policy describes storage only and makes no device-capability claim.
pub fn classify_checkpoint_projection_sources(
    paths: &[PathBuf],
    registry: &CheckpointCodecRegistry,
) -> crate::Result<Option<LogicalWeightPlan>> {
    let mut tables = Vec::new();
    for path in paths {
        if let Some(table) = crate::safetensors_path_quantization_metadata(path)? {
            tables.push(table);
        }
    }
    plan_projection_sources(paths, |_| true, &tables, registry, &SourceCodecResidency)
}

/// Validate siblings before the projection-only view drops unrelated tensors. Named parameters
/// such as OpenCLIP's `in_proj_weight` can also own companions at `<parameter>.<field>`.
fn validate_projection_fields(
    source: &BTreeMap<String, SafetensorsTensorHeader>,
    bases: &BTreeSet<String>,
    selected: &impl Fn(&str) -> bool,
) -> crate::Result<()> {
    for name in source.keys() {
        let Some((base, field)) = name.rsplit_once('.') else {
            continue;
        };
        let parameter = source.get(base).is_some_and(|header| {
            selected(base) && matches!(header.dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2)
        });
        if !parameter && !bases.contains(base) {
            continue;
        }
        if matches!(
            field,
            "weight"
                | "bias"
                | "running_mean"
                | "running_var"
                | "num_batches_tracked"
                | "comfy_quant"
                | "weight_scale"
                | "weight_scale_2"
                | "input_scale"
                | "scale_weight"
                | "scale_input"
        ) {
            continue;
        }
        return Err(format!(
            "unrecognized checkpoint field {name:?} beside an encoded projection; its scaling or layout requires an explicit source mapping"
        ).into());
    }
    Ok(())
}

/// Plan a component that already owns validated headers and descriptor bytes (e.g. memory
/// admission). Selection and descriptor/scale semantics are identical to the file constructor.
pub fn plan_projection_source_headers(
    headers: &[SafetensorsTensorHeader],
    descriptors: &BTreeMap<String, Vec<u8>>,
    selected: impl Fn(&str) -> bool,
    tables: &[String],
    registry: &CheckpointCodecRegistry,
    policy: &dyn CodecResidencyPolicy,
) -> crate::Result<Option<LogicalWeightPlan>> {
    let source = headers
        .iter()
        .map(|h| (h.name.clone(), h.clone()))
        .collect();
    ProjectionSourceHeaders::select(source, selected, tables, |names| {
        Ok(descriptors
            .iter()
            .filter(|(k, _)| names.contains(k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect())
    })?
    .map(|s| s.compile(&ConstructorMapping, registry, policy))
    .transpose()
}

/// Validated physical rows and sidecars, reusable on relocation without rescanning shard headers.
pub struct ProjectionSourceHeaders {
    headers: Vec<SafetensorsTensorHeader>,
    descriptors: BTreeMap<String, Vec<u8>>,
    metadata: Option<String>,
}

impl ProjectionSourceHeaders {
    pub fn read(
        paths: &[PathBuf],
        selected: impl Fn(&str) -> bool,
        tables: &[String],
    ) -> crate::Result<Option<Self>> {
        let mut source = BTreeMap::new();
        for path in paths {
            for h in crate::safetensors_path_tensor_headers(path)? {
                if source.insert(h.name.clone(), h).is_some() {
                    return Err(format!(
                        "duplicate tensor across checkpoint shards in {}",
                        path.display()
                    )
                    .into());
                }
            }
        }
        Self::select(source, selected, tables, |names| {
            let mut descriptors = BTreeMap::new();
            if !names.is_empty() {
                for path in paths {
                    descriptors.extend(crate::read_safetensors_tensor_payloads(
                        path,
                        |h| names.contains(h.name.as_str()),
                        65_536,
                    )?);
                }
            }
            Ok(descriptors)
        })
    }

    /// Compile a view of a provider-owned, validated physical source inventory. Descriptor bytes
    /// must come from the same immutable source as these header rows.
    pub fn select(
        source: BTreeMap<String, SafetensorsTensorHeader>,
        selected: impl Fn(&str) -> bool,
        tables: &[String],
        read_descriptors: impl FnOnce(&BTreeSet<&str>) -> crate::Result<BTreeMap<String, Vec<u8>>>,
    ) -> crate::Result<Option<Self>> {
        let by_name: BTreeMap<_, _> = source.values().map(|h| (h.name.as_str(), h)).collect();
        let mut declarations = serde_json::Map::new();
        let mut bases = BTreeSet::new();
        for table in tables {
            let entries = crate::comfy_quant::parse_quantization_metadata(table)
                .map_err(|e| crate::Error::Msg(e.to_string()))?;
            let prefix = super::compile::resolve_projection_metadata_prefix(&by_name, &entries)
                .map_err(|e| crate::Error::Msg(e.to_string()))?;
            let json: serde_json::Value =
                serde_json::from_str(table).map_err(|e| crate::Error::Msg(e.to_string()))?;
            for name in entries.keys() {
                let base = format!("{prefix}{name}");
                if selected(&format!("{base}.weight"))
                    || (by_name.contains_key(format!("{base}_weight").as_str())
                        && selected(&format!("{base}_weight")))
                    || (by_name.contains_key(base.as_str()) && selected(&base))
                {
                    bases.insert(base.clone());
                    let descriptor = &json["layers"][name];
                    if declarations
                        .insert(base.clone(), descriptor.clone())
                        .is_some_and(|old| old != *descriptor)
                    {
                        return Err(format!(
                            "{base}: conflicting source declarations across shards"
                        )
                        .into());
                    }
                }
            }
        }
        for h in source.values() {
            if selected(&h.name)
                && matches!(h.dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2)
                && [".scale_weight", ".weight_scale"]
                    .iter()
                    .any(|suffix| source.contains_key(&format!("{}{suffix}", h.name)))
            {
                bases.insert(h.name.clone());
            }
            if let Some(base) = h.name.strip_suffix(".comfy_quant") {
                if selected(&format!("{base}.weight"))
                    || (source.contains_key(&format!("{base}_weight"))
                        && selected(&format!("{base}_weight")))
                    || selected(&h.name)
                {
                    bases.insert(base.to_owned());
                }
            }
            if let Some(base) = projection_weight_base(&h.name) {
                if selected(&h.name)
                    && (matches!(h.dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2)
                        || (h.dtype == Dtype::I8
                            && source.contains_key(&format!("{base}.weight_scale"))))
                {
                    bases.insert(base.to_owned());
                }
            }
        }
        validate_projection_fields(&source, &bases, &selected)?;
        let mut headers: Vec<_> = source
            .into_values()
            .filter(|h| {
                let companion = projection_companion_base(&h.name);
                projection_weight_base(&h.name)
                    .or(companion)
                    .is_some_and(|b| bases.contains(b))
                    || (companion.is_none()
                        && selected(&h.name)
                        && matches!(h.dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2))
                    || (!bases.is_empty() && h.name == "scaled_fp8")
            })
            .collect();
        if headers.is_empty() {
            return Ok(None);
        }
        headers.sort_by(|a, b| a.name.cmp(&b.name));
        let names: BTreeSet<_> = headers
            .iter()
            .filter(|h| h.name.ends_with(".comfy_quant"))
            .map(|h| h.name.as_str())
            .collect();
        let descriptors = read_descriptors(&names)?;
        let metadata = (!declarations.is_empty())
            .then(|| serde_json::json!({"layers":declarations}).to_string());
        Ok(Some(Self {
            headers,
            descriptors,
            metadata,
        }))
    }
    pub fn compile(
        &self,
        mapping: &dyn LogicalKeyMapping,
        registry: &CheckpointCodecRegistry,
        residency: &dyn CodecResidencyPolicy,
    ) -> crate::Result<LogicalWeightPlan> {
        compile_logical_weight_plan_with_metadata(
            &self.headers,
            &self.descriptors,
            self.metadata.as_deref(),
            mapping,
            registry,
            residency,
        )
        .map_err(|e| crate::Error::Msg(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn write(path: &Path, rows: &[(&str, &str, &[usize], &[u8])]) {
        let mut objects = serde_json::Map::new();
        let mut bytes = Vec::new();
        for &(name, dtype, shape, payload) in rows {
            let start = bytes.len();
            bytes.extend_from_slice(payload);
            objects.insert(
                name.to_owned(),
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[start,bytes.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&objects).unwrap();
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend(header);
        output.extend(bytes);
        std::fs::write(path, output).unwrap();
    }
    #[test]
    fn bare_fp8_expert_declarations_bind_the_actual_owner_for_blob_and_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("experts.safetensors");
        let key = "model.experts.gate";
        let scale = 1.25f32.to_le_bytes();
        let desc = br#"{"format":"float8_e4m3fn","full_precision_matrix_mult":true}"#;
        let encoded = (key, "F8_E4M3", &[2, 3, 2][..], &[0x38; 12][..]);
        let scales = (
            "model.experts.gate.weight_scale",
            "F32",
            &[][..],
            &scale[..],
        );
        let declared = (
            "model.experts.gate.comfy_quant",
            "U8",
            &[desc.len()][..],
            &desc[..],
        );
        let tables = vec![r#"{"layers":{"experts.gate":{"format":"float8_e4m3fn","full_precision_matrix_mult":true}}}"#.into()];
        let paths = [path.clone()];
        let registry =
            CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied()).unwrap();
        for with_blob in [false, true] {
            let mut rows = vec![encoded, scales];
            if with_blob {
                rows.push(declared);
            }
            write(&path, &rows);
            let plan = plan_projection_sources(
                &paths,
                |k| k == key,
                &tables,
                &registry,
                &SourceCodecResidency,
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.tensors[0].shape, [2, 3, 2]);
            assert!(matches!(
                plan.tensors[0].codec,
                TensorCodecSpec::ScalarFp8 {
                    full_precision_matrix_mult: true,
                    ..
                }
            ));
            assert!(plan.companions.iter().all(|c| c.owner_physical_key == key));
        }
        write(
            &path,
            &[
                encoded,
                scales,
                declared,
                (
                    "model.experts.gate.weight",
                    "F8_E4M3",
                    &[2, 3, 2],
                    &[0x38; 12],
                ),
            ],
        );
        assert!(plan_projection_sources(
            &paths,
            |_| true,
            &tables,
            &registry,
            &SourceCodecResidency
        )
        .is_err());
    }

    #[test]
    fn named_projection_parameters_bind_original_file_owners_and_reject_ambiguity() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("openclip.safetensors");
        let registry =
            CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied()).unwrap();
        let base = "model.attn.in_proj";
        let desc =
            br#"{"format":"int8_tensorwise","convrot":true,"per_row":true,"convrot_groupsize":64}"#;
        let scale = [1f32.to_le_bytes(), 2f32.to_le_bytes()].concat();
        let descriptor_shape = [desc.len()];
        for suffix in [".weight", "_weight"] {
            let key = format!("{base}{suffix}");
            let descriptor_key = format!("{base}.comfy_quant");
            let scale_key = format!("{base}.weight_scale");
            for blob in [false, true] {
                let mut rows = vec![
                    (key.as_str(), "I8", &[2, 64][..], &[0x80; 128][..]),
                    (scale_key.as_str(), "F32", &[2, 1][..], scale.as_slice()),
                ];
                if blob {
                    rows.push((descriptor_key.as_str(), "U8", &descriptor_shape, desc));
                }
                write(&path, &rows);
                let tables = vec![format!(
                    "{{\"layers\":{{\"attn.in_proj\":{}}}}}",
                    std::str::from_utf8(desc).unwrap()
                )];
                let plan = plan_projection_sources(
                    std::slice::from_ref(&path),
                    |name| name == key,
                    &tables,
                    &registry,
                    &SourceCodecResidency,
                )
                .unwrap()
                .unwrap();
                assert_eq!(plan.tensors.len(), 1);
                assert_eq!(plan.tensors[0].physical_key, key);
                assert!(matches!(
                    plan.tensors[0].codec,
                    TensorCodecSpec::ConvRotInt8 { .. }
                ));
                assert!(plan.companions.iter().all(|c| c.owner_physical_key == key));
                // Descriptor-only checkpoints must resolve the same owner without a file table.
                if blob {
                    assert!(plan_projection_sources(
                        std::slice::from_ref(&path),
                        |name| name == key,
                        &[],
                        &registry,
                        &SourceCodecResidency,
                    )
                    .unwrap()
                    .is_some());
                }
            }
        }
        write(
            &path,
            &[
                ("model.attn.in_proj.weight", "I8", &[2, 64], &[0x80; 128]),
                ("model.attn.in_proj_weight", "I8", &[2, 64], &[0x80; 128]),
                ("model.attn.in_proj.weight_scale", "F32", &[2, 1], &scale),
                ("model.attn.in_proj.comfy_quant", "U8", &[desc.len()], desc),
            ],
        );
        assert!(
            plan_projection_sources(&[path], |_| true, &[], &registry, &SourceCodecResidency,)
                .is_err()
        );
    }

    #[test]
    fn named_fp8_parameters_keep_layer_and_parameter_scale_conventions_distinct() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("openclip-fp8.safetensors");
        let registry =
            CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied()).unwrap();
        let key = "model.attn.in_proj_weight";
        let scale = 1.25f32.to_le_bytes();
        for base in ["model.attn.in_proj", key] {
            let scale_key = format!("{base}.weight_scale");
            write(
                &path,
                &[
                    (key, "F8_E4M3", &[2, 64], &[0x38; 128]),
                    (&scale_key, "F32", &[], &scale),
                ],
            );
            let plan = plan_projection_sources(
                std::slice::from_ref(&path),
                |name| name == key,
                &[],
                &registry,
                &SourceCodecResidency,
            )
            .unwrap()
            .unwrap();
            assert_eq!(plan.tensors[0].physical_key, key);
            assert!(matches!(
                plan.tensors[0].codec,
                TensorCodecSpec::ScalarFp8 { .. }
            ));
            assert_eq!(plan.companions[0].physical_key, scale_key);
            assert_eq!(plan.companions[0].owner_physical_key, key);
        }
        write(
            &path,
            &[
                (key, "F8_E4M3", &[2, 64], &[0x38; 128]),
                (
                    "model.attn.in_proj.weight",
                    "F8_E4M3",
                    &[2, 64],
                    &[0x38; 128],
                ),
                ("model.attn.in_proj.weight_scale", "F32", &[], &scale),
            ],
        );
        assert!(
            plan_projection_sources(&[path], |_| true, &[], &registry, &SourceCodecResidency,)
                .is_err()
        );
    }

    #[test]
    fn source_classification_keeps_mixed_dtypes_and_refuses_unmapped_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let paths = [dir.path().join("mixed.safetensors")];
        let registry =
            CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied()).unwrap();
        let rows: &[(&str, &str, &[usize], &[u8])] = &[
            ("unet.proj.weight", "F8_E5M2", &[2, 2], &[0x3c; 4]),
            ("unet.proj.bias", "F16", &[2], &[0; 4]),
            ("clip.position_ids", "I64", &[1], &[0; 8]),
            ("vae.conv.weight", "F16", &[2, 2], &[0; 8]),
        ];
        write(&paths[0], rows);
        let plan = classify_checkpoint_projection_sources(&paths, &registry)
            .unwrap()
            .unwrap();
        assert_eq!(plan.tensors.len(), 1);
        assert_eq!(plan.tensors[0].physical_key, "unet.proj.weight");
        for name in [
            "unet.proj.inv_s",
            "unet.proj.weight.__scale",
            "unet.proj.scales",
        ] {
            let mut contaminated = rows.to_vec();
            contaminated.push((name, "F32", &[1], &[0; 4]));
            write(&paths[0], &contaminated);
            let error = classify_checkpoint_projection_sources(&paths, &registry).unwrap_err();
            assert!(error.to_string().contains(name), "{error}");
        }
    }
    #[test]
    fn selected_shards_resolve_all_names_before_filtering_and_validate_original_codes() {
        let dir = tempfile::tempdir().unwrap();
        let shards = [
            dir.path().join("w.safetensors"),
            dir.path().join("params.safetensors"),
        ];
        let desc =
            br#"{"format":"int8_tensorwise","convrot":true,"per_row":true,"convrot_groupsize":64}"#;
        write(
            &shards[0],
            &[
                ("model.a.weight", "I8", &[1, 64], &[0x80; 64]),
                ("model.a.bias", "F32", &[1], &1f32.to_le_bytes()),
                ("model.b.weight", "F8_E4M3", &[1, 2], &[0x81, 0x33]),
            ],
        );
        write(
            &shards[1],
            &[
                ("model.a.comfy_quant", "U8", &[desc.len()], desc),
                (
                    "model.a.weight_scale",
                    "F32",
                    &[1, 1],
                    &1.0f32.to_le_bytes(),
                ),
                ("model.b.weight_scale", "F32", &[], &0.5f32.to_le_bytes()),
            ],
        );
        let headers: Vec<_> = shards
            .iter()
            .flat_map(|p| crate::safetensors_path_tensor_headers(p).unwrap())
            .collect();
        let bytes = projected_checkpoint_bytes(
            &shards,
            &headers,
            |h| if h.name == "model.a.bias" { 2 } else { 4 },
            &SourceCodecResidency,
            |_, shape| shape.len() == 2,
        )
        .unwrap();
        assert_eq!(bytes["model.a.weight"], 64);
        assert_eq!(bytes["model.a.bias"], 4);
        assert_eq!(bytes["model.a.weight_scale"], 4);
        assert_eq!(bytes["model.a.comfy_quant"], 0);
        assert_eq!(bytes["model.b.weight"], 8);
        assert_eq!(bytes["model.b.weight_scale"], 0);
        let filtered = projected_checkpoint_bytes(
            &shards,
            &headers,
            |h| if h.name == "model.a.weight" { 0 } else { 4 },
            &SourceCodecResidency,
            |_, shape| shape.len() == 2,
        )
        .unwrap();
        assert_eq!(filtered["model.a.weight"], 0);
        assert!(!filtered.contains_key("model.a.bias"));
        assert_eq!(filtered["model.a.weight_scale"], 0);
        assert_eq!(filtered["model.b.weight"], 8);
        assert!(projected_checkpoint_bytes(
            &shards,
            &headers,
            |_| 4,
            &SourceCodecResidency,
            |_, _| false
        )
        .is_err());
        let metadata = vec![
            r#"{"layers":{"a":{"format":"int8_tensorwise"},"b":{"format":"float8_e4m3fn"}}}"#
                .to_owned(),
        ];
        let registry =
            CheckpointCodecRegistry::new(BASELINE_CHECKPOINT_CODECS.iter().copied()).unwrap();
        let select = |name: &str| name == "model.a.weight";
        let plan =
            plan_projection_sources(&shards, select, &metadata, &registry, &SourceCodecResidency)
                .unwrap()
                .unwrap();
        assert_eq!(plan.tensors.len(), 1);
        assert!(matches!(
            plan.tensors[0].codec,
            TensorCodecSpec::ConvRotInt8 { .. }
        ));
        assert_eq!(plan.companions.len(), 2);
        assert_eq!(plan.source_bytes, 64 + 4 + desc.len() as u64);
        write(
            &shards[0],
            &[
                ("model.a.weight", "U8", &[1, 64], &[0x80; 64]),
                ("model.b.weight", "F8_E4M3", &[1, 2], &[0x81, 0x33]),
            ],
        );
        assert!(plan_projection_sources(
            &shards,
            select,
            &metadata,
            &registry,
            &SourceCodecResidency
        )
        .is_err());
    }
}

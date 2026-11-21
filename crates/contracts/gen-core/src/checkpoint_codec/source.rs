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

    fn select(
        source: BTreeMap<String, SafetensorsTensorHeader>,
        selected: impl Fn(&str) -> bool,
        tables: &[String],
        read_descriptors: impl FnOnce(&BTreeSet<&str>) -> crate::Result<BTreeMap<String, Vec<u8>>>,
    ) -> crate::Result<Option<Self>> {
        let by_name: BTreeMap<_, _> = source.keys().map(|s| (s.as_str(), ())).collect();
        let mut declarations = serde_json::Map::new();
        let mut bases = BTreeSet::new();
        for table in tables {
            let entries = crate::comfy_quant::parse_quantization_metadata(table)
                .map_err(|e| crate::Error::Msg(e.to_string()))?;
            let prefix = resolve_metadata_prefix(&by_name, &entries)
                .map_err(|e| crate::Error::Msg(e.to_string()))?;
            let json: serde_json::Value =
                serde_json::from_str(table).map_err(|e| crate::Error::Msg(e.to_string()))?;
            for name in entries.keys() {
                let base = format!("{prefix}{name}");
                if selected(&format!("{base}.weight")) {
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
                if selected(&format!("{base}.weight")) || selected(&h.name) {
                    bases.insert(base.to_owned());
                }
            }
            if let Some(base) = h.name.strip_suffix(".weight") {
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
                h.name
                    .strip_suffix(".weight")
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

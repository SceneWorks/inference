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
                    return Err(
                        format!("{base}: conflicting source declarations across shards").into(),
                    );
                }
            }
        }
    }
    for h in source.values() {
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
    let metadata =
        (!declarations.is_empty()).then(|| serde_json::json!({"layers":declarations}).to_string());
    compile_logical_weight_plan_with_metadata(
        &headers,
        &descriptors,
        metadata.as_deref(),
        &ConstructorMapping,
        registry,
        residency,
    )
    .map(Some)
    .map_err(|e| crate::Error::Msg(e.to_string()))
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

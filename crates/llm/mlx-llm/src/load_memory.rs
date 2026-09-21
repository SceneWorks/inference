//! Header-only upper bounds for the actual MLX checkpoint load paths.
use core_llm::{Error, LoadSpec, Result};
use serde_json::Value;
use std::{fs::File, io::Read, path::Path};

fn overflow() -> Error {
    Error::Load("MLX load memory estimate overflow".into())
}

/// Price source buffers, conversion outputs and staging without evaluating any MLX array.
pub(crate) fn required_bytes(spec: &LoadSpec) -> Result<u64> {
    let path = Path::new(&spec.source);
    if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
        let language = gguf_bytes(path, false)?;
        let projector = spec
            .projector_source
            .as_ref()
            .map(|p| gguf_bytes(Path::new(p), true))
            .transpose()?
            .unwrap_or(0);
        return language.checked_add(projector).ok_or_else(overflow);
    }
    let source = core_llm::checkpoint_payload_bytes(path)?;
    let config: Value = serde_json::from_reader(
        File::open(path.join("config.json")).map_err(|e| Error::Load(e.to_string()))?,
    )
    .map_err(|e| Error::Load(e.to_string()))?;
    if spec.quantize.is_some()
        || !matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_text" | "qwen3_vl" | "prism_hadamard_qwen35")
        )
    {
        // Other loaders and explicit quantization retain their conservative conversion bound.
        return source.checked_mul(2).ok_or_else(overflow);
    }
    let mut additional = 0u64;
    for entry in std::fs::read_dir(path).map_err(|e| Error::Load(e.to_string()))? {
        let path = entry.map_err(|e| Error::Load(e.to_string()))?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("safetensors") {
            continue;
        }
        let mut file = File::open(&path).map_err(|e| Error::Load(e.to_string()))?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| Error::Load(e.to_string()))?;
        let len = u64::from_le_bytes(prefix);
        if len > 64 * 1024 * 1024 {
            return Err(Error::Load(
                "safetensors admission header exceeds 64 MiB".into(),
            ));
        }
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)
            .map_err(|e| Error::Load(e.to_string()))?;
        let header: Value =
            serde_json::from_slice(&header).map_err(|e| Error::Load(e.to_string()))?;
        additional = additional
            .checked_add(safetensors_extra(&header)?)
            .ok_or_else(overflow)?;
    }
    source.checked_add(additional).ok_or_else(overflow)
}

fn safetensors_extra(header: &Value) -> Result<u64> {
    let mut extra = 0u64;
    for (name, info) in header
        .as_object()
        .ok_or_else(|| Error::Load("safetensors header is not an object".into()))?
    {
        if name == "__metadata__" {
            continue;
        }
        let shape = info["shape"]
            .as_array()
            .ok_or_else(|| Error::Load("safetensors tensor lacks shape".into()))?;
        let elements = shape.iter().try_fold(1u64, |n, v| {
            n.checked_mul(v.as_u64().unwrap_or(u64::MAX))
                .ok_or_else(overflow)
        })?;
        let dtype = info["dtype"]
            .as_str()
            .ok_or_else(|| Error::Load("safetensors tensor lacks dtype".into()))?;
        let vision = name.starts_with("model.visual.") || name.starts_with("vision_tower.");
        // Qwen language casts to BF16. MLX astype returns the original Array for an unchanged
        // dtype; vision clones retain their original dtype and U32 packed words remain compact.
        if !vision && matches!(dtype, "F32" | "F16" | "F64") {
            extra = extra
                .checked_add(elements.checked_mul(2).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        // Vector norms, exp(A_log) and sign transforms can retain both source and result. Four
        // bytes per element covers two additional BF16 intermediates (also for BF16 sources).
        if !vision && shape.len() <= 1 {
            extra = extra
                .checked_add(elements.checked_mul(4).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        // A channels-last patch kernel may require a contiguous transpose before reshape.
        if vision && name.ends_with("patch_embed.proj.weight") {
            extra = extra
                .checked_add(elements.checked_mul(4).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
    }
    Ok(extra)
}

fn gguf_bytes(path: &Path, projector: bool) -> Result<u64> {
    let file = crate::gguf::GgufFile::open(path).map_err(|e| Error::Load(e.to_string()))?;
    let mut retained = 0u64;
    let mut staging = 0u64;
    for tensor in &file.tensors {
        let elements = tensor
            .shape
            .iter()
            .try_fold(1u64, |n, &v| n.checked_mul(v as u64).ok_or_else(overflow))?;
        let (keep, temporary) = gguf_tensor_bytes(
            elements,
            tensor.shape.last().copied().unwrap_or(0) as u64,
            tensor.ggml_type,
            projector,
        )?;
        retained = retained.checked_add(keep).ok_or_else(overflow)?;
        staging = staging.max(temporary);
    }
    // GGUF is memory mapped. Include every source byte even though pages are reclaimable, all
    // retained outputs, and the largest tensor's simultaneous conversion/reorder host buffers.
    core_llm::checkpoint_payload_bytes(path)?
        .checked_add(retained)
        .and_then(|v| v.checked_add(staging))
        .ok_or_else(overflow)
}

fn gguf_tensor_bytes(n: u64, width: u64, kind: u32, projector: bool) -> Result<(u64, u64)> {
    if !projector && matches!(kind, 142 | 143) {
        // U32 affine words: n/4 bytes. Both F32 and F16 scale+bias arrays may coexist until
        // lazy conversion completes: n/16+n/32. Explicit signs use four bytes per column.
        let signs = width.checked_mul(4).ok_or_else(overflow)?;
        let kept = n
            .checked_mul(11)
            .and_then(|v| v.checked_div(32))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        let temporary = n
            .checked_mul(5)
            .and_then(|v| v.checked_div(16))
            .and_then(|v| v.checked_add(signs))
            .ok_or_else(overflow)?;
        Ok((kept, temporary))
    } else {
        // Dense language: F32 source + BF16 cast + norm result. Projector remains F32.
        // Reordering dense language rows also holds an old and new host Vec<f32>.
        let multiplier = if projector { 4 } else { 8 };
        let bytes = n.checked_mul(multiplier).ok_or_else(overflow)?;
        Ok((bytes, bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn qwen35_flat_and_wrapped_bf16_load_once_and_unknown_falls_back() {
        for (model_type, prefix, shared_weight_path) in [
            ("qwen3_5", "model.language_model", true),
            ("qwen3_5_text", "model", true),
            ("other", "model", false),
        ] {
            let dir = tempfile::tempdir().unwrap();
            std::fs::write(
                dir.path().join("config.json"),
                json!({"model_type": model_type}).to_string(),
            )
            .unwrap();
            let mut tensors = serde_json::Map::new();
            tensors.insert(
                format!("{prefix}.norm.weight"),
                json!({"dtype":"BF16","shape":[8],"data_offsets":[0,16]}),
            );
            let header = serde_json::to_vec(&Value::Object(tensors)).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(&header);
            bytes.extend([0; 16]);
            std::fs::write(dir.path().join("model.safetensors"), &bytes).unwrap();
            let spec = LoadSpec::dense(dir.path().display().to_string());
            let expected = if shared_weight_path {
                bytes.len() as u64 + 32
            } else {
                (bytes.len() as u64) * 2
            };
            assert_eq!(required_bytes(&spec).unwrap(), expected, "{model_type}");
        }
    }

    /// Header-only operational audit: never constructs a tensor or starts a model.
    #[test]
    #[ignore = "requires MLX_LOAD_ADMISSION_CASES containing pinned local paths and expected bounds"]
    fn pinned_header_only_load_admission_bounds() {
        let path = std::env::var("MLX_LOAD_ADMISSION_CASES").expect("set MLX_LOAD_ADMISSION_CASES");
        let cases: Value = serde_json::from_reader(File::open(path).unwrap()).unwrap();
        for case in cases.as_array().unwrap() {
            let mut spec = LoadSpec::dense(case["source"].as_str().unwrap());
            if let Some(projector) = case["projector_source"].as_str() {
                spec = spec.with_projector(projector);
            }
            let actual = required_bytes(&spec).unwrap();
            assert_eq!(
                actual,
                case["expected_bytes"].as_u64().unwrap(),
                "{}",
                spec.source
            );
            println!(
                "{}",
                json!({"source":spec.source,"projector_source":spec.projector_source,"required_bytes":actual})
            );
        }
    }

    #[test]
    fn bf16_shared_weights_are_not_double_counted_but_conversions_are() {
        let h = json!({"language_model.layers.0.weight":{"dtype":"BF16","shape":[1024,1024]},
            "model.language_model.norm.weight":{"dtype":"BF16","shape":[1024]},
            "mtp.weight":{"dtype":"F32","shape":[16,16]},
            "model.visual.proj.weight":{"dtype":"F16","shape":[32,32]}});
        assert_eq!(safetensors_extra(&h).unwrap(), 1024 * 4 + 16 * 16 * 2);
        assert!(safetensors_extra(&json!({"x":{"dtype":"F32","shape":[u64::MAX,2]}})).is_err());
    }
    #[test]
    fn gguf_counts_source_conversion_outputs_and_largest_staging() {
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 142, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 143, false).unwrap(),
            (864, 832)
        );
        assert_eq!(
            gguf_tensor_bytes(1024, 128, 1, false).unwrap(),
            (8192, 8192)
        );
        assert_eq!(gguf_tensor_bytes(1024, 128, 8, true).unwrap(), (4096, 4096));
        assert!(gguf_tensor_bytes(u64::MAX, 128, 143, false).is_err());
    }
}

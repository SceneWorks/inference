//! Source-selected adapter facts; no model or adapter factor tensor is constructed.
use std::path::Path;

use gen_core_h3::h3_adapter_memory;
use mlx_gen::gen_core::h3_adapter_memory::H3AdapterMemoryProfile;
use mlx_gen::gen_core::h3_memory::NativeProjection;
use mlx_gen::gen_core::{weightsmeta, AdapterSpec, Error, Result};

pub(super) fn profile(
    directory: &Path,
    specs: &[AdapterSpec],
    native: &[NativeProjection],
    precision: mlx_gen::Precision,
) -> Result<H3AdapterMemoryProfile> {
    if specs.is_empty() {
        return Ok(H3AdapterMemoryProfile::default());
    }
    let text = std::fs::read_to_string(directory.join("config.json"))?;
    let config = crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
        .map_err(|error| Error::Unsupported(error.to_string()))?;
    let paths = crate::adapters::adapter_target_paths(&config);
    let headers = weightsmeta::safetensors_path_tensor_headers(directory)?;
    let mut targets = h3_adapter_memory::targets_from_headers(&headers, native, &paths, 2)?;
    for target in &mut targets {
        if native.iter().any(|p| p.base == target.path) {
            continue;
        }
        // Ordinary body/refiner dense bases follow model precision via cast_weights.
        // Packed affine bases retain their source scale dtype; native bases are BF16.
        if !target.packed {
            target.activation_dtype = if precision == mlx_gen::Precision::Fp32 {
                weightsmeta::Dtype::F32
            } else {
                weightsmeta::Dtype::BF16
            };
            target.activation_width = target.activation_dtype.size() as u64;
            continue;
        }
        // H3 quant::compute_dtype uses the affine scale dtype.
        // A requested tier is not authority for either choice in a mixed source.
        let name = format!(
            "{}.{}",
            target.path,
            if target.packed { "scales" } else { "weight" }
        );
        let header = headers
            .iter()
            .find(|h| h.name == name)
            .ok_or_else(|| Error::Unsupported(format!("H3 adapter target lacks {name}")))?;
        target.activation_width = header.dtype.size() as u64;
        target.activation_dtype = header.dtype;
    }
    h3_adapter_memory::profile_stack(super::native_backend(), specs, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::h3_memory::RowDomain;
    use mlx_gen::gen_core::AdapterKind;
    use std::io::Write;

    // Genuine safetensors byte ranges with sparse payload extents. The production
    // helper must inspect headers only; no tensor backend is used by these tests.
    fn headers(path: &Path, entries: &[(String, &'static str, Vec<usize>)]) {
        let mut object = serde_json::Map::new();
        let mut end = 0u64;
        for (name, dtype, shape) in entries {
            let width = if *dtype == "F32" { 4 } else { 2 };
            let bytes = shape.iter().product::<usize>() as u64 * width;
            object.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype, "shape": shape, "data_offsets": [end, end + bytes]
                }),
            );
            end += bytes;
        }
        let mut json = serde_json::to_vec(&object).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        file.write_all(&json).unwrap();
        file.set_len(8 + json.len() as u64 + end).unwrap();
    }

    fn source(root: &Path) -> Vec<(String, &'static str, Vec<usize>)> {
        let config = serde_json::json!({
            "num_attention_heads": 56, "attention_head_dim": 128, "hidden_size": 7168,
            "num_layers": 1, "num_refiner_layers": 1, "ffn_dim": 14336,
            "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
            "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 7168,
            "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
            "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5
        });
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let mut entries = Vec::new();
        // Explicit source inventory, independent of adapter_target_paths so a
        // mapper that drops a refiner/FFN leaf does not rewrite the fixture too.
        for container in ["transformer_blocks.0", "token_refiner.refiner_blocks.0"] {
            for (leaf, output, input) in [
                ("attn.to_q", 7168, 7168),
                ("attn.to_k", 7168, 7168),
                ("attn.to_v", 7168, 7168),
                ("attn.to_out.0", 7168, 7168),
                ("ff.net.0.proj", 28672, 7168),
                ("ff.net.2", 7168, 14336),
            ] {
                // F32 source on a BF16 compute selection discriminates the MLX
                // loader's precision cast from a source-dtype activation guess.
                entries.push((
                    format!("{container}.{leaf}.weight"),
                    "F32",
                    vec![output, input],
                ));
            }
        }
        headers(&root.join("model.safetensors"), &entries);
        entries
    }

    fn resolve(root: &Path, specs: &[AdapterSpec]) -> H3AdapterMemoryProfile {
        profile(root, specs, &[], mlx_gen::Precision::Bf16).unwrap()
    }

    #[test]
    fn cpu_header_profile_routes_complete_inventory_and_f32_lora_cache() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        std::fs::create_dir(&source_dir).unwrap();
        let mut inventory = source(&source_dir);
        let adapter = temp.path().join("adapter.safetensors");
        let mut factors = Vec::new();
        for stem in [
            "transformer_blocks.0.attn.to_q",
            "token_refiner.refiner_blocks.0.ff.net.2",
        ] {
            let input = if stem.ends_with("net.2") { 14336 } else { 7168 };
            factors.push((
                format!("{stem}.lora_A.default.weight"),
                "F32",
                vec![1, input],
            ));
            factors.push((
                format!("{stem}.lora_B.default.weight"),
                "F32",
                vec![7168, 1],
            ));
        }
        headers(&adapter, &factors);
        let specs = [AdapterSpec::new(adapter, 1.0, AdapterKind::Lora)];
        let profile = resolve(&source_dir, &specs);
        assert_eq!(profile.projections.len(), 2);
        assert_eq!(profile.resident_bytes, (7168 + 7168 + 14336 + 7168) * 4);
        assert!(profile
            .projections
            .iter()
            .all(|p| p.target.activation_dtype == weightsmeta::Dtype::BF16));
        assert!(profile
            .projections
            .iter()
            .any(|p| p.target.rows == RowDomain::Conditioning));
        assert!(profile
            .projections
            .iter()
            .any(|p| p.target.rows == RowDomain::Packed));
        // Drive the production provider contract too: a correct helper disconnected from
        // ComponentBytes would otherwise leave the old file-length admission bug green.
        let spec = mlx_gen::gen_core::LoadSpec::new(mlx_gen::gen_core::WeightsSource::Dir(
            temp.path().into(),
        ))
        .with_component(
            "transformer",
            mlx_gen::gen_core::WeightsSource::Dir(source_dir.clone()),
        )
        .with_adapters(specs.to_vec());
        let contract = super::super::contract_for(&spec).unwrap();
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            (7168 + 7168 + 14336 + 7168) * 4
        );
        let native = contract
            .phase_facts
            .as_ref()
            .unwrap()
            .native_quant
            .as_ref()
            .unwrap();
        let task = native.tasks.iter().find(|t| !t.reference_task).unwrap();
        assert_eq!(task.adapter.as_ref().unwrap(), &profile);
        // Even an untargeted declared leaf belongs to the selected source census.
        inventory.pop();
        headers(&source_dir.join("model.safetensors"), &inventory);
        assert!(super::profile(&source_dir, &specs, &[], mlx_gen::Precision::Bf16).is_err());
    }

    #[test]
    fn cpu_header_profile_prices_tiny_factored_lokr_actual_installed_representation() {
        let temp = tempfile::tempdir().unwrap();
        let source_dir = temp.path().join("source");
        std::fs::create_dir(&source_dir).unwrap();
        source(&source_dir);
        let adapter = temp.path().join("lokr.safetensors");
        let stem = "transformer_blocks.0.attn.to_q";
        headers(
            &adapter,
            &[
                (format!("{stem}.lokr_w1"), "BF16", vec![4, 4]),
                (format!("{stem}.lokr_w2_a"), "BF16", vec![1792, 1]),
                (format!("{stem}.lokr_w2_b"), "BF16", vec![1, 1792]),
            ],
        );
        let file_bytes = std::fs::metadata(&adapter).unwrap().len();
        let profile = resolve(
            &source_dir,
            &[AdapterSpec::new(adapter, 1.0, AdapterKind::Lokr)],
        );
        assert_eq!(profile.projections.len(), 1);
        assert_eq!(profile.projections[0].installed_bytes, 102_760_448);
        assert_eq!(profile.resident_bytes, 102_760_448);
        assert!(profile.resident_bytes > file_bytes * 100);
        assert!(profile.install_peak_bytes > profile.projections[0].installed_bytes);
    }
}

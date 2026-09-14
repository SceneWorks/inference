//! Synthetic Wan2.2 I2V snapshots that pass `PreparedWanI2vMemory::prepare` (epic SC-22657).
//!
//! `gen_core::wan_i2v_memory` seals a receipt over a real converted snapshot: the repository /
//! revision layout, the per-route component inventory, the safetensors headers and every file's
//! digest. A backend crate that wants to test the contract a **loaded** generator publishes —
//! `Generator::memory_strategy_contract()` returns the sealed `prepared.contract` — therefore needs
//! a directory that satisfies all of that without any real weights. These writers build one per
//! route and backend from a handful of tiny tensors, then run `prepare_load_spec` so the returned
//! [`LoadSpec`] carries the prepared file pins `prepare` demands.
//!
//! The written `config.json` files are **route-honest**: an MLX snapshot declares the
//! `model_type` / `dim` its converter would, so a backend that re-parses the snapshot config at
//! load (as `mlx-gen-wan` does) resolves the same preset the loader would. Nothing here is a
//! stand-in for the real weights — every tensor is a few bytes — so only structural admission and
//! the published contract can be exercised, never a render.

use std::path::{Path, PathBuf};

use gen_core::wan_i2v_memory::{
    ensure_wan_vace_fun_source_receipt, ensure_wan_vace_source_receipt, fixture_snapshot_root,
    prepare_load_spec, WanI2vBackend, WanI2vRoute,
};
use gen_core::{LoadSpec, WeightsSource};

/// Write a minimal safetensors file whose header declares `tensors` as `(name, dtype, shape)`.
///
/// The data region is filled with a per-tensor byte so two files with different tensor names never
/// share a digest.
pub fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[usize])]) {
    std::fs::create_dir_all(path.parent().expect("a file path has a parent")).unwrap();
    let mut offset = 0_u64;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for &(name, dtype, shape) in tensors {
        let width = match dtype {
            "F32" | "U32" => 4,
            "F16" | "BF16" => 2,
            other => panic!("unsupported fixture dtype {other}"),
        };
        let bytes = shape.iter().product::<usize>() * width;
        header.insert(
            name.to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + bytes as u64],
            }),
        );
        data.resize(data.len() + bytes, name.len() as u8);
        offset += bytes as u64;
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while !json.len().is_multiple_of(8) {
        json.push(b' ');
    }
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend(json);
    bytes.extend(data);
    std::fs::write(path, bytes).unwrap();
}

/// The native MLX `config.json` a converted snapshot of `route` declares, reduced to the keys
/// `WanModelConfig::from_config_json` keys its preset selection on.
fn mlx_route_config(route: WanI2vRoute) -> &'static str {
    match route {
        WanI2vRoute::Ti2v5b => {
            r#"{"model_type":"ti2v","model_version":"2.2","dim":3072,"dual_model":false}"#
        }
        // `WanModelConfig::from_config_json` resolves `t2v` to `wan22_t2v_14b` and `i2v` to
        // `wan22_i2v_14b`; the two A14B routes therefore need *different* `model_type` strings even
        // though their file inventory is identical.
        WanI2vRoute::T2v14b => {
            r#"{"model_type":"t2v","model_version":"2.2","dim":5120,"dual_model":true}"#
        }
        WanI2vRoute::I2v14b => {
            r#"{"model_type":"i2v","model_version":"2.2","dim":5120,"dual_model":true}"#
        }
        WanI2vRoute::Vace | WanI2vRoute::VaceFun => "{}",
    }
}

/// Write a dense (bf16-tier) MLX snapshot for `route` under `base` and return the prepared
/// [`LoadSpec`] that [`PreparedWanI2vMemory::prepare`] accepts for it.
///
/// [`PreparedWanI2vMemory::prepare`]: gen_core::wan_i2v_memory::PreparedWanI2vMemory::prepare
pub fn write_mlx_snapshot(base: &Path, route: WanI2vRoute) -> LoadSpec {
    let root = fixture_snapshot_root(base, WanI2vBackend::Mlx, route);
    std::fs::create_dir_all(&root).unwrap();
    match route {
        WanI2vRoute::Ti2v5b | WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => {
            std::fs::write(root.join("config.json"), mlx_route_config(route)).unwrap();
            std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
            let files: &[(&str, usize)] = match route {
                WanI2vRoute::Ti2v5b => &[
                    ("model.safetensors", 11),
                    ("t5_encoder.safetensors", 7),
                    ("vae.safetensors", 5),
                ],
                _ => &[
                    ("high_noise_model.safetensors", 13),
                    ("low_noise_model.safetensors", 17),
                    ("t5_encoder.safetensors", 7),
                    ("vae.safetensors", 5),
                ],
            };
            for &(name, logical) in files {
                write_safetensors(&root.join(name), &[("weight", "BF16", &[logical, 64])]);
            }
        }
        WanI2vRoute::Vace | WanI2vRoute::VaceFun => {
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(
                root.join("transformer/config.json"),
                mlx_route_config(route),
            )
            .unwrap();
            if route == WanI2vRoute::VaceFun {
                std::fs::create_dir_all(root.join("transformer_2")).unwrap();
                std::fs::write(root.join("transformer_2/config.json"), "{}").unwrap();
                write_safetensors(
                    &root.join("transformer_2/model.safetensors"),
                    &[("blocks.0.self_attn.to_q.weight", "BF16", &[9, 8])],
                );
            }
            std::fs::write(root.join("tokenizer.json"), "{}").unwrap();
            write_safetensors(
                &root.join("transformer/model.safetensors"),
                &[("blocks.0.self_attn.to_q.weight", "BF16", &[8, 8])],
            );
            write_safetensors(
                &root.join("t5_encoder.safetensors"),
                &[("weight", "BF16", &[4, 4])],
            );
            // Distinct from the text encoder's size: the E1 conformance check rejects two
            // components that repeat one total.
            write_safetensors(
                &root.join("vae.safetensors"),
                &[("weight", "BF16", &[6, 4])],
            );
            seal_vace_receipt(&root, WanI2vBackend::Mlx, route);
        }
    }
    prepared_spec(root, WanI2vBackend::Mlx, route)
}

/// Write a dense (diffusers-layout) Candle snapshot for `route` under `base` and return the
/// prepared [`LoadSpec`] that [`PreparedWanI2vMemory::prepare`] accepts for it.
///
/// [`PreparedWanI2vMemory::prepare`]: gen_core::wan_i2v_memory::PreparedWanI2vMemory::prepare
pub fn write_candle_snapshot(base: &Path, route: WanI2vRoute) -> LoadSpec {
    let root = fixture_snapshot_root(base, WanI2vBackend::Candle, route);
    std::fs::create_dir_all(root.join("tokenizer")).unwrap();
    std::fs::write(root.join("tokenizer/tokenizer.json"), "{}").unwrap();
    match route {
        WanI2vRoute::Ti2v5b | WanI2vRoute::T2v14b | WanI2vRoute::I2v14b => {
            std::fs::write(root.join("model_index.json"), "{}").unwrap();
            let components: &[&str] = match route {
                WanI2vRoute::Ti2v5b => &["text_encoder", "transformer", "vae"],
                _ => &["text_encoder", "transformer", "transformer_2", "vae"],
            };
            for (index, component) in components.iter().enumerate() {
                std::fs::create_dir_all(root.join(component)).unwrap();
                std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
                let shape = if component.starts_with("transformer") {
                    vec![index + 2, 64]
                } else {
                    vec![index + 2, index + 3]
                };
                write_safetensors(
                    &root.join(component).join("model.safetensors"),
                    &[("weight", "F32", &shape)],
                );
            }
        }
        WanI2vRoute::Vace | WanI2vRoute::VaceFun => {
            // Distinct per-component sizes: the E1 conformance check rejects two components that
            // repeat one total, so a fixture whose every component weighs the same would fail a
            // provider that prices them honestly.
            for (component, rows) in [("transformer", 8), ("text_encoder", 4), ("vae", 6)]
                .into_iter()
                .chain((route == WanI2vRoute::VaceFun).then_some(("transformer_2", 9)))
            {
                std::fs::create_dir_all(root.join(component)).unwrap();
                std::fs::write(root.join(component).join("config.json"), "{}").unwrap();
                write_safetensors(
                    &root.join(component).join("model.safetensors"),
                    &[("weight", "BF16", &[rows, 4])],
                );
            }
            seal_vace_receipt(&root, WanI2vBackend::Candle, route);
        }
    }
    prepared_spec(root, WanI2vBackend::Candle, route)
}

fn seal_vace_receipt(root: &Path, backend: WanI2vBackend, route: WanI2vRoute) {
    if route == WanI2vRoute::VaceFun {
        ensure_wan_vace_fun_source_receipt(root, backend)
    } else {
        ensure_wan_vace_source_receipt(root, backend)
    }
    .expect("the synthetic VACE snapshot carries a source receipt");
}

fn prepared_spec(root: PathBuf, backend: WanI2vBackend, route: WanI2vRoute) -> LoadSpec {
    let mut spec = LoadSpec::new(WeightsSource::Dir(root)).with_resolved_route(route.provider_id());
    prepare_load_spec(&mut spec, backend, route.provider_id())
        .expect("the synthetic snapshot passes Wan I2V load-spec preparation");
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::wan_i2v_memory::PreparedWanI2vMemory;

    /// Every route on both backends seals: the writers are only useful if `prepare` accepts them.
    #[test]
    fn every_synthetic_snapshot_seals_a_receipt() {
        for route in WanI2vRoute::ALL {
            let tmp = tempfile::tempdir().unwrap();
            let spec = write_candle_snapshot(tmp.path(), route);
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Candle, route.provider_id())
                    .unwrap_or_else(|error| panic!("candle {}: {error}", route.provider_id()));
            assert_eq!(prepared.route, route);
            assert_eq!(prepared.backend, WanI2vBackend::Candle);

            let tmp = tempfile::tempdir().unwrap();
            let spec = write_mlx_snapshot(tmp.path(), route);
            let prepared =
                PreparedWanI2vMemory::prepare(&spec, WanI2vBackend::Mlx, route.provider_id())
                    .unwrap_or_else(|error| panic!("mlx {}: {error}", route.provider_id()));
            assert_eq!(prepared.route, route);
            assert_eq!(prepared.backend, WanI2vBackend::Mlx);
            // The backend-neutral seal publishes no architecture axes; that is the backends' job.
            assert!(prepared.contract.architecture_facts.is_empty());
        }
    }
}

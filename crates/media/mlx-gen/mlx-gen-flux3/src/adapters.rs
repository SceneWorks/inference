//! Native additive LoRA installation after tensor-neutral identity/key/rank admission.
use crate::dit::Flux3Dit;
use gen_core_flux3::{adapters, DitConfig};
use mlx_gen::{gen_core::AdapterSpec, weights::Weights, Error, Result};

pub fn validate(specs: &[AdapterSpec], config: &DitConfig) -> Result<()> {
    for spec in specs {
        adapters::inspect(spec, config)?;
    }
    Ok(())
}

/// The admission header rebuilt from the loaded `Weights` themselves (tensor shapes/dtypes and
/// metadata from the one `load_safetensors_with_metadata` read), so validation describes exactly
/// the arrays that get installed rather than a second open of the file.
fn header_of(weights: &Weights) -> serde_json::Value {
    let mut header = serde_json::Map::new();
    let meta: serde_json::Map<String, serde_json::Value> = adapters::ADMISSION_METADATA_KEYS
        .iter()
        .filter_map(|key| weights.metadata(key).map(|v| ((*key).to_owned(), v.into())))
        .collect();
    header.insert("__metadata__".into(), meta.into());
    for key in weights.keys() {
        let array = weights.get(key).expect("listed key");
        let dtype = match array.dtype() {
            mlx_rs::Dtype::Float32 => "F32",
            mlx_rs::Dtype::Float16 => "F16",
            mlx_rs::Dtype::Bfloat16 => "BF16",
            _ => "UNSUPPORTED",
        };
        header.insert(
            key.to_owned(),
            serde_json::json!({"shape": array.shape(), "dtype": dtype}),
        );
    }
    header.into()
}

pub fn install(dit: &mut Flux3Dit, specs: &[AdapterSpec]) -> Result<()> {
    let loaded = specs
        .iter()
        .map(|spec| {
            let weights = Weights::from_file(&spec.path).map_err(|e| {
                Error::Unsupported(format!("flux3 LoRA {}: {e}", spec.path.display()))
            })?;
            let plan = adapters::inspect_header(spec, dit.config(), &header_of(&weights))?;
            Ok((weights, plan))
        })
        .collect::<Result<Vec<_>>>()?;
    for (weights, plan) in loaded {
        for projection in plan {
            let take = |key: &str| {
                weights
                    .require(key)
                    .cloned()
                    .map_err(|e| Error::Unsupported(format!("flux3 LoRA {key}: {e}")))
            };
            let down = take(&projection.down)?;
            let up = take(&projection.up)?;
            mlx_rs::transforms::eval_pending_loads([&down, &up])?;
            dit.install_lora(&projection.target, down, up, projection.scale)?;
        }
    }
    Ok(())
}

//! Native additive LoRA installation after tensor-neutral identity/key/rank admission.
use crate::dit::Flux3Dit;
use candle_gen::{
    candle_core::DType,
    gen_core::{
        flux3::{adapters, DitConfig},
        AdapterSpec,
    },
    CandleError as Error, Result,
};

pub fn validate(specs: &[AdapterSpec], config: &DitConfig) -> Result<()> {
    for spec in specs {
        adapters::inspect(spec, config).map_err(|e| Error::Unsupported(e.to_string()))?;
    }
    Ok(())
}

pub fn install(dit: &mut Flux3Dit, specs: &[AdapterSpec]) -> Result<()> {
    // Validate the entire stack before changing even one projection.
    let plans = specs
        .iter()
        .map(|spec| {
            adapters::inspect(spec, dit.config()).map_err(|e| Error::Unsupported(e.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    for (spec, plan) in specs.iter().zip(plans) {
        let tensors = candle_gen::candle_core::safetensors::load(&spec.path, dit.device())
            .map_err(|e| Error::Unsupported(format!("flux3 LoRA {}: {e}", spec.path.display())))?;
        for projection in plan {
            let take = |key: &str| {
                tensors
                    .get(key)
                    .ok_or_else(|| Error::Unsupported(format!("flux3 LoRA: missing tensor {key}")))?
                    .to_dtype(DType::F32)
                    .map_err(Error::from)
            };
            let down = take(&projection.down)?;
            let up = take(&projection.up)?;
            dit.install_lora(&projection.target, down, up, projection.scale)
                .map_err(|e| {
                    Error::Unsupported(format!(
                        "flux3 LoRA {} target {}: {e}",
                        spec.path.display(),
                        projection.target
                    ))
                })?;
        }
    }
    Ok(())
}

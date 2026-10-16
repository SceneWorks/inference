//! Native additive LoRA installation after tensor-neutral identity/key/rank admission.
use crate::dit::Flux3Dit;
use mlx_gen::{
    gen_core::{
        flux3::{adapters, DitConfig},
        AdapterSpec,
    },
    weights::Weights,
    Error, Result,
};

pub fn validate(specs: &[AdapterSpec], config: &DitConfig) -> Result<()> {
    for spec in specs {
        adapters::inspect(spec, config)?;
    }
    Ok(())
}

pub fn install(dit: &mut Flux3Dit, specs: &[AdapterSpec]) -> Result<()> {
    let plans = specs
        .iter()
        .map(|spec| adapters::inspect(spec, dit.config()))
        .collect::<Result<Vec<_>>>()?;
    for (spec, plan) in specs.iter().zip(plans) {
        let weights = Weights::from_file(&spec.path)
            .map_err(|e| Error::Unsupported(format!("flux3 LoRA {}: {e}", spec.path.display())))?;
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

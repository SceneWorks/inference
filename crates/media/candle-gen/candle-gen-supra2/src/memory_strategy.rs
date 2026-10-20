//! Structural resident memory contract: byte facts from the safetensors headers of exactly the
//! tensors the provider materializes (shared with the MLX twin through
//! [`gen_core::supra2::memory_asset_facts`]). No calibration is claimed (the terminal campaign
//! measures).
//!
//! [`gen_core::supra2::memory_asset_facts`]: candle_gen::gen_core::supra2::memory_asset_facts

use candle_gen::gen_core::supra2::{
    memory_architecture_facts, memory_asset_facts, Supra2Config, Supra2Sources, PROVIDER_ID,
};
use candle_gen::gen_core::*;

pub(crate) fn contract_with_facts(
    cfg: &Supra2Config,
    f: MemoryAssetFacts,
) -> MemoryProviderContract {
    let mut c = MemoryProviderContract::compatibility_default(
        PROVIDER_ID,
        // Every tensor is read once from its file and lives on the device for the generator's
        // lifetime; there is no streamed or windowed path.
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    c.asset_facts = f;
    c.architecture_facts = memory_architecture_facts(cfg);
    c
}

pub(crate) fn contract_for(s: &Supra2Sources) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(
        &s.bundle.manifest.config,
        memory_asset_facts(s)?,
    ))
}

pub(crate) fn contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    contract_for(&Supra2Sources::resolve(spec)?)
}

/// Weights-free surface: the frozen geometry is known without a snapshot; byte facts are absent.
pub(crate) fn fixture(_: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(
        &Supra2Config::published(),
        MemoryAssetFacts::default(),
    ))
}

/// Text-to-image only (shared with MLX: [`supra2::memory_safety`]).
///
/// [`supra2::memory_safety`]: candle_gen::gen_core::supra2::memory_safety
pub(crate) fn safety(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    candle_gen::gen_core::supra2::memory_safety(spec, contract, context)
}

/// The one surface the loader accepts: the default (dense) tier, resident, eager.
pub(crate) fn surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.selector.tier == MemoryContractSurfaceTier::Bf16
                && s.spec.offload_policy == OffloadPolicy::Resident
                && s.spec.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_free_surface_carries_the_published_geometry() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-supra2".into()));
        let c = fixture(&spec).unwrap();
        assert_eq!(c.asset_facts, MemoryAssetFacts::default());
        let a = c.architecture_facts;
        assert_eq!(a.attention_heads, Some(9));
        assert_eq!(a.head_dim, Some(64));
        assert_eq!(a.transformer_blocks, Some(14));
        assert_eq!(a.patch_size, Some(2));
        assert_eq!(a.latent_channels, Some(4));
        assert_eq!(a.vae_spatial_scale, Some(8));
        assert!(matches!(
            c.backend,
            MemoryBackendRealization::CandleCuda {
                device_residency: true,
                ..
            }
        ));
        assert_eq!(surface_specs().len(), 1);
    }
}

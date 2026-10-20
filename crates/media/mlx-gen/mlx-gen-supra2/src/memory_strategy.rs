//! Structural resident memory contract: byte facts from the safetensors headers of exactly the
//! tensors the provider materializes. No calibration is claimed (the terminal campaign measures).

use mlx_gen::gen_core::supra2::{
    memory_architecture_facts, memory_asset_facts, Supra2Config, PROVIDER_ID,
};
use mlx_gen::gen_core::*;

use crate::model::Supra2Sources;

/// Resident bytes (shared with the Candle port): see [`memory_asset_facts`].
pub(crate) fn facts(s: &Supra2Sources) -> Result<MemoryAssetFacts> {
    memory_asset_facts(s)
}

pub(crate) fn contract_with_facts(
    cfg: &Supra2Config,
    f: MemoryAssetFacts,
) -> MemoryProviderContract {
    let mut c = MemoryProviderContract::compatibility_default(
        PROVIDER_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: false,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    c.asset_facts = f;
    c.architecture_facts = memory_architecture_facts(cfg);
    c
}

pub(crate) fn contract_for(s: &Supra2Sources) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(&s.bundle.manifest.config, facts(s)?))
}

pub(crate) fn contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    contract_for(&Supra2Sources::resolve(spec).map_err(Error::from)?)
}

/// Weights-free surface: the frozen geometry is known without a snapshot; byte facts are absent.
pub(crate) fn fixture(_: &LoadSpec) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(
        &Supra2Config::published(),
        MemoryAssetFacts::default(),
    ))
}

pub(crate) fn safety(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::supra2::memory_safety(spec, contract, context)
}

pub(crate) fn surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.selector.tier == MemoryContractSurfaceTier::Bf16
                && s.spec.offload_policy == OffloadPolicy::Resident
                && s.spec.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

#[cfg(test)]
mod cpu_tests {
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
        assert!(!surface_specs().is_empty());
    }
}

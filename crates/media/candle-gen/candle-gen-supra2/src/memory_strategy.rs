//! Structural resident memory contract: byte facts from the safetensors headers of exactly the
//! tensors the provider materializes (shared with the MLX twin through
//! [`gen_core_checkpoint::supra2::memory_asset_facts`]). No calibration is claimed (the terminal campaign
//! measures).
//!
//! [`gen_core_checkpoint::supra2::memory_asset_facts`]: gen_core_checkpoint::supra2::memory_asset_facts

use candle_gen::gen_core::*;
use gen_core_checkpoint::supra2::{
    memory_architecture_facts, memory_asset_facts, BundleCheck, Supra2Config, Supra2Sources,
    PROVIDER_ID,
};

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
    c.architecture_facts = architecture_facts(cfg);
    c
}

pub(crate) fn contract_for(s: &Supra2Sources) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(
        &s.bundle.manifest.config,
        memory_asset_facts(s)?,
    ))
}

/// Admission queries read sizes only; the load re-verifies every digest.
pub(crate) fn contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    contract_for(&Supra2Sources::resolve_checked(
        spec,
        BundleCheck::SizesOnly,
    )?)
}

/// The DiT + SD VAE architecture axes of a resolved bundle (shared derivation:
/// [`memory_architecture_facts`]). Candle publishes them only from a materialized bundle; the
/// weights-free witness carries `MemoryArchitectureFacts::default()`.
pub(crate) fn architecture_facts(cfg: &Supra2Config) -> MemoryArchitectureFacts {
    memory_architecture_facts(cfg)
}

/// Weights-free resident-only witness: the dense tier only (the loader refuses quantization), no
/// byte or architecture facts (Candle axes are gated on a materialized snapshot root).
pub(crate) fn witness(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    if spec.quantize.is_some() || spec.precision != Precision::Bf16 {
        return Err(Error::Unsupported(format!(
            "{PROVIDER_ID}: only the dense f32 checkpoint is published (no quantization or \
             precision override)"
        )));
    }
    let mut c = contract_with_facts(&Supra2Config::published(), MemoryAssetFacts::default());
    c.architecture_facts = MemoryArchitectureFacts::default();
    c.load_shape = spec.load_shape;
    Ok(c)
}

/// The dense-tier selectors across the whole residency/materialization universe.
pub(crate) fn witness_specs() -> Vec<MemoryContractSurfaceSpec> {
    candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| s.selector.tier == MemoryContractSurfaceTier::Bf16)
        .collect()
}

/// Text-to-image only (shared with MLX: [`supra2::memory_safety`]).
///
/// [`supra2::memory_safety`]: gen_core_checkpoint::supra2::memory_safety
pub(crate) fn safety(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    gen_core_checkpoint::supra2::memory_safety(spec, contract, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn architecture_facts_follow_the_published_geometry() {
        let a = architecture_facts(&Supra2Config::published());
        assert_eq!(a.attention_heads, Some(9));
        assert_eq!(a.head_dim, Some(64));
        assert_eq!(a.transformer_blocks, Some(14));
        assert_eq!(a.patch_size, Some(2));
        assert_eq!(a.latent_channels, Some(4));
        assert_eq!(a.vae_spatial_scale, Some(8));
        assert_eq!(a.activation_dtype_width, Some(4));
    }

    #[test]
    fn weights_free_witness_is_dense_resident_only_and_fact_free() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-supra2".into()));
        let c = witness(&spec).unwrap();
        assert_eq!(c.asset_facts, MemoryAssetFacts::default());
        assert_eq!(c.architecture_facts, MemoryArchitectureFacts::default());
        assert!(matches!(
            c.backend,
            MemoryBackendRealization::CandleCuda {
                device_residency: true,
                ..
            }
        ));
        let mut q = spec.clone();
        q.quantize = Some(Quant::Q4);
        assert!(witness(&q).is_err());
        let specs = witness_specs();
        assert!(!specs.is_empty());
        assert!(specs
            .iter()
            .all(|s| s.selector.tier == MemoryContractSurfaceTier::Bf16));
    }
}

//! Structural resident memory contract: byte facts from the safetensors headers of exactly the
//! tensors the provider materializes. No calibration is claimed (the terminal campaign measures).

use mlx_gen::gen_core::supra2::{Supra2Config, PROVIDER_ID};
use mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers;
use mlx_gen::gen_core::*;

use crate::model::Supra2Sources;

fn sum(path: &std::path::Path, keep: impl Fn(&str) -> bool) -> Result<u64> {
    safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|h| keep(&h.name))
        .try_fold(0u64, |acc, h| {
            acc.checked_add(h.data_bytes)
                .ok_or_else(|| Error::Unsupported("supra2: byte facts overflow".into()))
        })
}

/// Resident bytes: the f32 denoiser (+ stored conditioning), the Flan-T5 **encoder** tensors (the
/// decoder half of the checkpoint is never read), and the full f32 AutoencoderKL.
pub(crate) fn facts(s: &Supra2Sources) -> Result<MemoryAssetFacts> {
    let mut f = MemoryAssetFacts {
        transformer_bytes: sum(&s.bundle.denoiser_path(), |_| true)?,
        ..Default::default()
    };
    let conditioning = match s.bundle.conditioning_path() {
        Some(path) => sum(&path, |_| true)?,
        None => 0,
    };
    f.conditioning_bytes = sum(&s.text_encoder.join("model.safetensors"), |n| {
        n.starts_with("encoder.") || n.starts_with("shared.")
    })?
    .saturating_add(conditioning);
    f.decoder_bytes = sum(&s.vae.join("diffusion_pytorch_model.safetensors"), |_| true)?;
    f.base_bytes = f
        .transformer_bytes
        .checked_add(f.conditioning_bytes)
        .and_then(|n| n.checked_add(f.decoder_bytes))
        .ok_or_else(|| Error::Unsupported("supra2: base bytes overflow".into()))?;
    Ok(f)
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
    c.architecture_facts = MemoryArchitectureFacts {
        attention_heads: Some(cfg.num_heads as u32),
        head_dim: Some(cfg.head_dim() as u32),
        transformer_blocks: Some(cfg.depth as u32),
        patch_size: Some(cfg.patch as u32),
        latent_channels: Some(cfg.latent_channels as u32),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: None,
        activation_dtype_width: Some(4),
    };
    c
}

pub(crate) fn contract_for(s: &Supra2Sources) -> Result<MemoryProviderContract> {
    Ok(contract_with_facts(&s.bundle.manifest.config, facts(s)?))
}

pub(crate) fn contract(spec: &LoadSpec) -> Result<MemoryProviderContract> {
    contract_for(
        &Supra2Sources::resolve_checked(spec, mlx_gen::gen_core::supra2::BundleCheck::SizesOnly)
            .map_err(Error::from)?,
    )
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
    if context.geometry.reference_count != 0 || context.use_pid || context.has_phases {
        return MemorySafetyDecision::Reject {
            reason: "Supra2-IMG supports text-to-image only".into(),
        };
    }
    default_registered_memory_strategy_safety_check(spec, contract, context)
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

//! Memory-contract surface tiers, selectors and the per-backend surface spec lists.

use super::*;

/// Numeric tier named by a weights-free memory-contract surface witness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryContractSurfaceTier {
    Bf16,
    Q4,
    Q8,
    Nvfp4,
    Fp8Rowwise,
}

impl MemoryContractSurfaceTier {
    pub(super) fn load_spec(self) -> LoadSpec {
        let spec = LoadSpec::new(crate::WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        match self {
            Self::Bf16 => spec,
            Self::Q4 => spec.with_quant(crate::Quant::Q4),
            Self::Q8 => spec.with_quant(crate::Quant::Q8),
            Self::Nvfp4 => spec.with_quant(crate::Quant::Nvfp4),
            Self::Fp8Rowwise => spec.with_quant(crate::Quant::Fp8Rowwise),
        }
    }
}

/// Exact selector axes for one shipped registry-load contract witness.
///
/// Registry routes consume provisioned snapshot directories. Single-file imports are a separate,
/// ad-hoc loader surface and are intentionally not allowed to stand in for the catalog contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryContractSurfaceSelector {
    /// Numeric tier of the already-resolved catalog artifact. This is an output fact, not a request
    /// to quantize a dense [`LoadSpec`] at load time.
    pub tier: MemoryContractSurfaceTier,
    pub offload_policy: crate::OffloadPolicy,
    pub load_shape: crate::LoadShape,
}

impl MemoryContractSurfaceSelector {
    pub(super) fn matches_spec(self, spec: &LoadSpec) -> bool {
        let tier_matches = match self.tier {
            MemoryContractSurfaceTier::Bf16 => {
                spec.quantize.is_none() && spec.precision == crate::Precision::Bf16
            }
            MemoryContractSurfaceTier::Q4 => spec.quantize == Some(crate::Quant::Q4),
            MemoryContractSurfaceTier::Q8 => spec.quantize == Some(crate::Quant::Q8),
            MemoryContractSurfaceTier::Nvfp4 => spec.quantize == Some(crate::Quant::Nvfp4),
            MemoryContractSurfaceTier::Fp8Rowwise => {
                spec.quantize == Some(crate::Quant::Fp8Rowwise)
            }
        };
        tier_matches
            && self.offload_policy == spec.offload_policy
            && self.load_shape == spec.load_shape
    }

    pub fn id(self) -> &'static str {
        match (self.tier, self.offload_policy, self.load_shape) {
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "bf16:resident:eager",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "bf16:resident:deferred",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "bf16:sequential:eager",
            (
                MemoryContractSurfaceTier::Bf16,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "bf16:sequential:deferred",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "q4:resident:eager",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "q4:resident:deferred",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "q4:sequential:eager",
            (
                MemoryContractSurfaceTier::Q4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "q4:sequential:deferred",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "q8:resident:eager",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "q8:resident:deferred",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "q8:sequential:eager",
            (
                MemoryContractSurfaceTier::Q8,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "q8:sequential:deferred",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "nvfp4:resident:eager",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "nvfp4:resident:deferred",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "nvfp4:sequential:eager",
            (
                MemoryContractSurfaceTier::Nvfp4,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "nvfp4:sequential:deferred",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::EagerMaterialization,
            ) => "fp8r:resident:eager",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Resident,
                crate::LoadShape::DeferredMaterialization,
            ) => "fp8r:resident:deferred",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::EagerMaterialization,
            ) => "fp8r:sequential:eager",
            (
                MemoryContractSurfaceTier::Fp8Rowwise,
                crate::OffloadPolicy::Sequential,
                crate::LoadShape::DeferredMaterialization,
            ) => "fp8r:sequential:deferred",
        }
    }
}

/// A provider-owned, weights-free input to one contract surface.
pub struct MemoryContractSurfaceSpec {
    pub selector: MemoryContractSurfaceSelector,
    pub spec: LoadSpec,
}

impl MemoryContractSurfaceSpec {
    /// Explicit resolved artifact tier supplied to selector-aware provider fixtures.
    pub const fn resolved_artifact_tier(&self) -> MemoryContractSurfaceTier {
        self.selector.tier
    }
}

/// One fully constructed contract surface returned by [`ProviderRegistry::memory_contract_surfaces`].
pub struct MemoryContractSurface {
    pub selector: MemoryContractSurfaceSelector,
    pub spec: LoadSpec,
    pub contract: MemoryProviderContract,
    pub composed: bool,
}

impl MemoryContractSurface {
    /// Explicit resolved artifact tier emitted to downstream capability facts.
    pub const fn resolved_artifact_tier(&self) -> MemoryContractSurfaceTier {
        self.selector.tier
    }
}

pub(super) fn registry_memory_contract_surface_specs(
    tiers: &[MemoryContractSurfaceTier],
) -> Vec<MemoryContractSurfaceSpec> {
    let mut surfaces = Vec::with_capacity(tiers.len() * 4);
    for &tier in tiers {
        for offload_policy in [
            crate::OffloadPolicy::Resident,
            crate::OffloadPolicy::Sequential,
        ] {
            for load_shape in [
                crate::LoadShape::EagerMaterialization,
                crate::LoadShape::DeferredMaterialization,
            ] {
                let selector = MemoryContractSurfaceSelector {
                    tier,
                    offload_policy,
                    load_shape,
                };
                let spec = tier
                    .load_spec()
                    .with_offload_policy(offload_policy)
                    .with_load_shape(load_shape);
                surfaces.push(MemoryContractSurfaceSpec { selector, spec });
            }
        }
    }
    surfaces
}

/// Complete numeric/load-policy surface shipped by the MLX registry.
pub fn mlx_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
    ])
}

/// Common numeric/load-policy surface shipped by Candle registry providers.
pub fn candle_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
    ])
}

/// Candle surface for providers that additionally expose the explicit NVFP4 load tier.
pub fn candle_nvfp4_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
        MemoryContractSurfaceTier::Nvfp4,
    ])
}

/// Candle CUDA surface for providers that load the publisher FP8R and NVFP4 tiers as shipped
/// (FLUX3, sc-26241) beside the dense and native q4/q8 tiers. CUDA-only formats are normal: a
/// backend that cannot run them simply does not advertise this surface.
pub fn candle_publisher_memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    registry_memory_contract_surface_specs(&[
        MemoryContractSurfaceTier::Bf16,
        MemoryContractSurfaceTier::Q4,
        MemoryContractSurfaceTier::Q8,
        MemoryContractSurfaceTier::Nvfp4,
        MemoryContractSurfaceTier::Fp8Rowwise,
    ])
}

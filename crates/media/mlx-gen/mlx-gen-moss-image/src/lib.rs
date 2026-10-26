//! Native frozen MOSS-Image Base/SFT, ordinary guidance and optional data-space DMD.
mod edit;
pub mod loader;
pub mod memory_strategy;
pub mod model;
mod pipeline;
mod precision;
mod text_encoder;
mod transformer;
pub use model::{descriptor_base, descriptor_sft, load_base, load_sft, MossImage};

/// Compose both MOSS generators and their footprint/memory surfaces into an explicit registry.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    use mlx_gen::gen_core::{moss_image::MossImageVariant, MemoryContractFixtureRegistration};
    registry
        .register_generator(model::BASE_REGISTRATION)
        .register_generator(model::SFT_REGISTRATION)
        .register_memory_strategy(model::BASE_MEMORY)
        .register_memory_strategy(model::SFT_MEMORY)
        .register_memory_contract_fixture(MemoryContractFixtureRegistration {
            provider_id: "moss_image_base",
            contract: |s| memory_strategy::fixture(MossImageVariant::Base, s),
            surface_specs: memory_strategy::surface_specs,
        })
        .register_memory_contract_fixture(MemoryContractFixtureRegistration {
            provider_id: "moss_image_sft",
            contract: |s| memory_strategy::fixture(MossImageVariant::Sft, s),
            surface_specs: memory_strategy::surface_specs,
        })
}

/// Frozen MOSS canvas lattice in output pixels.
pub const SIZE_MULTIPLE: u32 = 16;

//! Native frozen MOSS-Image Base/SFT, ordinary guidance and optional data-space DMD.
mod adapters;
pub mod loader;
pub mod memory_strategy;
pub mod model;
mod pipeline;
mod precision;
mod text_encoder;
pub mod training;
mod transformer;
pub use model::{descriptor_base, descriptor_sft, load_base, load_sft, MossImage};

/// Compose both MOSS generators and their footprint/memory surfaces into an explicit registry.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    use mlx_gen::gen_core::{moss_image::MossImageVariant, MemoryContractFixtureRegistration};
    registry
        .register_trainer(training::BASE_REGISTRATION)
        .register_trainer(training::SFT_REGISTRATION)
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

#[cfg(test)]
pub(crate) struct CpuTestDevice(mlx_rs::Device);
#[cfg(test)]
pub(crate) fn cpu_test_device() -> CpuTestDevice {
    let previous = mlx_rs::Device::try_default().expect("default device metadata");
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    CpuTestDevice(previous)
}
#[cfg(test)]
impl Drop for CpuTestDevice {
    fn drop(&mut self) {
        mlx_rs::Device::set_default(&self.0);
    }
}

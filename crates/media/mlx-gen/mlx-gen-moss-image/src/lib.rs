//! Native frozen MOSS-Image Base/SFT, ordinary guidance and optional data-space DMD.
mod adapters;
mod edit;
pub mod loader;
pub mod memory_strategy;
pub mod model;
mod pipeline;
mod precision;
#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;
mod text_encoder;
pub mod training;
mod training_memory;
mod transformer;
pub use model::{descriptor_base, descriptor_sft, load_base, load_sft, MossImage};

/// Compose both MOSS generators and their footprint/memory surfaces into an explicit registry.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_checkpoint_adapter(mlx_gen::gen_core::CheckpointAdapterRegistration {
            backend_bindings: &[mlx_gen::gen_core::CheckpointBackendBindingRegistration {
                backend: mlx_gen::gen_core::CheckpointBackend::Mlx,
                source: mlx_gen::gen_core::ImportedModelSource::TransformerFile,
                operation: mlx_gen::gen_core::ImportedModelOperation::Generate,
                provider_id: "moss_image_base",
                required_components: None,
                inherit_adapters: false,
            }],
            ..mlx_gen::gen_core::MOSS_IMAGE_CHECKPOINT_ADAPTER
        })
        .register_trainer(training::BASE_REGISTRATION)
        .register_trainer(training::SFT_REGISTRATION)
        .register_route(model::route_base())
        .register_route(model::route_sft())
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

#[cfg(test)]
mod portable_exchange_tests;

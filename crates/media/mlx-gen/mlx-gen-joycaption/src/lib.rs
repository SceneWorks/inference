//! # mlx-gen-joycaption
//!
//! JoyCaption provider crate for [`mlx-gen`](mlx_gen). Linking this crate registers the
//! `fancyfeast/llama-joycaption-beta-one-hf-llava` captioner with the core caption registry.

pub mod model;

pub use model::{descriptor, load, load_joycaption, JoyCaption};

/// Add the MLX JoyCaption provider to an explicit media registry builder.
pub fn register_providers(
    registry: mlx_gen::gen_core::ProviderRegistryBuilder,
) -> mlx_gen::gen_core::ProviderRegistryBuilder {
    registry.register_captioner(model::REGISTRATION)
}

/// Build the complete explicit MLX JoyCaption provider catalog.
pub fn provider_registry() -> mlx_gen::gen_core::Result<mlx_gen::gen_core::ProviderRegistry> {
    register_providers(mlx_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_matches_inventory_compatibility_catalog() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .captioners()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        let compatibility: Vec<String> = mlx_gen::gen_core::registry::captioners()
            .filter_map(|registration| {
                let descriptor = (registration.descriptor)();
                (descriptor.family == "joycaption" && descriptor.backend == "mlx")
                    .then(|| descriptor.id.to_string())
            })
            .collect();
        assert_eq!(explicit, compatibility);
        assert_eq!(explicit, ["fancyfeast/llama-joycaption-beta-one-hf-llava"]);
    }
}

//! Stable Audio 3 provider foundation, shared primitives, and the registered small-music provider.
//!
//! `candle-audio-catalog` composes this crate into every shipped audio runtime bundle:
//!
//! - [`config`] parses both upstream `model_config.json` families and applies the frozen Python
//!   constructor defaults, including DyT for SAME-S and SAME-L;
//! - [`weights`] recognizes only caller-provisioned snapshot directories, validates their
//!   safetensors headers, and maps the two checkpoint namespaces;
//! - [`transformer`] implements the ordinary and direct-subtraction differential transformer
//!   primitives shared by the DiT and SAME families;
//! - [`pretransform`], [`softnorm`], and [`weight_norm`] cover the shared autoencoder seams;
//! - [`t5gemma`] implements the bundled encoder-only T5Gemma text conditioner;
//! - [`prepare`] provides the catalog-composed dense passthrough implementation.
//!
//! Shared audio functionality stays in [`candle_audio`]. Consumers should use
//! [`candle_audio::dsp::resample`] for sample-rate conversion rather than adding provider-local DSP.

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod config;
pub mod dit;
pub mod model;
pub mod pipeline;
pub mod prepare;
pub mod pretransform;
pub mod same;
pub mod sampler;
pub mod softnorm;
pub mod t5gemma;
pub mod transformer;
pub mod weight_norm;
pub mod weights;

pub use model::{
    descriptor, load, load_generator, StableAudio3SmallMusicGenerator, HUB_REPO, HUB_REVISION,
    MODEL_ID, REGISTRATION, WEIGHT_LICENSES,
};
pub use pipeline::StableAudio3SmallMusicPipeline;

/// Add the Stable Audio 3 small-music generator to an explicit audio registry builder.
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(model::REGISTRATION)
}

/// Build this crate's complete explicit provider registry.
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// How a Stable Audio 3 provider chooses its device.
///
/// One variant today: the lane has no per-provider device override. sc-13698 added a force-CPU one
/// for the nearest-upsample vocoders, and sc-15074 retired it — candle-core's missing Metal
/// `upsample_nearest1d` had already been routed around by `candle_audio::ops::nearest_upsample1d`
/// (sc-13691 / sc-13886), so nothing needed to opt out. The policy stays typed so a component that
/// does hit a real backend gap can add a variant without cloning the lane's device logic here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePolicy {
    Default,
}

/// Resolve a device exclusively through the shared audio-lane device surface.
pub fn resolve_device(
    policy: DevicePolicy,
) -> candle_audio::Result<candle_audio::candle_core::Device> {
    match policy {
        DevicePolicy::Default => candle_audio::default_device(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_policy_uses_shared_audio_surface() {
        let device = resolve_device(DevicePolicy::Default).unwrap();
        let shared = candle_audio::default_device().unwrap();
        assert!(device.same_device(&shared));
    }
}

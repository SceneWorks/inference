//! Stable Audio 3 provider foundation, shared primitives, and the registered small providers.
//!
//! Two post-trained 433M checkpoints are registered, in this order:
//!
//! - [`MODEL_ID`] (`stable_audio_3_small_music`, sc-14543) — text-to-music;
//! - [`SFX_MODEL_ID`] (`stable_audio_3_small_sfx`, sc-14544) — text-to-sound-effects / Foley.
//!
//! They share the DiT/SAME-S architecture, the bundled encoder-only T5Gemma stack, 44.1 kHz stereo
//! output, the 120-second logical maximum, and the eight-step Pingpong default; only the trained
//! weights — and therefore the output character — differ. Because the two shipped
//! `model_config.json` files are architecturally identical apart from the conditioner `repo_id`,
//! loading is variant-bound through [`model::load_variant`] rather than shared and unconstrained.
//!
//! `stable_audio_3_small_sfx` overlaps in intent with the shipped `moss_sfx_v2` provider but is a
//! different tier and a different output shape: SA3 SFX is **44.1 kHz stereo** up to 120 s, while
//! MOSS-SoundEffect is **48 kHz mono**. The descriptor contract cannot machine-encode domain,
//! channel count, or quality tier today, so that distinction lives in the two ids and this note;
//! adding those typed fields is tracked as `sc-15041`.
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
    descriptor, descriptor_for, load, load_generator, load_sfx_generator, load_variant,
    sfx_descriptor, sfx_load, StableAudio3SmallGenerator, Variant, HUB_REPO, HUB_REVISION,
    MODEL_ID, REGISTRATION, SFX_HUB_REPO, SFX_HUB_REVISION, SFX_MODEL_ID, SFX_REGISTRATION,
    WEIGHT_LICENSES,
};
pub use pipeline::StableAudio3SmallPipeline;

/// Add both registered Stable Audio 3 small generators to an explicit audio registry builder.
///
/// Registration order is stable and contiguous: music first (sc-14543), then SFX (sc-14544).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_generator(model::SFX_REGISTRATION)
}

/// Build this crate's complete explicit provider registry.
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

/// How a later Stable Audio 3 provider chooses its device.
///
/// `MetalIncompatible` is the per-provider override added by sc-13698. Keeping this policy typed
/// lets a later component opt out if an operation is found to be unsupported on Metal without
/// cloning the lane's cfg/device logic here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePolicy {
    Default,
    MetalIncompatible,
}

/// Resolve a device exclusively through the shared audio-lane device surface.
pub fn resolve_device(
    policy: DevicePolicy,
) -> candle_audio::Result<candle_audio::candle_core::Device> {
    match policy {
        DevicePolicy::Default => candle_audio::default_device(),
        DevicePolicy::MetalIncompatible => candle_audio::default_device_metal_incompatible(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_exposes_both_variants_in_stable_contiguous_order() {
        let registry = provider_registry().unwrap();
        let ids = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            ["stable_audio_3_small_music", "stable_audio_3_small_sfx"]
        );
    }

    #[test]
    fn device_policy_uses_shared_audio_surface() {
        let override_device = resolve_device(DevicePolicy::MetalIncompatible).unwrap();
        #[cfg(all(feature = "metal", not(feature = "cuda")))]
        assert!(override_device.is_cpu());
        #[cfg(not(all(feature = "metal", not(feature = "cuda"))))]
        {
            let default = resolve_device(DevicePolicy::Default).unwrap();
            assert!(override_device.same_device(&default));
        }
    }
}

//! Stable Audio 3 provider foundation, shared primitives, and the registered providers.
//!
//! Six checkpoints are registered, in this order:
//!
//! - [`MODEL_ID`] (`stable_audio_3_small_music`, sc-14543) — 433M text-to-music, SAME-S, 120 s;
//! - [`SFX_MODEL_ID`] (`stable_audio_3_small_sfx`, sc-14544) — 433M text-to-sound-effects / Foley,
//!   SAME-S, 120 s;
//! - [`MEDIUM_MODEL_ID`] (`stable_audio_3_medium`, sc-14545) — 1.45B differential DiT over SAME-L,
//!   380 s;
//! - [`MUSIC_BASE_MODEL_ID`], [`SFX_BASE_MODEL_ID`], [`MEDIUM_BASE_MODEL_ID`] (sc-14546) — the
//!   pre-trained flow-matching checkpoints behind the three above.
//!
//! All six share the bundled encoder-only T5Gemma stack, 44.1 kHz stereo output, and the same
//! guidance surface; every load is variant-bound through [`model::load_variant`].
//!
//! # What the `-base` half actually changes
//!
//! Not the descriptor. Both `supports_negative_prompt` and `supports_guidance` have been `true` for
//! every post-trained id since sc-14544, because [`dit::StableAudio3Dit::forward_guided`] takes its
//! batch-1 shortcut on `cfg_scale == 1.0` and never on the checkpoint. Upstream's "these parameters
//! have no effect on post-trained checkpoints" describes what post-training did to the weights, not
//! a branch in the graph.
//!
//! What changes is the operating point a request resolves to when it omits a field:
//!
//! | | objective | omitted sampler | omitted steps | omitted guidance |
//! |---|---|---|---|---|
//! | post-trained | `rf_denoiser` | Pingpong | 8 | 1.0 |
//! | `-base` | `rectified_flow` | Euler | 50 | 7.0 |
//!
//! The sampler column is upstream's own rule ([`sampler::SamplerKind::recommended`]), which was
//! dead code on the provider path until sc-14546. The steps and guidance columns are a **product
//! choice** — upstream's Python and CLI defaults stay 8 / 1.0 for every checkpoint — taken from the
//! base configs' own `training.demo` block and from Stability's Gradio app. See
//! [`pipeline::BASE_DEFAULT_STEPS`].
//!
//! # Identity
//!
//! The two smalls are architecturally identical to each other, so only the conditioner `repo_id`
//! and the pinned file hashes separate them. Medium is a different graph — `1536x24` differential
//! attention, 997 root tensors, a `16,777,216`-frame ceiling — so its wrapper rejects a small
//! snapshot on shape before identity ever comes up. See [`model::VariantShape`].
//!
//! Across the post-trained/base split the conditioner `repo_id` discriminates **nothing** — every
//! base config declares its post-trained sibling's repository — so `diffusion_objective` is the only
//! universal discriminator and is carried per variant in [`pipeline::VariantGeometry`], alongside a
//! separate provenance field. For the medium pair it is the only config-level difference that
//! exists, which leaves the SHA-256 root pin as the rest of the gate.
//!
//! # Domain coverage is documentation-only
//!
//! Stability tags `stable_audio_3_medium` for **both** `music` and `sound-effects`; the two smalls
//! are single-domain specialists. `stable_audio_3_small_sfx` additionally overlaps in intent with
//! the shipped `moss_sfx_v2` provider but is a different tier and output shape: SA3 SFX is
//! **44.1 kHz stereo**, MOSS-SoundEffect is **48 kHz mono**. The descriptor contract cannot
//! machine-encode domain, channel count, or quality tier today, so those distinctions live in the
//! ids and in these notes and nowhere else. sc-14545 accepts that explicitly rather than claiming
//! typed coverage; adding the typed fields is tracked as `sc-15041`.
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
    descriptor, descriptor_for, load, load_generator, load_medium_base_generator,
    load_medium_generator, load_music_base_generator, load_sfx_base_generator, load_sfx_generator,
    load_variant, medium_base_descriptor, medium_base_load, medium_descriptor, medium_load,
    music_base_descriptor, music_base_load, sfx_base_descriptor, sfx_base_load, sfx_descriptor,
    sfx_load, StableAudio3Generator, Variant, VariantShape, HUB_REPO, HUB_REVISION,
    MAX_DURATION_SECS, MEDIUM_BASE_HUB_REPO, MEDIUM_BASE_HUB_REVISION, MEDIUM_BASE_MODEL_ID,
    MEDIUM_BASE_REGISTRATION, MEDIUM_HUB_REPO, MEDIUM_HUB_REVISION, MEDIUM_MAX_DURATION_SECS,
    MEDIUM_MODEL_ID, MEDIUM_REGISTRATION, MEDIUM_SHAPE, MODEL_ID, MUSIC_BASE_HUB_REPO,
    MUSIC_BASE_HUB_REVISION, MUSIC_BASE_MODEL_ID, MUSIC_BASE_REGISTRATION, REGISTRATION,
    SFX_BASE_HUB_REPO, SFX_BASE_HUB_REVISION, SFX_BASE_MODEL_ID, SFX_BASE_REGISTRATION,
    SFX_HUB_REPO, SFX_HUB_REVISION, SFX_MODEL_ID, SFX_REGISTRATION, SMALL_BASE_MAX_SAMPLE_SIZE,
    SMALL_BASE_SHAPE, SMALL_MAX_SAMPLE_SIZE, SMALL_SHAPE, WEIGHT_LICENSES,
};
pub use pipeline::{ComputeDTypes, StableAudio3Pipeline, VariantGeometry};

/// Add every registered Stable Audio 3 generator to an explicit audio registry builder.
///
/// Registration order is stable and contiguous: music (sc-14543), SFX (sc-14544), medium
/// (sc-14545), then the three `-base` siblings in the same order (sc-14546). New variants append,
/// so no shipped ordered-surface assertion has to be rewritten.
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_generator(model::REGISTRATION)
        .register_generator(model::SFX_REGISTRATION)
        .register_generator(model::MEDIUM_REGISTRATION)
        .register_generator(model::MUSIC_BASE_REGISTRATION)
        .register_generator(model::SFX_BASE_REGISTRATION)
        .register_generator(model::MEDIUM_BASE_REGISTRATION)
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
    fn registry_exposes_every_variant_in_stable_contiguous_order() {
        let registry = provider_registry().unwrap();
        let ids = registry
            .generators()
            .map(|registration| (registration.descriptor)().id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            [
                "stable_audio_3_small_music",
                "stable_audio_3_small_sfx",
                "stable_audio_3_medium",
                "stable_audio_3_small_music_base",
                "stable_audio_3_small_sfx_base",
                "stable_audio_3_medium_base"
            ]
        );
        assert_eq!(
            ids,
            Variant::ALL
                .iter()
                .map(|variant| variant.model_id())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn device_policy_uses_shared_audio_surface() {
        let device = resolve_device(DevicePolicy::Default).unwrap();
        let shared = candle_audio::default_device().unwrap();
        assert!(device.same_device(&shared));
    }
}

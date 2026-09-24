//! # candle-audio-yue
//!
//! **YuE** lyrics2song (HKUST / M-A-P, Apache-2.0) for the SceneWorks Candle audio lane — epic
//! sc-19373. Full songs with vocals from structured lyrics + genre tags, returned as a 44.1 kHz mix
//! plus separate `vocals` and `instrumental` [`gen_core::AudioStem`]s.
//!
//! ## Pipeline and seams
//!
//! [`engine::YueEngine::render`] runs one song through six stage seams, each its own module with a
//! small trait (or fn) and a loader, wired together by [`stages::StageSet`]:
//!
//! | stage | module | seam | replaced by |
//! |---|---|---|---|
//! | tokenizer / prompt builder | [`tokenizer`] | [`tokenizer::PromptTokenizer::build`] | sc-19376 |
//! | ICL reference encoder | [`icl`] | [`icl::IclEncoder::encode`] | sc-19379 |
//! | stage 1 (7B Llama) | [`stage1`] | [`stage1::Stage1Model`] (`begin_render` / `begin_segment` / `step` / `end_segment`) | sc-19380 |
//! | stage 2 (1B Llama) | [`stage2`] | [`stage2::Stage2Model::upsample`] | sc-19381 |
//! | xcodec decode | [`codec`] | [`codec::CodecDecoder::decode`] | sc-19377 |
//! | Vocos upsampler | [`vocoder`] | [`vocoder::Vocoder::decode`] | sc-19378 |
//! | low-band splice | [`splice`] | [`splice::SpliceFn`] | sc-19378 |
//!
//! **Walking skeleton (sc-19382).** Every production loader currently returns its module's
//! deterministic, weights-free stub; the story named in the table replaces that module's `load`
//! (the stub stays as the end-to-end seam test's double). The engine itself — staged residency
//! (each stage released before the next loads), the stage-1 decode loop with a cancel check before
//! every step, segment-level progress, codebook-0 de-interleave, stem assembly — is final.
//!
//! ## Providers
//!
//! Six registered generators, one per stage-1 checkpoint ([`config::Variant::ALL`]):
//! `yue_{en,zh,jp_kr}_{cot,icl}`. Each loads the stage-1 snapshot from `LoadSpec::weights` and the
//! [`model::STAGE2_COMPONENT_ID`] / [`model::XCODEC_COMPONENT_ID`] snapshots from
//! `LoadSpec::components` — caller-provisioned paths, never fetched (epic 13657). Tiers: `quantize`
//! `Q8` / `Q4` assert the LM tier, unset loads the staged tier; xcodec and Vocos stay fp16 at every
//! tier (the approved epic-R2 carve-out).

pub use candle_audio;
pub use candle_audio::gen_core;

pub mod codec;
pub mod config;
pub mod engine;
pub mod icl;
pub mod model;
pub mod splice;
pub mod stage1;
pub mod stage2;
pub mod stages;
pub mod tokenizer;
pub mod tokens;
pub mod vocoder;

mod stub;

#[cfg(test)]
mod end_to_end;

pub use config::{Tier, Variant, YueRequest};
pub use engine::{YueEngine, YueEvent, YueOutput};
pub use model::{
    descriptor_for, load_with_stages, COMPONENT_LICENSES, PROVIDER_COMPONENTS, REGISTRATIONS,
};
pub use stages::StageSet;

/// Add the six YuE generators to an explicit audio registry builder (catalog composition).
pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    REGISTRATIONS
        .into_iter()
        .fold(registry, |r, reg| r.register_generator(reg))
}

/// Build this crate's own explicit provider catalog.
pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_resolves_all_six_variants_through_an_explicit_registry() {
        let registry = provider_registry().unwrap();
        let ids: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(
            ids,
            [
                "yue_en_cot",
                "yue_en_icl",
                "yue_zh_cot",
                "yue_zh_icl",
                "yue_jp_kr_cot",
                "yue_jp_kr_icl"
            ]
        );
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
    }
}

//! Explicit, complete provider catalog for the SceneWorks **Candle audio lane** — the audio
//! composition root (sc-12835, `docs/architecture/audio-backend-strategy.md`).
//!
//! Audio generation is Candle-native on every platform: this one catalog supplies the audio
//! section of `runtime-cpu`, `runtime-cuda`, **and** `runtime-macos` (where it is the
//! sanctioned cross-backend seam beside the mlx media graph). Provider crates own their
//! registrations; this crate owns only composition and stable ordering, mirroring
//! `candle-gen-catalog`. It never touches the media catalogs — bundle inclusion of the audio
//! lane is a deliberate per-bundle edit through `runtime-catalog`'s `AudioLane`.
//!
//! The audio lane carries **generators plus the audio-shaped non-generator provider kinds** the
//! epic's slices need — [`gen_core::VoiceEmbedder`] (voice-cloning identity, sc-12838),
//! [`gen_core::AudioTransform`] (non-prompt audio→audio, sc-12839), and [`gen_core::Transcriber`]
//! (audio→text ASR, the Captioner-analog, sc-12850) — validated by
//! `runtime-catalog::validate_audio`. A [`gen_core::Transcriber`] rides the audio lane (candle
//! backend) rather than the media registry where captioners (image→text on the media backend)
//! surface: a transcriber consumes an [`gen_core::AudioTrack`], so it is an audio-lane provider by
//! the same rule that placed the generators, voice embedders, and audio transforms here. Generators
//! still implement the ordinary [`gen_core::Generator`] contract with [`gen_core::Modality::Audio`]
//! descriptors; the added kinds ride the same explicit ProviderRegistry, surfaced in the bundle
//! snapshot as `audio_voice_embedder_ids` / `audio_transform_ids` / `audio_transcriber_ids` beside
//! `audio_generator_ids` — no new trait beyond the merged contracts, no linker discovery. sc-12844
//! ships the Chatterbox voice encoder (**chatterbox_ve**); sc-13223 the OpenVoice V2 transform
//! (**openvoice_v2**); sc-12850 the Whisper transcriber (**whisper_base**).
//!
//! Since sc-12836 the catalog also owns the **audio lane's snapshot-preparer composition**
//! ([`snapshot_preparer_registry`]): one `candle` registration that recognizes audio
//! snapshots (Kokoro's pickle layout, which the LLM preparer's tokenizer.json demand cannot
//! serve) and delegates everything else to `candle-llm`'s preparer unchanged. Bundles wire it
//! as `AudioLane::preparers` — the composition moved here from the three bundles so lane
//! preparation has one owner.

pub use candle_audio as audio;
pub use candle_audio::gen_core;
pub use candle_audio::gen_core::{ProviderRegistry, ProviderRegistryBuilder};
pub use candle_llm::core_llm;

/// The single tensor backend every provider in this catalog registers under — `candle` on
/// every platform per the audio backend strategy. Bundles declare the same value as their
/// audio lane's backend; `runtime-catalog` validates every descriptor against it.
pub const AUDIO_BACKEND: &str = "candle";

/// Complete audio provider package surface owned by the Candle audio lane, in catalog order.
pub mod providers {
    pub use candle_audio_acestep;
    pub use candle_audio_chatterbox;
    pub use candle_audio_chatterbox_ve;
    pub use candle_audio_clap;
    pub use candle_audio_kokoro;
    pub use candle_audio_mmaudio;
    pub use candle_audio_moss_sfx;
    pub use candle_audio_moss_tts;
    pub use candle_audio_moss_tts_realtime;
    pub use candle_audio_openvoice;
    pub use candle_audio_stable_audio_3;
    pub use candle_audio_whisper;
}

/// Add every provider shipped by the Candle audio lane to an explicit registry builder, in
/// stable catalog order: the generators first (Kokoro TTS, MOSS SFX, ACE-Step music, Stable Audio 3
/// small-music — sc-14543 and small-sfx — sc-14544, MOSS-TTS-Realtime
/// streaming TTS — sc-13392, Chatterbox clone-TTS — sc-13239, MMAudio video→audio Foley 16k — sc-12843
/// and 44.1 kHz — sc-13441), then the voice-cloning identity embedder (Chatterbox `ve`, sc-12844),
/// then the audio transforms
/// (OpenVoice V2 voice conversion, sc-13223 — the first real `AudioTransform`), then the
/// transcribers (Whisper ASR, sc-12850 — the first real `Transcriber`, the audio Captioner-analog),
/// then the audio embedders (LAION CLAP, sc-12851 — the first real `AudioEmbedder`, semantic
/// audio-text joint-space retrieval).
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    let registry = candle_audio_kokoro::register_providers(registry);
    let registry = candle_audio_moss_sfx::register_providers(registry);
    let registry = candle_audio_acestep::register_providers(registry);
    let registry = candle_audio_stable_audio_3::register_providers(registry);
    let registry = candle_audio_moss_tts_realtime::register_providers(registry);
    let registry = candle_audio_chatterbox::register_providers(registry);
    let registry = candle_audio_mmaudio::register_providers(registry);
    let registry = candle_audio_moss_tts::register_providers(registry);
    let registry = candle_audio_chatterbox_ve::register_providers(registry);
    let registry = candle_audio_openvoice::register_providers(registry);
    let registry = candle_audio_whisper::register_providers(registry);
    candle_audio_clap::register_providers(registry)
}

/// Build the complete explicit Candle audio provider catalog.
pub fn provider_registry() -> gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
}

// ---------------------------------------------------------------------------------------------
// Model-weight licences (sc-13332, migrated to schema 3 by sc-16663).
//
// A separate axis from the crate/source SPDX SBOM the release tooling already emits: each audio
// provider pins its own Hugging Face weight checkpoint, whose licence must be surfaced so a
// consumer can list it on its end-product licences page.
//
// DISCLOSURE ONLY. Nothing in this section blocks, gates, degrades or withholds anything, and
// nothing added here ever should. Each row records what an upstream text *names*; whether a given
// use is permitted is the consumer's evaluation of those facts against its own situation — its
// revenue, whether it redistributes weights, which agreements it has with its own users — and this
// crate has none of that information.
//
// Three layers (see `gen_core::license`): the reviewed licence FAMILIES, one COMPONENT row per
// loaded artifact, and the provider→component mapping whose term union is DERIVED, never
// hand-authored. `every_shipped_provider_has_a_weight_license` below refuses any provider that
// reaches this catalog without resolving to well-formed component rows.
//
// What schema 2 got wrong, and why it is gone: it stored `commercial_use`, a legal *conclusion*
// depending on facts inference does not have. Six of this catalog's providers had no correct value
// for it — the Stable Audio 3 rows all carried `commercial_use: false`, yet the Stability text does
// not forbid commercial use at all: it names a revenue threshold and a registration. A `false`
// there was not a conservative reading, it was a wrong one, and a join computed over it would have
// looked authoritative.
// ---------------------------------------------------------------------------------------------

/// The licence families every component row in this catalog resolves against — the reviewed unit,
/// shared with the media catalogs. Re-exported here so a consumer reading the audio surface has the
/// whole table in one place.
pub fn license_families() -> &'static [gen_core::LicenseFamily] {
    gen_core::LICENSE_FAMILIES
}

/// Every distinct artifact the shipped audio providers load, with the licence it declares — in
/// catalog order.
///
/// **One row per artifact, not per provider.** `ResembleAI/chatterbox` is loaded by both
/// `chatterbox_tts` and `chatterbox_ve`, and MMAudio's Synchformer and DFN5B-CLIP conditioners by
/// both Foley providers; each contributes exactly one row that several providers point at. Schema 2
/// duplicated those rows per provider, which was a second and third place for one checkpoint's terms
/// to disagree with themselves.
pub fn component_licenses() -> Vec<gen_core::ComponentLicense> {
    let mut rows = Vec::new();
    rows.extend_from_slice(candle_audio_kokoro::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_moss_sfx::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_acestep::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_stable_audio_3::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_moss_tts_realtime::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_chatterbox::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_mmaudio::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_moss_tts::COMPONENT_LICENSES);
    // Deliberately empty: `chatterbox_ve` loads the row `candle-audio-chatterbox` already owns.
    rows.extend_from_slice(candle_audio_chatterbox_ve::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_openvoice::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_whisper::COMPONENT_LICENSES);
    rows.extend_from_slice(candle_audio_clap::COMPONENT_LICENSES);
    rows
}

/// Which components each registered provider loads, in catalog order — the mapping
/// [`gen_core::provider_terms`] derives a provider's effective terms from.
pub fn provider_components() -> Vec<gen_core::ProviderComponents> {
    let mut providers = Vec::new();
    providers.extend_from_slice(candle_audio_kokoro::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_moss_sfx::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_acestep::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_stable_audio_3::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_moss_tts_realtime::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_chatterbox::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_mmaudio::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_moss_tts::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_chatterbox_ve::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_openvoice::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_whisper::PROVIDER_COMPONENTS);
    providers.extend_from_slice(candle_audio_clap::PROVIDER_COMPONENTS);
    providers
}

/// The canonical model-licences manifest JSON at `schema_version` 3 — the exact bytes committed at
/// `release/model-weight-licenses.json` and emitted into the release bundle by
/// `scripts/release/build_release.py`.
pub fn component_licenses_manifest_json() -> String {
    gen_core::component_licenses_manifest_json(
        license_families(),
        &component_licenses(),
        &provider_components(),
    )
}

/// Every provider id this catalog registers, across all provider kinds (sc-13332) — the set the
/// weight-licence ship-gate cross-checks so no registered provider can escape a recorded licence.
#[cfg(test)]
fn registered_provider_ids(registry: &ProviderRegistry) -> Vec<String> {
    let mut ids: Vec<String> = Vec::new();
    ids.extend(
        registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .transforms()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .audio_transforms()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(registry.trainers().map(|r| (r.descriptor)().id.to_string()));
    ids.extend(
        registry
            .captioners()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .transcribers()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .image_embedders()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .text_embedders()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .voice_embedders()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids.extend(
        registry
            .audio_embedders()
            .map(|r| (r.descriptor)().id.to_string()),
    );
    ids
}

// ---------------------------------------------------------------------------------------------
// Audio-lane snapshot preparation (sc-12836).
//
// The lane carries ONE preparer registration (backend-name uniqueness in the registry builder
// plus runtime-catalog's every-lane-preparer-on-the-audio-backend rule leave room for exactly
// one `candle` entry): audio-shaped snapshots take the audio path, and every other source
// delegates byte-for-byte to `candle-llm`'s preparer — the LLM preparer itself is untouched.
// ---------------------------------------------------------------------------------------------

fn lane_backend() -> &'static str {
    AUDIO_BACKEND
}

fn lane_can_prepare(spec: &core_llm::PrepareSpec) -> bool {
    candle_audio_kokoro::prepare::can_prepare(spec)
        || candle_audio_moss_sfx::prepare::can_prepare(spec)
        || candle_audio_acestep::prepare::can_prepare(spec)
        || candle_audio_stable_audio_3::prepare::can_prepare(spec)
        || candle_audio_moss_tts_realtime::prepare::can_prepare(spec)
        || candle_audio_moss_tts::prepare::can_prepare(spec)
        || candle_audio_chatterbox::prepare::can_prepare(spec)
        || candle_audio_openvoice::prepare::can_prepare(spec)
        || candle_audio_whisper::prepare::can_prepare(spec)
        || candle_audio_clap::prepare::can_prepare(spec)
        || (candle_llm::prepare::REGISTRATION.can_prepare)(spec)
}

fn lane_prepare(spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
    if candle_audio_kokoro::prepare::can_prepare(spec) {
        candle_audio_kokoro::prepare::prepare(spec)
    } else if candle_audio_moss_sfx::prepare::can_prepare(spec) {
        candle_audio_moss_sfx::prepare::prepare(spec)
    } else if candle_audio_acestep::prepare::can_prepare(spec) {
        candle_audio_acestep::prepare::prepare(spec)
    } else if candle_audio_stable_audio_3::prepare::can_prepare(spec) {
        candle_audio_stable_audio_3::prepare::prepare(spec)
    } else if candle_audio_moss_tts_realtime::prepare::can_prepare(spec) {
        candle_audio_moss_tts_realtime::prepare::prepare(spec)
    } else if candle_audio_moss_tts::prepare::can_prepare(spec) {
        candle_audio_moss_tts::prepare::prepare(spec)
    } else if candle_audio_chatterbox::prepare::can_prepare(spec) {
        candle_audio_chatterbox::prepare::prepare(spec)
    } else if candle_audio_openvoice::prepare::can_prepare(spec) {
        candle_audio_openvoice::prepare::prepare(spec)
    } else if candle_audio_whisper::prepare::can_prepare(spec) {
        candle_audio_whisper::prepare::prepare(spec)
    } else if candle_audio_clap::prepare::can_prepare(spec) {
        candle_audio_clap::prepare::prepare(spec)
    } else {
        (candle_llm::prepare::REGISTRATION.prepare)(spec)
    }
}

/// The audio lane's composed `candle` snapshot preparer (see module docs).
pub const AUDIO_LANE_PREPARER: core_llm::SnapshotPreparerRegistration =
    core_llm::SnapshotPreparerRegistration {
        backend: lane_backend,
        can_prepare: lane_can_prepare,
        prepare: lane_prepare,
    };

/// The audio lane's snapshot-preparer registry — what every bundle's `audio_lane()` wires as
/// `AudioLane::preparers` so audio model snapshots are preparable through
/// `catalog.audio_preparers()` on every platform (sc-12835's promise, made true for the
/// pickle-shaped Kokoro snapshot here).
pub fn snapshot_preparer_registry() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
    core_llm::SnapshotPreparerRegistryBuilder::new()
        .register(AUDIO_LANE_PREPARER)
        .build()
}

#[cfg(test)]
mod tests {
    /// The ordered audio id surface (the audio twin of candle-gen-catalog's
    /// `complete_catalog_has_stable_conforming_surface`). sc-12836 landed the first shipped
    /// provider (**kokoro_82m**); sc-12841 adds the SFX/ambience diffusion provider
    /// (**moss_sfx_v2**). Later stories extend this exact assertion, in catalog order.
    /// The generators-only / candle-backend / audio-modality sweeps are asserted here too so
    /// a provider that would fail bundle validation is caught in its own family first.
    #[test]
    fn complete_catalog_has_stable_conforming_surface() {
        let registry = super::provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            generators,
            [
                "kokoro_82m",
                "moss_sfx_v2",
                "acestep_v15_turbo",
                "stable_audio_3_small_music",
                "stable_audio_3_small_sfx",
                "stable_audio_3_medium",
                "stable_audio_3_small_music_base",
                "stable_audio_3_small_sfx_base",
                "stable_audio_3_medium_base",
                "moss_tts_realtime",
                "chatterbox_tts",
                "mmaudio_small_16k",
                "mmaudio_large_44k",
                "moss_ttsd_v05"
            ]
        );
        // The voice-cloning identity embedder surfaces as its own kind (sc-12844), in catalog order.
        let voice_embedders: Vec<String> = registry
            .voice_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(voice_embedders, ["chatterbox_ve"]);
        // The audio transforms surface as their own kind (sc-13223), in catalog order — OpenVoice
        // V2 voice conversion is the first real AudioTransform (releasing the sc-12839 gate).
        let audio_transforms: Vec<String> = registry
            .audio_transforms()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(audio_transforms, ["openvoice_v2"]);
        // The transcribers surface as their own kind (sc-12850), in catalog order — Whisper ASR is
        // the first real Transcriber (the audio Captioner-analog).
        let transcribers: Vec<String> = registry
            .transcribers()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(transcribers, ["whisper_base"]);
        // The audio embedders surface as their own kind (sc-12851), in catalog order — LAION CLAP is
        // the first real AudioEmbedder (semantic audio-text joint-space retrieval).
        let audio_embedders: Vec<String> = registry
            .audio_embedders()
            .map(|r| (r.descriptor)().id.to_string())
            .collect();
        assert_eq!(audio_embedders, ["clap_htsat_unfused"]);
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        // The audio lane admits generators, voice embedders, audio transforms, transcribers, and
        // audio embedders only — never the image/text/trainer/captioner kinds (media families).
        assert_eq!(registry.transforms().len(), 0);
        assert_eq!(registry.trainers().len(), 0);
        assert_eq!(registry.captioners().len(), 0);
        assert_eq!(registry.image_embedders().len(), 0);
        assert_eq!(registry.text_embedders().len(), 0);
        // Every transcriber is candle-backed.
        assert!(registry
            .transcribers()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        // Every audio transform is candle-backed.
        assert!(registry
            .audio_transforms()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        // Every audio embedder is candle-backed and the "audio-embed" family.
        assert!(registry
            .audio_embedders()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        assert!(registry
            .audio_embedders()
            .all(|r| (r.descriptor)().family == "audio-embed"));
        // Every audio generator is candle-backed and audio-modality.
        assert!(registry
            .generators()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        assert!(registry
            .generators()
            .all(|r| matches!((r.descriptor)().modality, super::gen_core::Modality::Audio)));
        // Every voice embedder is candle-backed and the "voice" family.
        assert!(registry
            .voice_embedders()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
        assert!(registry
            .voice_embedders()
            .all(|r| (r.descriptor)().family == "voice"));
    }

    /// The weight-licence ship-gate (sc-13332, extended sc-13493, migrated to schema 3 by
    /// sc-16663): every provider this catalog registers — across EVERY kind — resolves to at least
    /// one well-formed component licence row, no row is an orphan, and the whole table is
    /// conformant. Adding a provider without wiring its component rows fails here, in the
    /// composition root that decides what ships.
    ///
    /// The v2 shape this replaces demanded exactly one hand-authored `component == None` composite
    /// row per provider. There is no composite any more: a provider's terms are derived from its
    /// components, so the thing that used to be asserted cannot drift and does not need asserting.
    #[test]
    fn every_shipped_provider_has_a_weight_license() {
        use std::collections::BTreeSet;

        let registry = super::provider_registry().unwrap();
        let registered: BTreeSet<String> = super::registered_provider_ids(&registry)
            .into_iter()
            .collect();
        assert!(!registered.is_empty(), "catalog registers no providers");

        let families = super::license_families();
        let components = super::component_licenses();
        let providers = super::provider_components();

        // The whole table is well-formed: unique family ids and component keys, unique provider
        // ids, every `family` resolves, every referenced component exists, every ISO date is a real
        // calendar date, and an attribution-requiring family implies a non-blank attribution.
        assert_eq!(
            super::gen_core::license_table_conformance_errors(families, &components, &providers),
            Vec::<String>::new()
        );

        let mapped: BTreeSet<String> = providers
            .iter()
            .map(|p| p.provider_id.to_string())
            .collect();
        for id in &registered {
            assert!(
                mapped.contains(id),
                "provider '{id}' ships without a recorded model-weight licence"
            );
        }
        for id in &mapped {
            assert!(
                registered.contains(id),
                "weight-licence mapping '{id}' has no registered provider"
            );
        }

        // No orphan component rows: every row is loaded by some provider. A row nothing points at
        // is either a dead checkpoint or — worse — a live one whose provider forgot to list it.
        let referenced: BTreeSet<&str> = providers
            .iter()
            .flat_map(|p| p.components.iter().copied())
            .collect();
        for row in &components {
            assert!(
                referenced.contains(row.component),
                "component row '{}' is not loaded by any registered provider",
                row.component
            );
        }

        // Source URLs point at a real upstream: the pinned Hugging Face artifact, or the upstream
        // GitHub repository for a checkpoint whose licence lives with its code (Synchformer).
        for row in &components {
            assert!(
                row.source_url.starts_with("https://huggingface.co/")
                    || row.source_url.starts_with("https://github.com/"),
                "component '{}' source_url is not a Hugging Face or GitHub URL",
                row.component
            );
            assert_eq!(row.retrieved.len(), 10, "component '{}'", row.component);
        }

        // The full shipped surface, in catalog order: provider id, its components, and its DERIVED
        // term tags. Pinning the derived union — not a hand-typed summary — is what makes a licence
        // change visible: re-pointing one component at a different family moves the tags of every
        // provider that loads it, right here.
        let ordered: Vec<(&str, Vec<&str>, Vec<String>)> = providers
            .iter()
            .map(|p| {
                (
                    p.provider_id,
                    p.components.to_vec(),
                    super::gen_core::provider_terms(p, &components, families)
                        .iter()
                        .map(|term| term.tag().to_string())
                        .collect(),
                )
            })
            .collect();

        const APACHE: [&str; 3] = [
            "attribution_required",
            "downstream_license_copy",
            "notice_file_required",
        ];
        const MIT_ONLY: [&str; 1] = ["attribution_required"];
        // Two acceptable-use policies (Gemma's and Stability's) and two licence-copy duties stay
        // TWO elements each: a distributor of a Stable Audio 3 render hands over two documents, not
        // one, and a union that deduped them would show a user one obligation where the catalog
        // carries several.
        const SA3_UNGATED: [&str; 9] = [
            "acceptable_use_policy",
            "acceptable_use_policy",
            "attribution_required",
            "downstream_license_copy",
            "downstream_license_copy",
            "downstream_restrictions",
            "notice_file_required",
            "registration_required",
            "revenue_ceiling",
        ];
        const SA3_GATED: [&str; 10] = [
            "acceptable_use_policy",
            "acceptable_use_policy",
            "attribution_required",
            "downstream_license_copy",
            "downstream_license_copy",
            "downstream_restrictions",
            "gated_access",
            "notice_file_required",
            "registration_required",
            "revenue_ceiling",
        ];
        const MMAUDIO: [&str; 3] = [
            "attribution_required",
            "downstream_license_copy",
            "non_commercial_weights",
        ];

        let expected: Vec<(&str, Vec<&str>, Vec<&str>)> = vec![
            ("kokoro_82m", vec!["kokoro_82m"], APACHE.to_vec()),
            ("moss_sfx_v2", vec!["moss_sound_effect_v2"], APACHE.to_vec()),
            // The bundled Qwen3 text encoder is Apache-2.0 inside an otherwise-MIT repository, so
            // the provider carries Apache's notice and licence-copy duties as well as MIT's
            // attribution. v2's composite said "MIT" and left that in a prose note.
            (
                "acestep_v15_turbo",
                vec![
                    "acestep_v15_xl_turbo",
                    "acestep_v15_turbo_text_encoder",
                    "acestep_v15_sft_audio_tokenizer",
                    "acestep_v15_sft_audio_token_detokenizer",
                    "acestep_v15_sft_transformer",
                ],
                APACHE.to_vec(),
            ),
            (
                "stable_audio_3_small_music",
                vec![
                    "stable_audio_3_small_music_root",
                    "stable_audio_3_small_music_t5gemma",
                ],
                SA3_GATED.to_vec(),
            ),
            (
                "stable_audio_3_small_sfx",
                vec![
                    "stable_audio_3_small_sfx_root",
                    "stable_audio_3_small_sfx_t5gemma",
                ],
                SA3_GATED.to_vec(),
            ),
            (
                "stable_audio_3_medium",
                vec![
                    "stable_audio_3_medium_root",
                    "stable_audio_3_medium_t5gemma",
                ],
                SA3_GATED.to_vec(),
            ),
            // The three `-base` repositories are ungated, which is an acquisition difference and the
            // only difference: they ship the same two licence files, so every other tag matches.
            (
                "stable_audio_3_small_music_base",
                vec![
                    "stable_audio_3_small_music_base_root",
                    "stable_audio_3_small_music_base_t5gemma",
                ],
                SA3_UNGATED.to_vec(),
            ),
            (
                "stable_audio_3_small_sfx_base",
                vec![
                    "stable_audio_3_small_sfx_base_root",
                    "stable_audio_3_small_sfx_base_t5gemma",
                ],
                SA3_UNGATED.to_vec(),
            ),
            (
                "stable_audio_3_medium_base",
                vec![
                    "stable_audio_3_medium_base_root",
                    "stable_audio_3_medium_base_t5gemma",
                ],
                SA3_UNGATED.to_vec(),
            ),
            (
                "moss_tts_realtime",
                vec!["moss_tts_realtime_1_7b"],
                APACHE.to_vec(),
            ),
            ("chatterbox_tts", vec!["chatterbox"], MIT_ONLY.to_vec()),
            (
                "mmaudio_small_16k",
                vec![
                    "synchformer_vfeat",
                    "dfn5b_clip_vit_h14_384",
                    "mmaudio_mmdit_small_16k",
                    "mmaudio_vae_16k",
                    "mmaudio_bigvgan_16k",
                ],
                MMAUDIO.to_vec(),
            ),
            (
                "mmaudio_large_44k",
                vec![
                    "synchformer_vfeat",
                    "dfn5b_clip_vit_h14_384",
                    "mmaudio_mmdit_large_44k_v2",
                    "mmaudio_vae_44k",
                    "nvidia_bigvgan_v2_44khz_128band_512x",
                ],
                MMAUDIO.to_vec(),
            ),
            ("moss_ttsd_v05", vec!["moss_ttsd_v05"], APACHE.to_vec()),
            // The same artifact row the generator points at — one checkpoint, one row, two
            // providers.
            ("chatterbox_ve", vec!["chatterbox"], MIT_ONLY.to_vec()),
            ("openvoice_v2", vec!["openvoice_v2"], MIT_ONLY.to_vec()),
            ("whisper_base", vec!["whisper_base"], APACHE.to_vec()),
            (
                "clap_htsat_unfused",
                vec!["clap_htsat_unfused"],
                APACHE.to_vec(),
            ),
        ];
        let expected: Vec<(&str, Vec<&str>, Vec<String>)> = expected
            .into_iter()
            .map(|(id, components, tags)| {
                (
                    id,
                    components,
                    tags.into_iter().map(str::to_string).collect(),
                )
            })
            .collect();
        assert_eq!(ordered, expected);
        assert_eq!(providers.len(), 18);
        assert_eq!(components.len(), 33);
    }

    /// **The migration proof: no provider lost a term it previously carried (sc-16663).**
    ///
    /// `V2_ROWS` is the schema-2 surface exactly as it shipped — all 43 rows of the committed
    /// `release/model-weight-licenses.json` at the parent commit, frozen here so the comparison
    /// survives the deletion of the v2 types. For each row this asserts that the provider still
    /// exists, that the row's SPDX id still names a landed licence family with that *same* SPDX id
    /// (so no identity was quietly renamed), and that **every term of that family** is present in
    /// the provider's derived union. A dropped component, a mis-pointed family, or a provider that
    /// stopped loading an artifact all fail here.
    ///
    /// Note what is deliberately *not* asserted: that `commercial_use: false` implies
    /// [`gen_core::LicenseTerm::NonCommercialWeights`]. Twelve of these rows are Stability AI
    /// Community rows carrying `commercial_use: false`, and that text does not restrict commercial
    /// use — it names a revenue threshold and a registration. Asserting the implication would pin
    /// the very error this migration exists to remove.
    #[test]
    fn no_provider_lost_a_term_from_the_v2_rows() {
        use super::gen_core::{provider_terms, resolve_family};

        // (provider_id, component, spdx_id, commercial_use, carried an attribution)
        const V2_ROWS: &[(&str, Option<&str>, &str, bool, bool)] = &[
            ("acestep_v15_turbo", None, "MIT", true, true),
            (
                "acestep_v15_turbo",
                Some("audio_token_detokenizer"),
                "MIT",
                true,
                true,
            ),
            (
                "acestep_v15_turbo",
                Some("audio_tokenizer"),
                "MIT",
                true,
                true,
            ),
            ("acestep_v15_turbo", Some("transformer"), "MIT", true, true),
            ("chatterbox_tts", None, "MIT", true, true),
            ("chatterbox_ve", None, "MIT", true, true),
            ("clap_htsat_unfused", None, "Apache-2.0", true, true),
            ("kokoro_82m", None, "Apache-2.0", true, true),
            (
                "mmaudio_large_44k",
                None,
                "LicenseRef-MMAudio-large-44k-composite",
                false,
                true,
            ),
            (
                "mmaudio_large_44k",
                Some("dfn5b_clip_vit_h14_384"),
                "LicenseRef-Apple-MLR",
                false,
                true,
            ),
            (
                "mmaudio_large_44k",
                Some("mmaudio_mmdit_large_44k_v2"),
                "CC-BY-NC-4.0",
                false,
                true,
            ),
            (
                "mmaudio_large_44k",
                Some("mmaudio_vae_44k"),
                "CC-BY-NC-4.0",
                false,
                true,
            ),
            (
                "mmaudio_large_44k",
                Some("nvidia_bigvgan_v2_44khz_128band_512x"),
                "MIT",
                true,
                true,
            ),
            (
                "mmaudio_large_44k",
                Some("synchformer_vfeat"),
                "MIT",
                true,
                true,
            ),
            (
                "mmaudio_small_16k",
                None,
                "LicenseRef-MMAudio-small-16k-composite",
                false,
                true,
            ),
            (
                "mmaudio_small_16k",
                Some("dfn5b_clip_vit_h14_384"),
                "LicenseRef-Apple-MLR",
                false,
                true,
            ),
            (
                "mmaudio_small_16k",
                Some("mmaudio_bigvgan_16k"),
                "CC-BY-NC-4.0",
                false,
                true,
            ),
            (
                "mmaudio_small_16k",
                Some("mmaudio_mmdit_small_16k"),
                "CC-BY-NC-4.0",
                false,
                true,
            ),
            (
                "mmaudio_small_16k",
                Some("mmaudio_vae_16k"),
                "CC-BY-NC-4.0",
                false,
                true,
            ),
            (
                "mmaudio_small_16k",
                Some("synchformer_vfeat"),
                "MIT",
                true,
                true,
            ),
            ("moss_sfx_v2", None, "Apache-2.0", true, true),
            ("moss_tts_realtime", None, "Apache-2.0", true, true),
            ("moss_ttsd_v05", None, "Apache-2.0", true, true),
            ("openvoice_v2", None, "MIT", true, true),
            (
                "stable_audio_3_medium",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_medium",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_medium",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            (
                "stable_audio_3_medium_base",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_medium_base",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_medium_base",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            (
                "stable_audio_3_small_music",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_music",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_music",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            (
                "stable_audio_3_small_music_base",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_music_base",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_music_base",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            (
                "stable_audio_3_small_sfx",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_sfx",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_sfx",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            (
                "stable_audio_3_small_sfx_base",
                None,
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_sfx_base",
                Some("root"),
                "LicenseRef-Stability-AI-Community",
                false,
                true,
            ),
            (
                "stable_audio_3_small_sfx_base",
                Some("t5gemma"),
                "LicenseRef-Gemma-Terms",
                true,
                true,
            ),
            ("whisper_base", None, "Apache-2.0", true, true),
        ];
        assert_eq!(V2_ROWS.len(), 43);

        // The only two v2 rows with no schema-3 counterpart: the hand-authored MMAudio composites,
        // whose whole content is now derived. `mmaudio_derived_composite_covers_the_hand_authored_one`
        // is where that drop is justified term by term.
        const DERIVED_NOW: &[&str] = &[
            "LicenseRef-MMAudio-small-16k-composite",
            "LicenseRef-MMAudio-large-44k-composite",
        ];

        let families = super::license_families();
        let components = super::component_licenses();
        let providers = super::provider_components();

        let v2_providers: std::collections::BTreeSet<&str> =
            V2_ROWS.iter().map(|(id, ..)| *id).collect();
        assert_eq!(v2_providers.len(), 18);

        for id in &v2_providers {
            assert!(
                providers.iter().any(|p| p.provider_id == *id),
                "provider '{id}' had licence rows in v2 and has none now"
            );
        }

        // The frozen `commercial_use` column is not decoration: twelve rows carried `false` under
        // the Stability AI Community License, whose text names a revenue threshold and a
        // registration and prohibits nothing. Migrating that `false` into
        // `LicenseTerm::NonCommercialWeights` would have transcribed the epic's central error into
        // the new schema, so assert it did not happen — the derived union carries the threshold and
        // the registration as facts, and no non-commercial term at all.
        let mut wrongly_non_commercial = 0;
        for (provider_id, _component, spdx_id, commercial_use, _attribution) in V2_ROWS {
            if *commercial_use || *spdx_id != "LicenseRef-Stability-AI-Community" {
                continue;
            }
            wrongly_non_commercial += 1;
            let provider = providers
                .iter()
                .find(|p| p.provider_id == *provider_id)
                .unwrap();
            let terms = provider_terms(provider, &components, families);
            assert!(
                !terms.contains(&super::gen_core::LicenseTerm::NonCommercialWeights),
                "provider '{provider_id}' must not inherit a non-commercial term from v2's \
                 commercial_use: false — the Stability text states no such restriction"
            );
            assert!(
                terms
                    .iter()
                    .any(|t| t.tag() == "revenue_ceiling" || t.tag() == "registration_required"),
                "provider '{provider_id}' must disclose the threshold and registration the \
                 Stability text actually names"
            );
        }
        assert_eq!(wrongly_non_commercial, 12);

        for (provider_id, component, spdx_id, _commercial_use, has_attribution) in V2_ROWS {
            if DERIVED_NOW.contains(spdx_id) {
                continue;
            }
            let provider = providers
                .iter()
                .find(|p| p.provider_id == *provider_id)
                .unwrap();
            let terms = provider_terms(provider, &components, families);

            // The v2 SPDX id still names a landed family, and that family still declares the same
            // SPDX id — so a licence identity cannot have been renamed under the migration.
            let family = families
                .iter()
                .find(|f| f.spdx_id == *spdx_id)
                .unwrap_or_else(|| {
                    panic!(
                        "v2 row ({provider_id}, {component:?}) declared {spdx_id}, which matches no \
                         landed licence family"
                    )
                });
            assert!(
                resolve_family(families, family.id).is_some(),
                "{spdx_id} resolves"
            );

            // Every term that family's text states is in the provider's derived union.
            for term in family.terms {
                assert!(
                    terms.contains(term),
                    "provider '{provider_id}' carried {spdx_id} in v2 (component {component:?}) but \
                     its derived terms are missing {}",
                    term.tag()
                );
            }

            // The provider loads at least one artifact under that family, so the term did not
            // arrive by coincidence from an unrelated component.
            assert!(
                provider
                    .components
                    .iter()
                    .filter_map(|key| super::gen_core::resolve_component(&components, key))
                    .any(|row| row.family == family.id),
                "provider '{provider_id}' carried {spdx_id} in v2 but loads no component under it"
            );

            if *has_attribution {
                assert!(
                    provider
                        .components
                        .iter()
                        .filter_map(|key| super::gen_core::resolve_component(&components, key))
                        .any(|row| row.attribution.is_some_and(|text| !text.trim().is_empty())),
                    "provider '{provider_id}' carried an attribution in v2 and records none now"
                );
            }
        }
    }

    /// The MMAudio composite comparison sc-16663 asked for, run as an assertion rather than left as
    /// a claim in a PR description.
    ///
    /// v2 hand-typed one composite row per MMAudio provider, summarizing "the intersection of five
    /// component licenses" as prose. Schema 3 derives it. The derived union **covers** the
    /// hand-authored one and adds one duty the composite never mentioned: Apple MLR §2 requires
    /// providing a copy of the agreement to any third party the model is redistributed to, which a
    /// strictest-wins prose summary had nowhere to put. So the old row was not merely redundant, it
    /// was incomplete — recorded here rather than papered over.
    ///
    /// The second finding: the two composites carried *different* identifiers
    /// (`…-small-16k-composite`, `…-large-44k-composite`) for term sets that are identical. Swapping
    /// the CC-BY-NC-4.0 16k BigVGAN for the MIT NVIDIA BigVGAN v2 changes no term, because MIT's
    /// only condition is attribution and four other components already impose it.
    #[test]
    fn mmaudio_derived_composite_covers_the_hand_authored_one() {
        use super::gen_core::LicenseTerm;

        let families = super::license_families();
        let components = super::component_licenses();
        let providers = super::provider_components();
        let terms_for = |id: &str| {
            let provider = providers.iter().find(|p| p.provider_id == id).unwrap();
            super::gen_core::provider_terms(provider, &components, families)
        };

        let small = terms_for("mmaudio_small_16k");
        let large = terms_for("mmaudio_large_44k");

        // What the hand-authored composites actually disclosed: a non-commercial weights
        // restriction (their "research / non-commercial only") and an attribution obligation (they
        // carried an attribution string naming all five upstreams).
        for terms in [&small, &large] {
            assert!(terms.contains(&LicenseTerm::NonCommercialWeights));
            assert!(terms.contains(&LicenseTerm::AttributionRequired));
        }

        // What they missed. Apple MLR's flow-down survives into the derived union.
        for terms in [&small, &large] {
            assert!(
                terms.contains(&LicenseTerm::DownstreamLicenseCopy {
                    family: "apple-mlr"
                }),
                "the derived union must carry the Apple MLR licence-copy duty the composite omitted"
            );
        }

        // The composites' prose said "excludes any commercial product or service", which reads as
        // an outputs restriction. It is not transcribed as one: the Apple and CC-BY-NC texts both
        // restrict the licensed material and are silent on generated audio, and sc-16662 withheld
        // that inference deliberately. Silence stays silence.
        for terms in [&small, &large] {
            assert!(!terms.contains(&LicenseTerm::NonCommercialOutputs));
        }

        // Two hand-typed composite identifiers, one term set.
        assert_eq!(
            small, large,
            "the two MMAudio composites named different licences for identical terms"
        );
    }

    /// The committed `release/model-weight-licenses.json` is byte-for-byte what the catalog
    /// produces (sc-13332, schema 3 since sc-16663) — the drift gate tying the release manifest the
    /// tooling emits to the per-provider source of truth. Regenerate with
    /// `UPDATE_WEIGHT_LICENSES=1 cargo test -p candle-audio-catalog
    /// component_licenses_manifest_matches_committed_file`.
    #[test]
    fn component_licenses_manifest_matches_committed_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../release/model-weight-licenses.json");
        let generated = super::component_licenses_manifest_json();
        if std::env::var_os("UPDATE_WEIGHT_LICENSES").is_some() {
            std::fs::write(&path, &generated).unwrap();
        }
        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read committed model-weight-licenses manifest at {}: {e} (regenerate with \
                 UPDATE_WEIGHT_LICENSES=1)",
                path.display()
            )
        });
        assert_eq!(
            committed, generated,
            "release/model-weight-licenses.json is stale — regenerate with UPDATE_WEIGHT_LICENSES=1"
        );
        assert!(generated.contains("\"schema_version\": 3"));
        assert!(generated.contains("\"kind\": \"model-weight-licenses\""));
        // A consumer reads one file and finds all three layers in it.
        for section in ["\"families\": [", "\"components\": [", "\"providers\": ["] {
            assert!(generated.contains(section), "manifest is missing {section}");
        }
        // The legal conclusion schema 2 stored is not in the emitted bytes, under any spelling.
        assert!(!generated.contains("commercial_use"));
    }

    /// The Stability AI Community License requires products using the weights to display "Powered
    /// by Stability AI". Pin it at the shipping catalog boundary so every present or future row
    /// under that licence carries the mark into the generated release manifest.
    ///
    /// The family already states [`gen_core::LicenseTerm::AttributionRequired`], and the table
    /// conformance checker rejects a row that resolves to an attribution-requiring family with no
    /// attribution. What it cannot check is that the attribution is *the right string* — this can.
    #[test]
    fn stability_ai_community_rows_require_powered_by_attribution() {
        const FAMILY: &str = "stability-ai-community";
        const REQUIRED_MARK: &str = "Powered by Stability AI";

        let components = super::component_licenses();
        let rows: Vec<_> = components
            .iter()
            .filter(|row| row.family == FAMILY)
            .collect();
        assert_eq!(
            rows.len(),
            6,
            "one root row per Stable Audio 3 registration"
        );
        for row in rows {
            assert!(
                row.attribution
                    .is_some_and(|text| text.contains(REQUIRED_MARK)),
                "component '{}' resolves to {FAMILY} without the required {REQUIRED_MARK:?} \
                 product attribution",
                row.component
            );
        }
    }

    /// The lane's preparer registry: exactly one `candle` registration whose probe accepts a
    /// Kokoro-shaped snapshot (the audio path) AND still accepts what the LLM preparer accepts
    /// (delegation) — the sc-12836 accommodation without weakening candle-llm.
    #[test]
    fn lane_preparer_probes_audio_and_delegates_llm_shapes() {
        let registry = super::snapshot_preparer_registry().unwrap();
        let regs: Vec<_> = registry.registrations().collect();
        assert_eq!(regs.len(), 1);
        assert_eq!((regs[0].backend)(), "candle");

        // A Kokoro-shaped snapshot dir is accepted by the lane probe...
        let dir = std::env::temp_dir().join("audio-catalog-kokoro-probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"istftnet": {}, "vocab": {"a": 1}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("kokoro-v1_0.pth"), b"stub").unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&dir, dir.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        // ...a MOSS-SoundEffect-shaped snapshot dir is accepted too (sc-12841)...
        let moss = std::env::temp_dir().join("audio-catalog-moss-probe");
        let _ = std::fs::remove_dir_all(&moss);
        std::fs::create_dir_all(moss.join("transformer")).unwrap();
        std::fs::write(
            moss.join("model_index.json"),
            r#"{"_class_name": "MossSoundEffectPipeline"}"#,
        )
        .unwrap();
        std::fs::write(
            moss.join("transformer/diffusion_pytorch_model.safetensors"),
            b"stub",
        )
        .unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&moss, moss.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&moss);
        // ...an ACE-Step snapshot dir is accepted too (sc-12842)...
        let ace = std::env::temp_dir().join("audio-catalog-acestep-probe");
        let _ = std::fs::remove_dir_all(&ace);
        std::fs::create_dir_all(ace.join("transformer")).unwrap();
        std::fs::write(
            ace.join("model_index.json"),
            r#"{"_class_name": "AceStepPipeline"}"#,
        )
        .unwrap();
        std::fs::write(
            ace.join("transformer/diffusion_pytorch_model.safetensors.index.json"),
            r#"{"weight_map": {"a": "diffusion_pytorch_model.safetensors"}}"#,
        )
        .unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&ace, ace.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&ace);
        // ...an OpenVoice V2 converter snapshot dir is accepted too (sc-13223)...
        let ov = std::env::temp_dir().join("audio-catalog-openvoice-probe");
        let _ = std::fs::remove_dir_all(&ov);
        std::fs::create_dir_all(&ov).unwrap();
        std::fs::write(
            ov.join("config.json"),
            r#"{"data":{"filter_length":1024},"model":{"gin_channels":256}}"#,
        )
        .unwrap();
        std::fs::write(ov.join("checkpoint.pth"), b"stub").unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&ov, ov.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&ov);
        // ...a CLAP snapshot dir is accepted too (sc-12851)...
        let clap = std::env::temp_dir().join("audio-catalog-clap-probe");
        let _ = std::fs::remove_dir_all(&clap);
        std::fs::create_dir_all(&clap).unwrap();
        std::fs::write(clap.join("config.json"), r#"{"model_type": "clap"}"#).unwrap();
        std::fs::write(clap.join("pytorch_model.bin"), b"stub").unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&clap, clap.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&clap);
        // ...a Chatterbox clone-TTS snapshot dir is accepted too (sc-13239: t3 + s3gen + tokenizer)...
        let cb = std::env::temp_dir().join("audio-catalog-chatterbox-probe");
        let _ = std::fs::remove_dir_all(&cb);
        std::fs::create_dir_all(&cb).unwrap();
        std::fs::write(cb.join("t3_cfg.safetensors"), b"stub").unwrap();
        std::fs::write(cb.join("s3gen.safetensors"), b"stub").unwrap();
        std::fs::write(cb.join("tokenizer.json"), r#"{"model":{"type":"BPE"}}"#).unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&cb, cb.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&cb);
        // ...a MOSS-TTSD dialogue snapshot dir is accepted too (sc-13518: config.json naming
        // MossTTSDForCausalLM + model.safetensors)...
        let mt = std::env::temp_dir().join("audio-catalog-moss-ttsd-probe");
        let _ = std::fs::remove_dir_all(&mt);
        std::fs::create_dir_all(&mt).unwrap();
        std::fs::write(mt.join("model.safetensors"), b"stub").unwrap();
        std::fs::write(
            mt.join("config.json"),
            r#"{"architectures": ["MossTTSDForCausalLM"]}"#,
        )
        .unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&mt, mt.join("out"));
        assert!((regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&mt);
        // ...while a bare dir (neither audio- nor LLM-shaped) is not.
        let empty = std::env::temp_dir().join("audio-catalog-empty-probe");
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&empty).unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&empty, empty.join("out"));
        assert!(!(regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }

    fn assert_sa3_snapshot_prepares(env: &str) {
        let snapshot = std::path::PathBuf::from(
            std::env::var(env)
                .unwrap_or_else(|_| panic!("set {env} to the pinned immutable snapshot")),
        );
        let registry = super::snapshot_preparer_registry().unwrap();
        let registration = registry
            .registrations()
            .find(|registration| (registration.backend)() == "candle")
            .expect("composed Candle audio preparer");
        let spec =
            super::core_llm::PrepareSpec::dense(&snapshot, snapshot.join("unused-prepared-output"));
        assert!((registration.can_prepare)(&spec), "{env}");
        let report = (registration.prepare)(&spec).expect("prepare exact SA3 snapshot");
        assert!(report.passthrough, "{env}");
        assert_eq!(report.out_dir, snapshot, "{env}");
        assert_eq!(report.num_tensors, 1_025, "{env}");
    }

    #[test]
    #[ignore = "requires the pinned Stable Audio 3 small-music snapshot"]
    fn connected_sa3_snapshot_resolves_through_the_composed_lane_preparer() {
        assert_sa3_snapshot_prepares("SA3_SMALL_MUSIC_SNAPSHOT");
    }

    #[test]
    #[ignore = "requires the pinned Stable Audio 3 small-sfx snapshot"]
    fn connected_sa3_sfx_snapshot_resolves_through_the_composed_lane_preparer() {
        assert_sa3_snapshot_prepares("SA3_SMALL_SFX_SNAPSHOT");
    }
}

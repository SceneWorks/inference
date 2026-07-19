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
    pub use candle_audio_chatterbox_ve;
    pub use candle_audio_clap;
    pub use candle_audio_kokoro;
    pub use candle_audio_moss_sfx;
    pub use candle_audio_moss_tts_realtime;
    pub use candle_audio_openvoice;
    pub use candle_audio_whisper;
}

/// Add every provider shipped by the Candle audio lane to an explicit registry builder, in
/// stable catalog order: the generators first (Kokoro TTS, MOSS SFX, ACE-Step music, MOSS-TTS-Realtime
/// streaming TTS — sc-13392), then the voice-cloning identity embedder (Chatterbox `ve`, sc-12844),
/// then the audio transforms
/// (OpenVoice V2 voice conversion, sc-13223 — the first real `AudioTransform`), then the
/// transcribers (Whisper ASR, sc-12850 — the first real `Transcriber`, the audio Captioner-analog),
/// then the audio embedders (LAION CLAP, sc-12851 — the first real `AudioEmbedder`, semantic
/// audio-text joint-space retrieval).
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    let registry = candle_audio_kokoro::register_providers(registry);
    let registry = candle_audio_moss_sfx::register_providers(registry);
    let registry = candle_audio_acestep::register_providers(registry);
    let registry = candle_audio_moss_tts_realtime::register_providers(registry);
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
// Model-weight licenses (sc-13332).
//
// A separate axis from the crate/source SPDX SBOM the release tooling already emits: each audio
// provider pins its own Hugging Face weight checkpoint, whose license (Apache-2.0 / MIT / and, for
// a model that lands later, possibly CC-BY-NC) must be surfaced so SceneWorks — a NON-COMMERCIAL
// product — can list it on its end-product licenses page. Each provider records a
// `gen_core::WeightLicense` as source of truth (traveling with the provider, beside its pinned
// HUB_REPO/HUB_REVISION); this catalog aggregates every registered provider's license in catalog
// order, and the release tooling serializes the aggregate into `release/model-weight-licenses.json`
// beside the SPDX SBOM. The `every_shipped_provider_has_a_weight_license` ship-gate below refuses
// any provider that reaches this catalog without a recorded, well-formed license.
// ---------------------------------------------------------------------------------------------

/// Every shipped audio provider's model-weight license, in catalog order — the aggregate the
/// release tooling serializes into the model-licenses manifest SceneWorks consumes (one row per
/// registered provider, keyed by its registry id).
pub fn weight_licenses() -> Vec<gen_core::WeightLicenseEntry> {
    let mut entries = Vec::new();
    entries.extend_from_slice(candle_audio_kokoro::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_moss_sfx::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_acestep::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_moss_tts_realtime::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_chatterbox_ve::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_openvoice::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_whisper::WEIGHT_LICENSES);
    entries.extend_from_slice(candle_audio_clap::WEIGHT_LICENSES);
    entries
}

/// The canonical model-licenses manifest JSON (deterministic, sorted by provider id) — the exact
/// bytes committed at `release/model-weight-licenses.json` and emitted into the release bundle by
/// `scripts/release/build_release.py`.
pub fn weight_licenses_manifest_json() -> String {
    gen_core::weight_licenses_manifest_json(&weight_licenses())
}

/// Every provider id this catalog registers, across all provider kinds (sc-13332) — the set the
/// weight-license ship-gate cross-checks so no registered provider can escape a recorded license.
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
        || candle_audio_moss_tts_realtime::prepare::can_prepare(spec)
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
    } else if candle_audio_moss_tts_realtime::prepare::can_prepare(spec) {
        candle_audio_moss_tts_realtime::prepare::prepare(spec)
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
                "moss_tts_realtime"
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

    /// The weight-license ship-gate (sc-13332): every provider this catalog registers — across
    /// EVERY kind — has a recorded, well-formed model-weight license, and no license entry is an
    /// orphan. Adding a provider to the catalog without wiring its `WEIGHT_LICENSES` slice fails
    /// here, so "no provider ships without its weight license recorded" is enforced in the
    /// composition root that decides what ships.
    #[test]
    fn every_shipped_provider_has_a_weight_license() {
        use std::collections::BTreeSet;

        let registry = super::provider_registry().unwrap();
        let registered: BTreeSet<String> = super::registered_provider_ids(&registry)
            .into_iter()
            .collect();
        assert!(!registered.is_empty(), "catalog registers no providers");

        let entries = super::weight_licenses();
        let licensed: BTreeSet<String> =
            entries.iter().map(|e| e.provider_id.to_string()).collect();

        // No duplicate license rows (one per provider).
        assert_eq!(
            entries.len(),
            licensed.len(),
            "duplicate provider id in weight_licenses()"
        );
        // Every registered provider has a license...
        for id in &registered {
            assert!(
                licensed.contains(id),
                "provider '{id}' ships without a recorded model-weight license"
            );
        }
        // ...and every license row maps to a registered provider (no stale/orphan entry).
        for id in &licensed {
            assert!(
                registered.contains(id),
                "weight-license entry '{id}' has no registered provider"
            );
        }
        // Every recorded license honors the restriction discipline (identity fields present; a
        // non-commercial license carries its restriction note).
        for entry in &entries {
            assert!(
                entry.license.is_well_formed(),
                "provider '{}' has a malformed weight license (non-commercial without a \
                 restriction note, or an empty identity field)",
                entry.provider_id
            );
            assert!(
                entry
                    .license
                    .source_url
                    .starts_with("https://huggingface.co/"),
                "provider '{}' weight-license source_url is not a Hugging Face URL",
                entry.provider_id
            );
        }
        // The eight currently-shipped audio providers, in catalog order, with their verified SPDX
        // ids — all permissive (MIT / Apache-2.0). This pins the surface so a change is deliberate.
        let ordered: Vec<(&str, &str, bool)> = super::weight_licenses()
            .iter()
            .map(|e| (e.provider_id, e.license.spdx_id, e.license.commercial_use))
            .collect();
        assert_eq!(
            ordered,
            vec![
                ("kokoro_82m", "Apache-2.0", true),
                ("moss_sfx_v2", "Apache-2.0", true),
                ("acestep_v15_turbo", "MIT", true),
                ("moss_tts_realtime", "Apache-2.0", true),
                ("chatterbox_ve", "MIT", true),
                ("openvoice_v2", "MIT", true),
                ("whisper_base", "Apache-2.0", true),
                ("clap_htsat_unfused", "Apache-2.0", true),
            ]
        );
    }

    /// The committed `release/model-weight-licenses.json` is byte-for-byte what the catalog
    /// produces (sc-13332) — the drift gate tying the release manifest the tooling emits to the
    /// per-provider source of truth. Regenerate with `UPDATE_WEIGHT_LICENSES=1 cargo test -p
    /// candle-audio-catalog weight_licenses_manifest_matches_committed_file`.
    #[test]
    fn weight_licenses_manifest_matches_committed_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../release/model-weight-licenses.json");
        let generated = super::weight_licenses_manifest_json();
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
        // ...while a bare dir (neither audio- nor LLM-shaped) is not.
        let empty = std::env::temp_dir().join("audio-catalog-empty-probe");
        let _ = std::fs::remove_dir_all(&empty);
        std::fs::create_dir_all(&empty).unwrap();
        let spec = super::core_llm::PrepareSpec::dense(&empty, empty.join("out"));
        assert!(!(regs[0].can_prepare)(&spec));
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&empty);
    }
}

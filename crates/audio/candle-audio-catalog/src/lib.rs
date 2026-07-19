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
//! The audio lane carries **generators plus the two identity-/transform-shaped audio provider
//! kinds** the epic's later slices need — [`gen_core::VoiceEmbedder`] (voice-cloning identity,
//! sc-12838) and [`gen_core::AudioTransform`] (non-prompt audio→audio, sc-12839) — validated by
//! `runtime-catalog::validate_audio`. Generators still implement the ordinary
//! [`gen_core::Generator`] contract with [`gen_core::Modality::Audio`] descriptors; the added kinds
//! ride the same explicit ProviderRegistry, surfaced in the bundle snapshot as
//! `audio_voice_embedder_ids` / `audio_transform_ids` beside `audio_generator_ids` (sc-12844) — no
//! new trait beyond the merged contracts, no linker discovery. sc-12844 ships the first of these:
//! the Chatterbox voice encoder (**chatterbox_ve**).
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
    pub use candle_audio_chatterbox_ve;
    pub use candle_audio_kokoro;
    pub use candle_audio_moss_sfx;
    pub use candle_audio_openvoice;
}

/// Add every provider shipped by the Candle audio lane to an explicit registry builder, in
/// stable catalog order: the generators first (Kokoro TTS, MOSS SFX), then the voice-cloning
/// identity embedder (Chatterbox `ve`, sc-12844), then the audio transforms (OpenVoice V2 voice
/// conversion, sc-13223 — the first real `AudioTransform`, releasing the sc-12839 gate).
pub fn register_providers(registry: ProviderRegistryBuilder) -> ProviderRegistryBuilder {
    let registry = candle_audio_kokoro::register_providers(registry);
    let registry = candle_audio_moss_sfx::register_providers(registry);
    let registry = candle_audio_chatterbox_ve::register_providers(registry);
    candle_audio_openvoice::register_providers(registry)
}

/// Build the complete explicit Candle audio provider catalog.
pub fn provider_registry() -> gen_core::Result<ProviderRegistry> {
    register_providers(ProviderRegistryBuilder::new()).build()
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
        || candle_audio_openvoice::prepare::can_prepare(spec)
        || (candle_llm::prepare::REGISTRATION.can_prepare)(spec)
}

fn lane_prepare(spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
    if candle_audio_kokoro::prepare::can_prepare(spec) {
        candle_audio_kokoro::prepare::prepare(spec)
    } else if candle_audio_moss_sfx::prepare::can_prepare(spec) {
        candle_audio_moss_sfx::prepare::prepare(spec)
    } else if candle_audio_openvoice::prepare::can_prepare(spec) {
        candle_audio_openvoice::prepare::prepare(spec)
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

        assert_eq!(generators, ["kokoro_82m", "moss_sfx_v2"]);
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
        assert_eq!(
            registry.descriptor_conformance_errors(),
            Vec::<String>::new()
        );
        // The audio lane admits generators, voice embedders, and audio transforms only — never the
        // image/text/trainer/captioner kinds (those belong in a media family).
        assert_eq!(registry.transforms().len(), 0);
        assert_eq!(registry.trainers().len(), 0);
        assert_eq!(registry.captioners().len(), 0);
        assert_eq!(registry.image_embedders().len(), 0);
        assert_eq!(registry.text_embedders().len(), 0);
        // Every audio transform is candle-backed.
        assert!(registry
            .audio_transforms()
            .all(|r| (r.descriptor)().backend == super::AUDIO_BACKEND));
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

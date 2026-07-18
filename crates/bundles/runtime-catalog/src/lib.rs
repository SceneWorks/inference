//! Validated, machine-readable composition shared by the named inference runtime bundles.
//!
//! This crate is tensor-neutral. Platform bundles supply their explicit media, LLM, and snapshot
//! preparation registries; [`RuntimeCatalog::try_new`] validates that the composition belongs to
//! the declared backend before a product can use it.

pub use core_llm;
pub use gen_core;

use core_llm::{SnapshotPreparerRegistry, TextLlmRegistry};
use gen_core::ProviderRegistry;

/// Failure to construct a supported runtime composition.
#[derive(Debug)]
pub struct RuntimeCatalogError {
    message: String,
}

impl RuntimeCatalogError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for RuntimeCatalogError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for RuntimeCatalogError {}

pub type Result<T> = std::result::Result<T, RuntimeCatalogError>;

/// The bundle's audio lane: an explicit provider registry validated against its **own** declared
/// backend, which may differ from the bundle's media backend.
///
/// Audio is the one sanctioned exception to "one tensor backend per bundle" (sc-12901,
/// `docs/architecture/audio-backend-strategy.md`): audio generation is Candle-native on every
/// platform, so the `mlx` macOS bundle carries its audio providers on `candle`. The exception is
/// scoped — the audio section carries **generators only** (the existing generator contract, no new
/// trait), and every other provider kind plus the media registry remain strictly single-backend.
struct AudioSection {
    backend: &'static str,
    registry: ProviderRegistry,
}

/// The complete, validated provider composition for one named runtime bundle.
pub struct RuntimeCatalog {
    platform: &'static str,
    backend: &'static str,
    media: ProviderRegistry,
    text: TextLlmRegistry,
    preparers: SnapshotPreparerRegistry,
    audio: Option<AudioSection>,
}

impl RuntimeCatalog {
    /// Construct and validate a platform catalog from explicit backend registries, with no audio
    /// lane declared.
    pub fn try_new(
        platform: &'static str,
        backend: &'static str,
        media: gen_core::Result<ProviderRegistry>,
        text: core_llm::Result<TextLlmRegistry>,
        preparers: core_llm::Result<SnapshotPreparerRegistry>,
    ) -> Result<Self> {
        Self::build(platform, backend, media, text, preparers, None)
    }

    /// Construct and validate a platform catalog that also declares an audio lane.
    ///
    /// `audio_backend` is the single tensor backend every provider in the audio registry must use.
    /// It may equal the bundle's media `backend` (the Candle bundles) or differ from it (the `mlx`
    /// macOS bundle carrying `candle` audio) — the sanctioned cross-backend seam described by
    /// [`Self::audio_backend`] and `docs/architecture/audio-backend-strategy.md`. The audio
    /// registry is generators-only at this release; registering any other provider kind in it
    /// fails validation.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_audio(
        platform: &'static str,
        backend: &'static str,
        media: gen_core::Result<ProviderRegistry>,
        text: core_llm::Result<TextLlmRegistry>,
        preparers: core_llm::Result<SnapshotPreparerRegistry>,
        audio_backend: &'static str,
        audio: gen_core::Result<ProviderRegistry>,
    ) -> Result<Self> {
        let audio = audio.map_err(|error| {
            RuntimeCatalogError::new(format!("{platform} audio catalog: {error}"))
        })?;
        Self::build(
            platform,
            backend,
            media,
            text,
            preparers,
            Some(AudioSection {
                backend: audio_backend,
                registry: audio,
            }),
        )
    }

    fn build(
        platform: &'static str,
        backend: &'static str,
        media: gen_core::Result<ProviderRegistry>,
        text: core_llm::Result<TextLlmRegistry>,
        preparers: core_llm::Result<SnapshotPreparerRegistry>,
        audio: Option<AudioSection>,
    ) -> Result<Self> {
        let catalog = Self {
            platform,
            backend,
            media: media.map_err(|error| {
                RuntimeCatalogError::new(format!("{platform} media catalog: {error}"))
            })?,
            text: text.map_err(|error| {
                RuntimeCatalogError::new(format!("{platform} LLM catalog: {error}"))
            })?,
            preparers: preparers.map_err(|error| {
                RuntimeCatalogError::new(format!("{platform} snapshot catalog: {error}"))
            })?,
            audio,
        };
        catalog.validate()?;
        Ok(catalog)
    }

    /// The bundle's platform label, e.g. `"macos"`, `"cuda"`, `"cpu"`.
    pub fn platform(&self) -> &'static str {
        self.platform
    }

    /// The single tensor backend every media, LLM, and snapshot-preparer provider in this
    /// catalog belongs to, e.g. `"mlx"` or `"candle"`. Enforced by `try_new`. The audio lane
    /// (when declared) carries its own single backend — see [`Self::audio_backend`].
    pub fn backend(&self) -> &'static str {
        self.backend
    }

    /// The validated media provider registry — generators, trainers, captioners, and
    /// image/text embedders. Load by id through the registry's `load_*` methods.
    pub fn media(&self) -> &ProviderRegistry {
        &self.media
    }

    /// The validated text-LLM provider registry (streaming, cancellable, multimodal).
    pub fn text(&self) -> &TextLlmRegistry {
        &self.text
    }

    /// The validated snapshot-preparer registry for this backend.
    pub fn preparers(&self) -> &SnapshotPreparerRegistry {
        &self.preparers
    }

    /// The single tensor backend of the audio lane, when this bundle declares one — e.g.
    /// `"candle"` on every platform under the sc-12901 audio backend strategy. `None` means the
    /// bundle ships no audio lane (e.g. an LLM-only composition profile).
    pub fn audio_backend(&self) -> Option<&'static str> {
        self.audio.as_ref().map(|audio| audio.backend)
    }

    /// The validated audio provider registry (generators-only), when this bundle declares an
    /// audio lane. Audio generators use the ordinary generator contract; load by id through
    /// this registry's `load` method.
    pub fn audio(&self) -> Option<&ProviderRegistry> {
        self.audio.as_ref().map(|audio| &audio.registry)
    }

    /// Return a stable, serializable inventory without loading model weights.
    pub fn snapshot(&self) -> RuntimeCatalogSnapshot {
        RuntimeCatalogSnapshot {
            platform: self.platform.to_string(),
            backend: self.backend.to_string(),
            generator_ids: self
                .media
                .generators()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            transform_ids: self
                .media
                .transforms()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            trainer_ids: self
                .media
                .trainers()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            captioner_ids: self
                .media
                .captioners()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            image_embedder_ids: self
                .media
                .image_embedders()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            text_embedder_ids: self
                .media
                .text_embedders()
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            text_llm_ids: self
                .text
                .registrations()
                .map(|registration| (registration.descriptor)().id)
                .collect(),
            snapshot_preparer_backends: self
                .preparers
                .registrations()
                .map(|registration| (registration.backend)().to_string())
                .collect(),
            audio_backend: self.audio_backend().map(str::to_string),
            audio_generator_ids: self
                .audio
                .iter()
                .flat_map(|audio| audio.registry.generators())
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
        }
    }

    fn validate(&self) -> Result<()> {
        let mut errors = Vec::new();
        if self.platform.is_empty() {
            errors.push("runtime platform is empty".to_string());
        }
        if self.backend.is_empty() {
            errors.push("runtime backend is empty".to_string());
        }
        errors.extend(self.media.descriptor_conformance_errors());

        macro_rules! validate_media_backend {
            ($registrations:expr, $kind:literal) => {
                for registration in $registrations {
                    let descriptor = (registration.descriptor)();
                    if descriptor.backend != self.backend {
                        errors.push(format!(
                            "{} '{}' uses backend '{}' in the '{}' runtime",
                            $kind, descriptor.id, descriptor.backend, self.backend
                        ));
                    }
                }
            };
        }
        validate_media_backend!(self.media.generators(), "generator");
        validate_media_backend!(self.media.transforms(), "transform");
        validate_media_backend!(self.media.trainers(), "trainer");
        validate_media_backend!(self.media.captioners(), "captioner");
        validate_media_backend!(self.media.image_embedders(), "image embedder");
        validate_media_backend!(self.media.text_embedders(), "text embedder");

        for registration in self.text.registrations() {
            let descriptor = (registration.descriptor)();
            if descriptor.id.is_empty()
                || descriptor.family.is_empty()
                || descriptor.backend.is_empty()
            {
                errors.push("LLM descriptor contains an empty identity field".to_string());
            }
            if descriptor.backend != self.backend {
                errors.push(format!(
                    "LLM '{}' uses backend '{}' in the '{}' runtime",
                    descriptor.id, descriptor.backend, self.backend
                ));
            }
        }

        if self.preparers.registrations().len() == 0 {
            errors.push("runtime has no snapshot preparer".to_string());
        }
        for registration in self.preparers.registrations() {
            let backend = (registration.backend)();
            if backend != self.backend {
                errors.push(format!(
                    "snapshot preparer '{backend}' is in the '{}' runtime",
                    self.backend
                ));
            }
        }

        if let Some(audio) = &self.audio {
            self.validate_audio(audio, &mut errors);
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(RuntimeCatalogError::new(errors.join("; ")))
        }
    }

    /// Validate the declared audio lane (sc-12901). The audio section is the one sanctioned
    /// cross-backend seam, so its rules are deliberately tighter than "any backend goes":
    ///
    /// - the audio backend is declared and non-empty;
    /// - every audio generator's descriptor belongs to the **audio** backend (single backend per
    ///   lane, exactly like the media invariant — just against the audio lane's own declaration);
    /// - the audio registry is generators-only — no other provider kind may ride in through the
    ///   audio seam;
    /// - audio generator ids do not collide with media generator ids (consumers key loads by id
    ///   across both registries);
    /// - the audio registry passes the same weights-free descriptor conformance sweep as media.
    ///
    /// Once `Modality::Audio` lands in gen-core (sc-12834), tighten this further: an audio-lane
    /// generator must advertise the audio modality, and a media-registry generator must not.
    fn validate_audio(&self, audio: &AudioSection, errors: &mut Vec<String>) {
        if audio.backend.is_empty() {
            errors.push("runtime audio backend is empty".to_string());
        }
        errors.extend(audio.registry.descriptor_conformance_errors());

        for registration in audio.registry.generators() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio generator '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
            if self
                .media
                .generators()
                .any(|media| (media.descriptor)().id == descriptor.id)
            {
                errors.push(format!(
                    "audio generator '{}' collides with a media generator id",
                    descriptor.id
                ));
            }
        }

        let non_generator_kinds = [
            ("transform", audio.registry.transforms().count()),
            ("trainer", audio.registry.trainers().count()),
            ("captioner", audio.registry.captioners().count()),
            ("image embedder", audio.registry.image_embedders().count()),
            ("text embedder", audio.registry.text_embedders().count()),
        ];
        for (kind, count) in non_generator_kinds {
            if count != 0 {
                errors.push(format!(
                    "audio registry carries {count} {kind} registration(s) — the audio lane is \
                     generators-only"
                ));
            }
        }
    }
}

/// Stable, machine-readable provider inventory for release and product compatibility checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCatalogSnapshot {
    /// The bundle's platform label (`"macos"` / `"cuda"` / `"cpu"`).
    pub platform: String,
    /// The single backend every provider belongs to (`"mlx"` / `"candle"`).
    pub backend: String,
    /// Generator ids, in stable catalog order — each a `load` key.
    pub generator_ids: Vec<String>,
    /// Standalone transform ids (empty at this release).
    pub transform_ids: Vec<String>,
    /// Trainer ids — each a `load_trainer` key.
    pub trainer_ids: Vec<String>,
    /// Captioner ids — each a `load_captioner` key.
    pub captioner_ids: Vec<String>,
    /// Image-embedder ids — each a `load_image_embedder` key.
    pub image_embedder_ids: Vec<String>,
    /// Text-embedder ids — each a `load_text_embedder` key.
    pub text_embedder_ids: Vec<String>,
    /// Text-LLM ids — each a `load_textllm` key on the text registry.
    pub text_llm_ids: Vec<String>,
    /// The backend of each registered snapshot preparer (all equal to `backend`).
    pub snapshot_preparer_backends: Vec<String>,
    /// The audio lane's single backend (`"candle"` under the sc-12901 strategy), or `None` when
    /// the bundle declares no audio lane. Additive field — absent lanes serialize as `null`.
    pub audio_backend: Option<String>,
    /// Audio generator ids, in stable catalog order — each a `load` key on the audio registry.
    /// Additive field — empty when the bundle declares no audio lane.
    pub audio_generator_ids: Vec<String>,
}

impl RuntimeCatalogSnapshot {
    /// JSON representation consumed by release tooling and external smoke projects.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "platform": self.platform,
            "backend": self.backend,
            "generator_ids": self.generator_ids,
            "transform_ids": self.transform_ids,
            "trainer_ids": self.trainer_ids,
            "captioner_ids": self.captioner_ids,
            "image_embedder_ids": self.image_embedder_ids,
            "text_embedder_ids": self.text_embedder_ids,
            "text_llm_ids": self.text_llm_ids,
            "snapshot_preparer_backends": self.snapshot_preparer_backends,
            "audio_backend": self.audio_backend,
            "audio_generator_ids": self.audio_generator_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candle_descriptor() -> core_llm::TextLlmDescriptor {
        core_llm::TextLlmDescriptor {
            id: "test-llm".to_string(),
            family: "test".to_string(),
            backend: "candle".to_string(),
            capabilities: core_llm::TextLlmCapabilities::default(),
        }
    }

    fn never_load(_spec: &core_llm::LoadSpec) -> core_llm::Result<Box<dyn core_llm::TextLlm>> {
        Err(core_llm::Error::Msg(
            "not used by catalog tests".to_string(),
        ))
    }

    fn cannot_load(_spec: &core_llm::LoadSpec) -> bool {
        false
    }

    fn mlx_backend() -> &'static str {
        "mlx"
    }

    fn cannot_prepare(_spec: &core_llm::PrepareSpec) -> bool {
        false
    }

    fn never_prepare(_spec: &core_llm::PrepareSpec) -> core_llm::Result<core_llm::PrepareReport> {
        Err(core_llm::Error::Msg(
            "not used by catalog tests".to_string(),
        ))
    }

    // ---------------------------------------------------------------------------------------
    // Audio-lane stubs (sc-12901). Real audio providers land in sc-12835/12836; these exist
    // only to prove the catalog-validation mechanics. Modality::Audio arrives with sc-12834 —
    // until then the stub reuses an existing variant, which validation does not inspect.
    // ---------------------------------------------------------------------------------------

    fn stub_audio_caps() -> gen_core::Capabilities {
        gen_core::Capabilities {
            min_size: 1,
            max_size: 4096,
            max_count: 1,
            ..Default::default()
        }
    }

    fn candle_audio_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub-audio",
            family: "test-audio",
            backend: "candle",
            modality: gen_core::Modality::Image,
            capabilities: stub_audio_caps(),
        }
    }

    fn mlx_audio_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub-audio",
            family: "test-audio",
            backend: "mlx",
            modality: gen_core::Modality::Image,
            capabilities: stub_audio_caps(),
        }
    }

    fn never_load_generator(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Generator>> {
        Err(gen_core::Error::Msg(
            "not used by catalog tests".to_string(),
        ))
    }

    fn candle_audio_registration() -> gen_core::ModelRegistration {
        gen_core::ModelRegistration {
            descriptor: candle_audio_descriptor,
            load: never_load_generator,
            footprint: None,
        }
    }

    fn mlx_preparers() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
        core_llm::SnapshotPreparerRegistryBuilder::new()
            .register(core_llm::SnapshotPreparerRegistration {
                backend: mlx_backend,
                can_prepare: cannot_prepare,
                prepare: never_prepare,
            })
            .build()
    }

    fn empty_text() -> core_llm::Result<core_llm::TextLlmRegistry> {
        core_llm::TextLlmRegistryBuilder::new().build()
    }

    /// The sanctioned cross-backend seam: an `mlx` runtime validates with a `candle` audio lane
    /// alongside its (here empty) mlx media registry, and the snapshot carries the lane.
    #[test]
    fn mlx_runtime_accepts_candle_audio_lane() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(candle_audio_registration())
            .build();

        let catalog = RuntimeCatalog::try_new_with_audio(
            "test", "mlx", media, empty_text(), mlx_preparers(), "candle", audio,
        )
        .unwrap();

        assert_eq!(catalog.backend(), "mlx");
        assert_eq!(catalog.audio_backend(), Some("candle"));
        let snapshot = catalog.snapshot();
        assert_eq!(snapshot.audio_backend.as_deref(), Some("candle"));
        assert_eq!(snapshot.audio_generator_ids, ["stub-audio"]);
        assert_eq!(snapshot.to_json()["audio_backend"], "candle");
        assert_eq!(snapshot.to_json()["audio_generator_ids"][0], "stub-audio");
    }

    /// A catalog built without an audio lane keeps the pre-audio surface: `None` backend, empty
    /// ids, and a `null` additive JSON field (serialized-compat with pre-sc-12901 consumers).
    #[test]
    fn audioless_catalog_serializes_null_audio_lane() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let catalog =
            RuntimeCatalog::try_new("test", "mlx", media, empty_text(), mlx_preparers()).unwrap();
        assert_eq!(catalog.audio_backend(), None);
        assert!(catalog.audio().is_none());
        let snapshot = catalog.snapshot();
        assert_eq!(snapshot.audio_backend, None);
        assert!(snapshot.audio_generator_ids.is_empty());
        assert!(snapshot.to_json()["audio_backend"].is_null());
    }

    /// The audio lane is itself single-backend: a generator off the declared audio backend is
    /// rejected even when it matches the bundle's media backend.
    #[test]
    fn rejects_audio_generator_off_the_audio_backend() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: mlx_audio_descriptor,
                load: never_load_generator,
                footprint: None,
            })
            .build();

        let error = RuntimeCatalog::try_new_with_audio(
            "test", "mlx", media, empty_text(), mlx_preparers(), "candle", audio,
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("audio generator 'stub-audio' uses backend 'mlx' in the 'candle' audio lane"));
    }

    /// The audio seam must not become a general cross-backend hole: only generators may ride it.
    #[test]
    fn rejects_non_generator_providers_in_audio_lane() {
        fn candle_text_embedder_descriptor() -> gen_core::TextEmbedderDescriptor {
            gen_core::TextEmbedderDescriptor {
                id: "stub-audio-text-embed",
                family: "test-audio",
                backend: "candle",
                embedding_dim: 8,
                space: "test-space",
                mac_only: false,
            }
        }
        fn never_load_text_embedder(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::TextEmbedder>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(candle_audio_registration())
            .register_text_embedder(gen_core::TextEmbedderRegistration {
                descriptor: candle_text_embedder_descriptor,
                load: never_load_text_embedder,
            })
            .build();

        let error = RuntimeCatalog::try_new_with_audio(
            "test", "mlx", media, empty_text(), mlx_preparers(), "candle", audio,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("generators-only"));
    }

    /// Consumers key loads by id across both registries, so an id shared between the media and
    /// audio registries is a composition error.
    #[test]
    fn rejects_audio_generator_id_colliding_with_media() {
        fn mlx_media_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "stub-audio", // deliberately the same id as the audio stub
                family: "test-media",
                backend: "mlx",
                modality: gen_core::Modality::Image,
                capabilities: stub_audio_caps(),
            }
        }

        let media = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: mlx_media_descriptor,
                load: never_load_generator,
                footprint: None,
            })
            .build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(candle_audio_registration())
            .build();

        let error = RuntimeCatalog::try_new_with_audio(
            "test", "mlx", media, empty_text(), mlx_preparers(), "candle", audio,
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("audio generator 'stub-audio' collides with a media generator id"));
    }

    /// An empty declared audio backend is a composition error, not a silent no-audio lane.
    #[test]
    fn rejects_empty_audio_backend() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new().build();
        let error = RuntimeCatalog::try_new_with_audio(
            "test", "mlx", media, empty_text(), mlx_preparers(), "", audio,
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("runtime audio backend is empty"));
    }

    #[test]
    fn rejects_cross_backend_llm_composition() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let text = core_llm::TextLlmRegistryBuilder::new()
            .register(core_llm::TextLlmRegistration {
                descriptor: candle_descriptor,
                load: never_load,
                can_load: cannot_load,
                weightless_vision: None,
            })
            .build();
        let preparers = core_llm::SnapshotPreparerRegistryBuilder::new()
            .register(core_llm::SnapshotPreparerRegistration {
                backend: mlx_backend,
                can_prepare: cannot_prepare,
                prepare: never_prepare,
            })
            .build();

        let error = RuntimeCatalog::try_new("test", "mlx", media, text, preparers)
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("LLM 'test-llm' uses backend 'candle' in the 'mlx' runtime"));
    }
}

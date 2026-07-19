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

/// The bundle's audio-lane declaration, as supplied by a platform bundle to
/// [`RuntimeCatalog::try_new_with_audio`] (sc-12835): the lane's single tensor backend, its
/// audio provider registry — generators plus voice embedders and audio transforms (built by the
/// audio composition root, `candle-audio-catalog`), and the snapshot-preparer registry that
/// prepares audio model
/// weights on that backend.
///
/// The preparers ride **in the lane**, not in the bundle's main preparer registry, because the
/// main registry stays strictly single-backend on the bundle's own backend: on `runtime-macos`
/// (mlx) the candle audio preparer would otherwise be unplaceable, and audio snapshots could not
/// be prepared on macOS at all (audio-backend-strategy.md, "Consequences"). On the candle bundles
/// the lane carries the same candle preparer as the main registry — redundant but uniform, so a
/// consumer prepares audio snapshots through `catalog.audio_preparers()` identically everywhere.
pub struct AudioLane {
    /// The single tensor backend every provider in the lane must use — `"candle"` on all three
    /// bundles under the sc-12901 strategy.
    pub backend: &'static str,
    /// The lane's provider registry (generators + voice embedders + audio transforms; validated).
    pub generators: gen_core::Result<ProviderRegistry>,
    /// The lane's snapshot-preparer registry (non-empty; every preparer on `backend`).
    pub preparers: core_llm::Result<SnapshotPreparerRegistry>,
}

/// The bundle's validated audio lane: an explicit provider registry + snapshot-preparer registry
/// validated against the lane's **own** declared backend, which may differ from the bundle's
/// media backend.
///
/// Audio is the one sanctioned exception to "one tensor backend per bundle" (sc-12901,
/// `docs/architecture/audio-backend-strategy.md`): audio generation is Candle-native on every
/// platform, so the `mlx` macOS bundle carries its audio providers on `candle`. The exception is
/// scoped — the audio section carries **audio-shaped providers only** (generators, voice embedders,
/// and audio transforms — the merged audio contracts, no new trait) plus the lane's own preparers,
/// and every other provider kind plus the media registry remain strictly single-backend.
struct AudioSection {
    backend: &'static str,
    registry: ProviderRegistry,
    preparers: SnapshotPreparerRegistry,
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
    /// `audio.backend` is the single tensor backend every provider in the audio registry must
    /// use. It may equal the bundle's media `backend` (the Candle bundles) or differ from it (the
    /// `mlx` macOS bundle carrying `candle` audio) — the sanctioned cross-backend seam described
    /// by [`Self::audio_backend`] and `docs/architecture/audio-backend-strategy.md`. The audio
    /// registry admits only audio-shaped kinds (generators, voice embedders, audio transforms);
    /// registering an image/text/trainer/captioner provider in it fails validation, and the lane
    /// must carry at least one snapshot preparer on the audio backend so audio model weights are
    /// preparable on every platform (sc-12835).
    pub fn try_new_with_audio(
        platform: &'static str,
        backend: &'static str,
        media: gen_core::Result<ProviderRegistry>,
        text: core_llm::Result<TextLlmRegistry>,
        preparers: core_llm::Result<SnapshotPreparerRegistry>,
        audio: AudioLane,
    ) -> Result<Self> {
        let audio_registry = audio.generators.map_err(|error| {
            RuntimeCatalogError::new(format!("{platform} audio catalog: {error}"))
        })?;
        let audio_preparers = audio.preparers.map_err(|error| {
            RuntimeCatalogError::new(format!("{platform} audio snapshot catalog: {error}"))
        })?;
        Self::build(
            platform,
            backend,
            media,
            text,
            preparers,
            Some(AudioSection {
                backend: audio.backend,
                registry: audio_registry,
                preparers: audio_preparers,
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

    /// The validated audio provider registry (generators + voice embedders + audio transforms +
    /// transcribers), when this bundle declares an audio lane. Audio generators use the ordinary
    /// generator contract; load by id through this registry's `load` / `load_voice_embedder` /
    /// `load_audio_transform` / `load_transcriber` methods.
    pub fn audio(&self) -> Option<&ProviderRegistry> {
        self.audio.as_ref().map(|audio| &audio.registry)
    }

    /// The audio lane's validated snapshot-preparer registry, when this bundle declares an audio
    /// lane — every preparer on the **audio** backend ([`Self::audio_backend`]). Prepare audio
    /// model snapshots through this registry: on `runtime-macos` the main [`Self::preparers`]
    /// registry is mlx-only and cannot prepare candle audio weights (sc-12835).
    pub fn audio_preparers(&self) -> Option<&SnapshotPreparerRegistry> {
        self.audio.as_ref().map(|audio| &audio.preparers)
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
            audio_voice_embedder_ids: self
                .audio
                .iter()
                .flat_map(|audio| audio.registry.voice_embedders())
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            audio_transform_ids: self
                .audio
                .iter()
                .flat_map(|audio| audio.registry.audio_transforms())
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            audio_transcriber_ids: self
                .audio
                .iter()
                .flat_map(|audio| audio.registry.transcribers())
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            audio_embedder_ids: self
                .audio
                .iter()
                .flat_map(|audio| audio.registry.audio_embedders())
                .map(|registration| (registration.descriptor)().id.to_string())
                .collect(),
            audio_snapshot_preparer_backends: self
                .audio
                .iter()
                .flat_map(|audio| audio.preparers.registrations())
                .map(|registration| (registration.backend)().to_string())
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
        // Audio generators never ride the media registry (sc-12835): the audio modality belongs
        // exclusively to the catalog's dedicated audio lane, whether or not this bundle declares
        // one — a media-registered audio model would dodge the lane's backend/preparer rules.
        for registration in self.media.generators() {
            let descriptor = (registration.descriptor)();
            if matches!(descriptor.modality, gen_core::Modality::Audio) {
                errors.push(format!(
                    "media generator '{}' declares Modality::Audio — audio generators belong to \
                     the audio lane, not the media registry",
                    descriptor.id
                ));
            }
        }
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

    /// Validate the declared audio lane (sc-12901, tightened by sc-12835). The audio section is
    /// the one sanctioned cross-backend seam, so its rules are deliberately tighter than "any
    /// backend goes":
    ///
    /// - the audio backend is declared and non-empty;
    /// - every audio generator's descriptor belongs to the **audio** backend (single backend per
    ///   lane, exactly like the media invariant — just against the audio lane's own declaration);
    /// - every audio generator advertises [`gen_core::Modality::Audio`] (sc-12834/sc-12835) — the
    ///   lane exists for the audio modality, not as a side door for cross-backend image/video
    ///   providers (the media registry symmetrically forbids `Modality::Audio`, see `validate`);
    /// - the audio registry admits only audio-shaped kinds — generators, voice embedders,
    ///   audio transforms, and transcribers (audio→text ASR, sc-12850), each on the audio backend;
    ///   the image/text/trainer/captioner kinds may not ride in through the audio seam;
    /// - audio generator ids do not collide with media generator ids (consumers key loads by id
    ///   across both registries);
    /// - the audio registry passes the same weights-free descriptor conformance sweep as media,
    ///   with every message prefixed `audio:` so a sweep failure names its registry;
    /// - the lane carries at least one snapshot preparer, and every one is on the **audio**
    ///   backend — audio model weights must be preparable on every platform, including the mlx
    ///   macOS bundle whose main preparer registry cannot carry the candle preparer (sc-12835).
    fn validate_audio(&self, audio: &AudioSection, errors: &mut Vec<String>) {
        if audio.backend.is_empty() {
            errors.push("runtime audio backend is empty".to_string());
        }
        errors.extend(
            audio
                .registry
                .descriptor_conformance_errors()
                .into_iter()
                .map(|error| format!("audio: {error}")),
        );

        for registration in audio.registry.generators() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio generator '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
            if !matches!(descriptor.modality, gen_core::Modality::Audio) {
                errors.push(format!(
                    "audio generator '{}' declares modality {:?} — audio-lane providers must \
                     declare Modality::Audio",
                    descriptor.id, descriptor.modality
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

        // The audio lane admits generators plus the two audio-shaped provider kinds the epic's
        // later slices need — voice embedders (voice-cloning identity, sc-12838) and audio
        // transforms (non-prompt audio→audio, sc-12839) — each on the audio backend. The
        // image/text/trainer/captioner kinds remain forbidden (they belong in a media family).
        for registration in audio.registry.voice_embedders() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio voice embedder '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
        }
        for registration in audio.registry.audio_transforms() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio transform '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
        }
        // Transcribers (audio→text, the Captioner-analog kind, sc-12850) are the fourth admitted
        // audio-shaped kind: a Transcriber consumes an AudioTrack on the candle audio backend, so it
        // rides the lane exactly like the generators, voice embedders, and audio transforms — NOT
        // the media registry where captioners (image→text on the media backend) surface.
        for registration in audio.registry.transcribers() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio transcriber '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
        }
        // Audio embedders (semantic audio-text joint-space retrieval, sc-12851) are the fifth
        // admitted audio-shaped kind: an AudioEmbedder consumes an AudioTrack on the candle audio
        // backend, so it rides the lane like the other audio kinds. It is the audio parallel of the
        // media image embedder — distinct from a voice embedder (speaker identity) — and lives on
        // the audio lane rather than the media registry because it consumes audio.
        for registration in audio.registry.audio_embedders() {
            let descriptor = (registration.descriptor)();
            if descriptor.backend != audio.backend {
                errors.push(format!(
                    "audio embedder '{}' uses backend '{}' in the '{}' audio lane",
                    descriptor.id, descriptor.backend, audio.backend
                ));
            }
        }

        let forbidden_kinds = [
            ("transform", audio.registry.transforms().count()),
            ("trainer", audio.registry.trainers().count()),
            ("captioner", audio.registry.captioners().count()),
            ("image embedder", audio.registry.image_embedders().count()),
            ("text embedder", audio.registry.text_embedders().count()),
        ];
        for (kind, count) in forbidden_kinds {
            if count != 0 {
                errors.push(format!(
                    "audio registry carries {count} {kind} registration(s) — the audio lane admits \
                     only generators, voice embedders, audio transforms, transcribers, and audio \
                     embedders"
                ));
            }
        }

        if audio.preparers.registrations().len() == 0 {
            errors.push("audio lane has no snapshot preparer".to_string());
        }
        for registration in audio.preparers.registrations() {
            let backend = (registration.backend)();
            if backend != audio.backend {
                errors.push(format!(
                    "audio snapshot preparer '{backend}' is in the '{}' audio lane",
                    audio.backend
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
    /// Audio voice-embedder ids (voice-cloning identity, sc-12844), in stable catalog order —
    /// each a `load_voice_embedder` key on the audio registry. Additive field (sc-12844) — empty
    /// when the bundle declares no audio lane or ships no voice embedder.
    pub audio_voice_embedder_ids: Vec<String>,
    /// Audio transform ids (non-prompt audio→audio: voice conversion / stem separation /
    /// super-resolution, sc-12839), in stable catalog order — each a `load_audio_transform` key on
    /// the audio registry. Additive field (sc-12844) — empty when the bundle declares no audio lane
    /// or ships no audio transform.
    pub audio_transform_ids: Vec<String>,
    /// Audio transcriber ids (audio→text ASR, the Captioner-analog kind, sc-12850), in stable
    /// catalog order — each a `load_transcriber` key on the audio registry. A Transcriber rides the
    /// audio lane (candle backend) rather than the media registry where captioners surface, so it
    /// surfaces here beside the other audio-shaped kinds. Additive field (sc-12850) — empty when the
    /// bundle declares no audio lane or ships no transcriber.
    pub audio_transcriber_ids: Vec<String>,
    /// Audio embedder ids (semantic audio-text joint-space retrieval, CLAP-class, sc-12851), in
    /// stable catalog order — each a `load_audio_embedder` key on the audio registry. An
    /// AudioEmbedder is the audio parallel of the media image embedder (distinct from a voice
    /// embedder's speaker identity); it rides the audio lane because it consumes an AudioTrack.
    /// Additive field (sc-12851) — empty when the bundle declares no audio lane or ships no audio
    /// embedder.
    pub audio_embedder_ids: Vec<String>,
    /// The backend of each snapshot preparer carried **in the audio lane** (all equal to
    /// `audio_backend` — `"candle"` on every platform under the sc-12901 strategy). Additive
    /// field (sc-12835) — empty when the bundle declares no audio lane.
    pub audio_snapshot_preparer_backends: Vec<String>,
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
            "audio_voice_embedder_ids": self.audio_voice_embedder_ids,
            "audio_transform_ids": self.audio_transform_ids,
            "audio_transcriber_ids": self.audio_transcriber_ids,
            "audio_embedder_ids": self.audio_embedder_ids,
            "audio_snapshot_preparer_backends": self.audio_snapshot_preparer_backends,
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

    fn candle_backend() -> &'static str {
        "candle"
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
    // Audio-lane stubs (sc-12901/sc-12835). Real audio providers land in sc-12836+; these
    // exist only to prove the catalog-validation mechanics. The stubs declare
    // Modality::Audio (sc-12834) — validation now requires it inside the audio lane.
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
            modality: gen_core::Modality::Audio,
            capabilities: stub_audio_caps(),
        }
    }

    fn mlx_audio_descriptor() -> gen_core::ModelDescriptor {
        gen_core::ModelDescriptor {
            id: "stub-audio",
            family: "test-audio",
            backend: "mlx",
            modality: gen_core::Modality::Audio,
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

    /// The audio lane's preparer stub — a candle preparer carried in the lane (sc-12835), the
    /// shape `candle_llm::snapshot_preparer_registry()` supplies on the real bundles.
    fn candle_audio_preparers() -> core_llm::Result<core_llm::SnapshotPreparerRegistry> {
        core_llm::SnapshotPreparerRegistryBuilder::new()
            .register(core_llm::SnapshotPreparerRegistration {
                backend: candle_backend,
                can_prepare: cannot_prepare,
                prepare: never_prepare,
            })
            .build()
    }

    /// A well-formed `candle` audio lane around the given generator registry.
    fn candle_audio_lane(generators: gen_core::Result<gen_core::ProviderRegistry>) -> AudioLane {
        AudioLane {
            backend: "candle",
            generators,
            preparers: candle_audio_preparers(),
        }
    }

    fn empty_text() -> core_llm::Result<core_llm::TextLlmRegistry> {
        core_llm::TextLlmRegistryBuilder::new().build()
    }

    /// The sanctioned cross-backend seam: an `mlx` runtime validates with a `candle` audio lane
    /// alongside its (here empty) mlx media registry, and the snapshot carries the lane — its
    /// backend, generator ids, and the lane's own candle preparer (sc-12835).
    #[test]
    fn mlx_runtime_accepts_candle_audio_lane() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(candle_audio_registration())
            .build();

        let catalog = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .unwrap();

        assert_eq!(catalog.backend(), "mlx");
        assert_eq!(catalog.audio_backend(), Some("candle"));
        assert!(catalog.audio_preparers().is_some());
        let snapshot = catalog.snapshot();
        assert_eq!(snapshot.audio_backend.as_deref(), Some("candle"));
        assert_eq!(snapshot.audio_generator_ids, ["stub-audio"]);
        assert_eq!(snapshot.audio_snapshot_preparer_backends, ["candle"]);
        assert_eq!(snapshot.snapshot_preparer_backends, ["mlx"]);
        assert_eq!(snapshot.to_json()["audio_backend"], "candle");
        assert_eq!(snapshot.to_json()["audio_generator_ids"][0], "stub-audio");
        assert_eq!(
            snapshot.to_json()["audio_snapshot_preparer_backends"][0],
            "candle"
        );
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
        assert!(catalog.audio_preparers().is_none());
        let snapshot = catalog.snapshot();
        assert_eq!(snapshot.audio_backend, None);
        assert!(snapshot.audio_generator_ids.is_empty());
        assert!(snapshot.audio_voice_embedder_ids.is_empty());
        assert!(snapshot.audio_transform_ids.is_empty());
        assert!(snapshot.audio_transcriber_ids.is_empty());
        assert!(snapshot.audio_embedder_ids.is_empty());
        assert!(snapshot.audio_snapshot_preparer_backends.is_empty());
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
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains(
            "audio generator 'stub-audio' uses backend 'mlx' in the 'candle' audio lane"
        ));
    }

    /// The audio seam must not become a general cross-backend hole: only the audio-shaped kinds
    /// (generators, voice embedders, audio transforms) may ride it — a text embedder is rejected.
    #[test]
    fn rejects_non_audio_providers_in_audio_lane() {
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
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains(
            "audio registry carries 1 text embedder registration(s) — the audio lane admits only \
             generators, voice embedders, audio transforms, transcribers, and audio embedders"
        ));
    }

    fn candle_transcriber_descriptor() -> gen_core::TranscriberDescriptor {
        gen_core::TranscriberDescriptor {
            id: "stub-asr",
            family: "asr",
            backend: "candle",
            capabilities: gen_core::TranscribeCapabilities {
                supports_segment_timestamps: true,
                max_audio_seconds: 30.0,
                max_new_tokens: 448,
                ..Default::default()
            },
        }
    }
    fn never_load_transcriber(
        _spec: &gen_core::LoadSpec,
    ) -> gen_core::Result<Box<dyn gen_core::Transcriber>> {
        Err(gen_core::Error::Msg(
            "not used by catalog tests".to_string(),
        ))
    }

    /// A voice embedder (sc-12844), an audio transform (sc-12839), and a transcriber (sc-12850) are
    /// admitted to the audio lane and surfaced in the snapshot beside the generators — the audio
    /// Captioner-analog lane-surfacing decision.
    #[test]
    fn admits_and_surfaces_voice_embedder_and_audio_transform() {
        fn candle_voice_embedder_descriptor() -> gen_core::VoiceEmbedderDescriptor {
            gen_core::VoiceEmbedderDescriptor {
                id: "stub-voice-embed",
                family: "voice",
                backend: "candle",
                embedding_dim: 8,
                mac_only: false,
            }
        }
        fn never_load_voice_embedder(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::VoiceEmbedder>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }
        fn candle_audio_transform_descriptor() -> gen_core::AudioTransformDescriptor {
            gen_core::AudioTransformDescriptor {
                id: "stub-voice-convert",
                family: "audio",
                backend: "candle",
                capabilities: gen_core::AudioTransformCapabilities {
                    kind: gen_core::AudioTransformKind::VoiceConversion,
                    ..Default::default()
                },
            }
        }
        fn never_load_audio_transform(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::AudioTransform>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }
        fn candle_audio_embedder_descriptor() -> gen_core::AudioEmbedderDescriptor {
            gen_core::AudioEmbedderDescriptor {
                id: "stub-audio-embed",
                family: "audio-embed",
                backend: "candle",
                embedding_dim: 16,
                space: "stub-space",
                mac_only: false,
            }
        }
        fn never_load_audio_embedder(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::AudioEmbedder>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(candle_audio_registration())
            .register_voice_embedder(gen_core::VoiceEmbedderRegistration {
                descriptor: candle_voice_embedder_descriptor,
                load: never_load_voice_embedder,
            })
            .register_audio_transform(gen_core::AudioTransformRegistration {
                descriptor: candle_audio_transform_descriptor,
                load: never_load_audio_transform,
            })
            .register_transcriber(gen_core::TranscriberRegistration {
                descriptor: candle_transcriber_descriptor,
                load: never_load_transcriber,
            })
            .register_audio_embedder(gen_core::AudioEmbedderRegistration {
                descriptor: candle_audio_embedder_descriptor,
                load: never_load_audio_embedder,
            })
            .build();

        let catalog = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .unwrap();
        let snapshot = catalog.snapshot();
        assert_eq!(snapshot.audio_generator_ids, ["stub-audio"]);
        assert_eq!(snapshot.audio_voice_embedder_ids, ["stub-voice-embed"]);
        assert_eq!(snapshot.audio_transform_ids, ["stub-voice-convert"]);
        assert_eq!(snapshot.audio_transcriber_ids, ["stub-asr"]);
        assert_eq!(snapshot.audio_embedder_ids, ["stub-audio-embed"]);
        assert_eq!(
            snapshot.to_json()["audio_voice_embedder_ids"][0],
            "stub-voice-embed"
        );
        assert_eq!(
            snapshot.to_json()["audio_transform_ids"][0],
            "stub-voice-convert"
        );
        assert_eq!(snapshot.to_json()["audio_transcriber_ids"][0], "stub-asr");
        assert_eq!(
            snapshot.to_json()["audio_embedder_ids"][0],
            "stub-audio-embed"
        );
    }

    /// A transcriber off the declared audio backend is rejected (single-backend lane), exactly like
    /// the voice-embedder and audio-transform kinds.
    #[test]
    fn rejects_transcriber_off_the_audio_backend() {
        fn mlx_transcriber_descriptor() -> gen_core::TranscriberDescriptor {
            gen_core::TranscriberDescriptor {
                id: "stub-asr",
                family: "asr",
                backend: "mlx",
                capabilities: gen_core::TranscribeCapabilities {
                    supports_segment_timestamps: true,
                    max_audio_seconds: 30.0,
                    max_new_tokens: 448,
                    ..Default::default()
                },
            }
        }
        fn never_load_transcriber(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::Transcriber>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_transcriber(gen_core::TranscriberRegistration {
                descriptor: mlx_transcriber_descriptor,
                load: never_load_transcriber,
            })
            .build();

        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains(
            "audio transcriber 'stub-asr' uses backend 'mlx' in the 'candle' audio lane"
        ));
    }

    /// A voice embedder off the declared audio backend is rejected (single-backend lane).
    #[test]
    fn rejects_voice_embedder_off_the_audio_backend() {
        fn mlx_voice_embedder_descriptor() -> gen_core::VoiceEmbedderDescriptor {
            gen_core::VoiceEmbedderDescriptor {
                id: "stub-voice-embed",
                family: "voice",
                backend: "mlx",
                embedding_dim: 8,
                mac_only: false,
            }
        }
        fn never_load_voice_embedder(
            _spec: &gen_core::LoadSpec,
        ) -> gen_core::Result<Box<dyn gen_core::VoiceEmbedder>> {
            Err(gen_core::Error::Msg(
                "not used by catalog tests".to_string(),
            ))
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_voice_embedder(gen_core::VoiceEmbedderRegistration {
                descriptor: mlx_voice_embedder_descriptor,
                load: never_load_voice_embedder,
            })
            .build();

        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains(
            "audio voice embedder 'stub-voice-embed' uses backend 'mlx' in the 'candle' audio lane"
        ));
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
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
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
        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            AudioLane {
                backend: "",
                generators: gen_core::ProviderRegistryBuilder::new().build(),
                preparers: candle_audio_preparers(),
            },
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("runtime audio backend is empty"));
    }

    /// The audio lane requires `Modality::Audio` (sc-12834/sc-12835): a cross-backend image
    /// provider cannot ride the sanctioned audio seam.
    #[test]
    fn rejects_non_audio_modality_in_audio_lane() {
        fn candle_image_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "stub-audio",
                family: "test-audio",
                backend: "candle",
                modality: gen_core::Modality::Image,
                capabilities: stub_audio_caps(),
            }
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: candle_image_descriptor,
                load: never_load_generator,
                footprint: None,
            })
            .build();
        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("audio-lane providers must declare Modality::Audio"));
    }

    /// The symmetric rule: `Modality::Audio` is forbidden in the media registry — audio
    /// providers must ride the audio lane, never the media section (sc-12835). Enforced with or
    /// without a declared audio lane.
    #[test]
    fn rejects_audio_modality_in_media_registry() {
        fn mlx_media_audio_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "smuggled-audio",
                family: "test-audio",
                backend: "mlx",
                modality: gen_core::Modality::Audio,
                capabilities: stub_audio_caps(),
            }
        }

        let media = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: mlx_media_audio_descriptor,
                load: never_load_generator,
                footprint: None,
            })
            .build();
        let error = RuntimeCatalog::try_new("test", "mlx", media, empty_text(), mlx_preparers())
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("media generator 'smuggled-audio' declares Modality::Audio"));
    }

    /// The audio conformance sweep names its registry: a malformed audio descriptor's message
    /// carries the `audio:` prefix so it cannot be mistaken for a media-registry failure.
    #[test]
    fn audio_conformance_errors_carry_the_audio_prefix() {
        fn malformed_audio_descriptor() -> gen_core::ModelDescriptor {
            gen_core::ModelDescriptor {
                id: "stub-audio",
                family: "test-audio",
                backend: "candle",
                modality: gen_core::Modality::Audio,
                // max_count 0 + Default size bounds — two conformance violations.
                capabilities: gen_core::Capabilities::default(),
            }
        }

        let media = gen_core::ProviderRegistryBuilder::new().build();
        let audio = gen_core::ProviderRegistryBuilder::new()
            .register_generator(gen_core::ModelRegistration {
                descriptor: malformed_audio_descriptor,
                load: never_load_generator,
                footprint: None,
            })
            .build();
        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            candle_audio_lane(audio),
        )
        .err()
        .unwrap();
        assert!(
            error.to_string().contains("audio: generator 'stub-audio'"),
            "sweep message must carry the audio: prefix, got: {error}"
        );
    }

    /// The audio lane must be able to prepare its own snapshots: a lane with no preparer is a
    /// composition error (on macOS the main preparer registry is mlx-only and cannot cover it).
    #[test]
    fn rejects_audio_lane_without_preparer() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            AudioLane {
                backend: "candle",
                generators: gen_core::ProviderRegistryBuilder::new().build(),
                preparers: core_llm::SnapshotPreparerRegistryBuilder::new().build(),
            },
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("audio lane has no snapshot preparer"));
    }

    /// The lane's preparers are single-backend against the **audio** backend — an mlx preparer
    /// cannot ride the candle audio lane even inside an mlx bundle.
    #[test]
    fn rejects_audio_preparer_off_the_audio_backend() {
        let media = gen_core::ProviderRegistryBuilder::new().build();
        let error = RuntimeCatalog::try_new_with_audio(
            "test",
            "mlx",
            media,
            empty_text(),
            mlx_preparers(),
            AudioLane {
                backend: "candle",
                generators: gen_core::ProviderRegistryBuilder::new().build(),
                preparers: mlx_preparers(),
            },
        )
        .err()
        .unwrap();
        assert!(error
            .to_string()
            .contains("audio snapshot preparer 'mlx' is in the 'candle' audio lane"));
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

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

/// The complete, validated provider composition for one named runtime bundle.
pub struct RuntimeCatalog {
    platform: &'static str,
    backend: &'static str,
    media: ProviderRegistry,
    text: TextLlmRegistry,
    preparers: SnapshotPreparerRegistry,
}

impl RuntimeCatalog {
    /// Construct and validate a platform catalog from explicit backend registries.
    pub fn try_new(
        platform: &'static str,
        backend: &'static str,
        media: gen_core::Result<ProviderRegistry>,
        text: core_llm::Result<TextLlmRegistry>,
        preparers: core_llm::Result<SnapshotPreparerRegistry>,
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
        };
        catalog.validate()?;
        Ok(catalog)
    }

    pub fn platform(&self) -> &'static str {
        self.platform
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn media(&self) -> &ProviderRegistry {
        &self.media
    }

    pub fn text(&self) -> &TextLlmRegistry {
        &self.text
    }

    pub fn preparers(&self) -> &SnapshotPreparerRegistry {
        &self.preparers
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

        if errors.is_empty() {
            Ok(())
        } else {
            Err(RuntimeCatalogError::new(errors.join("; ")))
        }
    }
}

/// Stable, machine-readable provider inventory for release and product compatibility checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RuntimeCatalogSnapshot {
    pub platform: String,
    pub backend: String,
    pub generator_ids: Vec<String>,
    pub transform_ids: Vec<String>,
    pub trainer_ids: Vec<String>,
    pub captioner_ids: Vec<String>,
    pub image_embedder_ids: Vec<String>,
    pub text_embedder_ids: Vec<String>,
    pub text_llm_ids: Vec<String>,
    pub snapshot_preparer_backends: Vec<String>,
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

//! Versioned, hardware-free capability export schema (`sga.capabilities/2`), canonical
//! deterministic serializer, semantic equality checker, and runtime-vs-export disagreement
//! refusal (`SW-6980`: `REG-3`, `HOOK-4`, `M8`, `AC1`).
//!
//! This module is deliberately **tensor-free** and **host-metadata-free**:
//! - [`CapabilitiesExportV2`] contains pure semantic capability truth derived from linked
//!   primary [`RouteRecord`] and registry declarations — **zero** git SHA, pin hash,
//!   timestamp, or host filesystem path.
//! - [`CapabilitiesExportV2::to_canonical_json_bytes`] produces deterministic UTF-8 JSON with
//!   recursively sorted keys so identical semantic contents across a pin change emit identical
//!   bytes and require no artifact refresh (`M8`).
//! - [`verify_runtime_against_export`] enforces `REG-3`: the worker's linked engine remains
//!   authoritative at runtime and refuses with a typed [`ExportDisagreementError`] when its
//!   linked registry disagrees with an exported capability document.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::approximation::{
    ApproximationSurface, CharacterizationBinding, FeatureCacheDomain, TokenPruningDomain,
};
use crate::audio_embed::AudioEmbedderDescriptor;
use crate::audio_transform::{AudioTransformDescriptor, AudioTransformKind};
use crate::caption::CaptionerDescriptor;
use crate::control::AcceptedControlKinds;
use crate::core_llm::{self, ConstraintKind, TextLlmDescriptor};
use crate::execution_domains::{
    CfgBatching, CfgBatchingDomain, ExecutionSurface, ExecutionValueDomain, PrefixKvCacheDomain,
    PrefixKvCacheMode,
};
use crate::feature::{
    FeatureApplicability, FeatureMemoryEffect, Pass0ConditioningRequirement, TransitionResidency,
};
use crate::generator::{
    AudioEditMode, Capabilities, ComponentPrecisionFloor, ConditioningKind, ControlKind, Modality,
    ModelDescriptor, SizeFloor, StagedResidencyAvailability, StepSupport,
};
use crate::image_embed::ImageEmbedderDescriptor;
use crate::latent::{
    latent_spaces_compatible, LatentNormalization, LatentPatchLayout, LatentSpace,
    LatentTemporalLaw, DECODER_OPTIONS,
};
use crate::memory_strategy::MemoryStrategy;
use crate::pag::PagCapability;
use crate::registry::{
    CheckpointAdapterRegistration, CheckpointBackend, ImportedModelOperation,
    ImportedModelRegistration, ImportedModelSource, ProviderRegistry,
};
use crate::route::{
    AdapterInstallPolicy, ModelBinding, PreviewSupport, ProductLimits, RenderPlanScope,
    ResidencyPhaseKind, ResidencyPhaseNode, RouteDefaults, RouteRecord, SeedAndBatchSemantics,
    SeedScheme, WeightSourceKind,
};
use crate::runtime::Quant;
use crate::text_embed::TextEmbedderDescriptor;
use crate::train::TrainerDescriptor;
use crate::transcribe::TranscriberDescriptor;
use crate::transform::TransformDescriptor;
use crate::voice_embed::VoiceEmbedderDescriptor;
use crate::{Error, Result};

/// Canonical schema version string for `sga.capabilities/2` build products (`REG-3`).
pub const CAPABILITIES_EXPORT_SCHEMA_VERSION: &str = "sga.capabilities/2";

/// Shipping feature-set identifiers covered by hosted descriptor-only capability exports (`AC1`).
pub const FEATURE_SET_MACOS_MLX: &str = "macos-mlx";
pub const FEATURE_SET_IOS_MEDIA: &str = "ios-media";
pub const FEATURE_SET_IOS_LLM_ONLY: &str = "ios-llm-only";
pub const FEATURE_SET_CANDLE_CUDA: &str = "candle-cuda";
pub const FEATURE_SET_CANDLE_CPU: &str = "candle-cpu";

/// Top-level versioned, hardware-free capability export (`sga.capabilities/2`).
///
/// Every field represents pure semantic registry capability truth. No git commit SHA, pin hash,
/// wall-clock timestamp, or machine/host path is permitted anywhere in this document (`M8`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CapabilitiesExportV2 {
    /// Always [`CAPABILITIES_EXPORT_SCHEMA_VERSION`] (`"sga.capabilities/2"`).
    pub schema_version: String,
    /// Primary tensor backend (`"mlx"` or `"candle"`).
    pub backend: String,
    /// Target platform label (`"macos"`, `"ios"`, `"cuda"`, `"cpu"`).
    pub platform: String,
    /// Shipping feature set label (e.g. `"macos-mlx"`, `"ios-media"`, `"ios-llm-only"`,
    /// `"candle-cuda"`, `"candle-cpu"`).
    pub feature_set: String,
    /// Audio lane tensor backend when this feature set ships an audio lane (`Some("candle")`),
    /// or `None` when no audio lane is composed (e.g. iOS or LLM-only profiles).
    pub audio_backend: Option<String>,
    /// Advanced quantization tiers surfaced beyond standard `q4`/`q8` (e.g. `["nvfp4"]` on
    /// `candle-cuda`, empty on `candle-cpu` / `macos-mlx` / `ios-*`).
    pub advanced_quant_tiers: Vec<String>,
    /// Explicitly rejected quantization tiers on this platform (`tier -> refusal reason`).
    pub rejected_quants: BTreeMap<String, String>,
    /// Registered main-lane snapshot preparer backends.
    pub snapshot_preparer_backends: Vec<String>,
    /// Registered audio-lane snapshot preparer backends.
    pub audio_snapshot_preparer_backends: Vec<String>,

    /// Media generation routes keyed by `route_id`.
    pub routes: BTreeMap<String, ExportedRouteV2>,
    /// Audio generation routes keyed by `route_id`.
    pub audio_routes: BTreeMap<String, ExportedRouteV2>,
    /// Conditioned / composed feature sub-routes keyed by `route_id`.
    pub feature_routes: BTreeMap<String, ExportedFeatureRouteV2>,
    /// Text-LLM and vector (StarVector) entrypoints keyed by provider `id`.
    pub llm_models: BTreeMap<String, ExportedLlmEntry>,
    /// Media trainers keyed by trainer `id`.
    pub trainers: BTreeMap<String, ExportedTrainerEntry>,
    /// Media captioners (image → text) keyed by captioner `id`.
    pub captioners: BTreeMap<String, ExportedCaptionerEntry>,
    /// Audio transcribers (audio → text) keyed by transcriber `id`.
    pub transcribers: BTreeMap<String, ExportedTranscriberEntry>,
    /// Embedders separated by modality (image, text, voice, audio).
    pub embedders: ExportedEmbeddersV2,
    /// Non-prompt transforms separated by modality (image, audio).
    pub transforms: ExportedTransformsV2,
    /// Imported model routes derived from checkpoint adapters.
    pub imported_models: Vec<ExportedImportedModelEntry>,
    /// Portable checkpoint adapters registered on this backend keyed by `adapter_id`.
    pub checkpoint_adapters: BTreeMap<String, ExportedCheckpointAdapterEntry>,
    /// Platform-owned bespoke utility crates and descriptor-less memory route waivers.
    pub bespoke_waivers: ExportedBespokeSurfaceV2,
}

impl CapabilitiesExportV2 {
    /// Convert to a canonical `serde_json::Value` with recursively sorted object keys.
    pub fn to_canonical_json_value(&self) -> Result<serde_json::Value> {
        let raw = serde_json::to_value(self)
            .map_err(|err| Error::Msg(format!("failed to serialize sga.capabilities/2: {err}")))?;
        Ok(canonicalize_json_value(raw))
    }

    /// Serialize to deterministic canonical UTF-8 JSON bytes (pretty-printed, recursively sorted
    /// object keys, trailing newline).
    pub fn to_canonical_json_bytes(&self) -> Result<Vec<u8>> {
        let canonical = self.to_canonical_json_value()?;
        let mut bytes = serde_json::to_vec_pretty(&canonical).map_err(|err| {
            Error::Msg(format!("failed to encode sga.capabilities/2 JSON: {err}"))
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Parse and validate a `sga.capabilities/2` JSON document.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let parsed: Self = serde_json::from_slice(bytes)
            .map_err(|err| Error::Msg(format!("failed to parse sga.capabilities/2 JSON: {err}")))?;
        if parsed.schema_version != CAPABILITIES_EXPORT_SCHEMA_VERSION {
            return Err(Error::Msg(format!(
                "unsupported capability export schema_version {:?}; expected {:?}",
                parsed.schema_version, CAPABILITIES_EXPORT_SCHEMA_VERSION
            )));
        }
        Ok(parsed)
    }

    /// Pure SHA-256 hex digest of the canonical semantic JSON bytes.
    ///
    /// Computed on demand and deliberately never embedded inside the JSON payload (`M8`).
    pub fn semantic_digest_hex(&self) -> Result<String> {
        let bytes = self.to_canonical_json_bytes()?;
        let digest = Sha256::digest(&bytes);
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            use std::fmt::Write as _;
            let _ = write!(&mut hex, "{byte:02x}");
        }
        Ok(hex)
    }
}

/// Recursively canonicalize any `serde_json::Value` so every object's keys are in deterministic
/// lexicographical order.
pub fn canonicalize_json_value(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let sorted: BTreeMap<String, serde_json::Value> = map
                .into_iter()
                .map(|(k, v)| (k, canonicalize_json_value(v)))
                .collect();
            let mut out = serde_json::Map::with_capacity(sorted.len());
            for (k, v) in sorted {
                out.insert(k, v);
            }
            serde_json::Value::Object(out)
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(canonicalize_json_value).collect())
        }
        other => other,
    }
}

/// Pure semantic equality check between two `sga.capabilities/2` exports (`M8`, `AC1`).
///
/// Equal semantic contents across a pin change require no artifact refresh or measured rerun.
pub fn semantic_equal(a: &CapabilitiesExportV2, b: &CapabilitiesExportV2) -> bool {
    match (a.to_canonical_json_bytes(), b.to_canonical_json_bytes()) {
        (Ok(left), Ok(right)) => left == right,
        _ => a == b,
    }
}

/// Parse two `sga.capabilities/2` JSON documents and compare their canonical semantic bytes (`M8`).
pub fn canonical_bytes_equal(a: &[u8], b: &[u8]) -> Result<bool> {
    let left = CapabilitiesExportV2::from_json_bytes(a)?.to_canonical_json_bytes()?;
    let right = CapabilitiesExportV2::from_json_bytes(b)?.to_canonical_json_bytes()?;
    Ok(left == right)
}

// -------------------------------------------------------------------------------------------------
// Runtime-vs-Export Disagreement Refusal (`REG-3`)
// -------------------------------------------------------------------------------------------------

/// Typed classification of a disagreement between a worker's live linked registry and an exported
/// `sga.capabilities/2` document (`REG-3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportDisagreementKind {
    SchemaVersionMismatch,
    PlatformMismatch,
    BackendMismatch,
    FeatureSetMismatch,
    AudioBackendMismatch,
    AdvancedQuantTiersMismatch,
    RejectedQuantsMismatch,
    SnapshotPreparersMismatch,
    MissingRouteInRuntime,
    ExtraRouteInRuntime,
    RouteFeatureApplicabilityMismatch,
    RouteDefinitionMismatch,
    MissingAudioRouteInRuntime,
    ExtraAudioRouteInRuntime,
    AudioRouteDefinitionMismatch,
    FeatureRoutesMismatch,
    LlmModelsMismatch,
    TrainersMismatch,
    CaptionersMismatch,
    TranscribersMismatch,
    EmbeddersMismatch,
    TransformsMismatch,
    ImportedModelsMismatch,
    CheckpointAdaptersMismatch,
    BespokeWaiversMismatch,
}

/// One concrete typed disagreement between the authoritative runtime catalog and a build export.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportDisagreement {
    pub kind: ExportDisagreementKind,
    pub subject: String,
    pub detail: String,
}

/// Typed refusal returned when a worker's linked engine disagrees with an exported capability
/// document (`REG-3`).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "linked runtime disagrees with capability export ({} disagreement(s)): {}",
    self.disagreements.len(),
    self.summary()
)]
pub struct ExportDisagreementError {
    pub disagreements: Vec<ExportDisagreement>,
}

impl ExportDisagreementError {
    pub fn summary(&self) -> String {
        self.disagreements
            .iter()
            .take(5)
            .map(|d| format!("{:?}[{}]: {}", d.kind, d.subject, d.detail))
            .collect::<Vec<_>>()
            .join("; ")
    }
}

/// Verify that a live linked runtime's export matches an expected `sga.capabilities/2` document.
///
/// Per `REG-3`, the worker's linked engine is authoritative at runtime; on any disagreement this
/// returns a typed [`ExportDisagreementError`] rather than silently proceeding.
pub fn verify_runtime_against_export(
    runtime: &CapabilitiesExportV2,
    expected: &CapabilitiesExportV2,
) -> std::result::Result<(), ExportDisagreementError> {
    let mut disagreements = Vec::new();

    if runtime.schema_version != expected.schema_version {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::SchemaVersionMismatch,
            subject: "schema_version".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.schema_version, expected.schema_version
            ),
        });
    }
    if runtime.platform != expected.platform {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::PlatformMismatch,
            subject: "platform".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.platform, expected.platform
            ),
        });
    }
    if runtime.backend != expected.backend {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::BackendMismatch,
            subject: "backend".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.backend, expected.backend
            ),
        });
    }
    if runtime.feature_set != expected.feature_set {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::FeatureSetMismatch,
            subject: "feature_set".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.feature_set, expected.feature_set
            ),
        });
    }
    if runtime.audio_backend != expected.audio_backend {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::AudioBackendMismatch,
            subject: "audio_backend".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.audio_backend, expected.audio_backend
            ),
        });
    }
    if runtime.advanced_quant_tiers != expected.advanced_quant_tiers {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::AdvancedQuantTiersMismatch,
            subject: "advanced_quant_tiers".into(),
            detail: format!(
                "runtime {:?} != export {:?}",
                runtime.advanced_quant_tiers, expected.advanced_quant_tiers
            ),
        });
    }
    if runtime.rejected_quants != expected.rejected_quants {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::RejectedQuantsMismatch,
            subject: "rejected_quants".into(),
            detail: "rejected quantization surface differs".into(),
        });
    }
    if runtime.snapshot_preparer_backends != expected.snapshot_preparer_backends
        || runtime.audio_snapshot_preparer_backends != expected.audio_snapshot_preparer_backends
    {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::SnapshotPreparersMismatch,
            subject: "snapshot_preparer_backends".into(),
            detail: format!(
                "runtime main={:?} audio={:?} != export main={:?} audio={:?}",
                runtime.snapshot_preparer_backends,
                runtime.audio_snapshot_preparer_backends,
                expected.snapshot_preparer_backends,
                expected.audio_snapshot_preparer_backends
            ),
        });
    }

    diff_route_maps(
        &runtime.routes,
        &expected.routes,
        ExportDisagreementKind::MissingRouteInRuntime,
        ExportDisagreementKind::ExtraRouteInRuntime,
        ExportDisagreementKind::RouteFeatureApplicabilityMismatch,
        ExportDisagreementKind::RouteDefinitionMismatch,
        &mut disagreements,
    );
    diff_route_maps(
        &runtime.audio_routes,
        &expected.audio_routes,
        ExportDisagreementKind::MissingAudioRouteInRuntime,
        ExportDisagreementKind::ExtraAudioRouteInRuntime,
        ExportDisagreementKind::RouteFeatureApplicabilityMismatch,
        ExportDisagreementKind::AudioRouteDefinitionMismatch,
        &mut disagreements,
    );

    if runtime.feature_routes != expected.feature_routes {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::FeatureRoutesMismatch,
            subject: "feature_routes".into(),
            detail: format!(
                "runtime has {} feature route(s), export has {}",
                runtime.feature_routes.len(),
                expected.feature_routes.len()
            ),
        });
    }
    if runtime.llm_models != expected.llm_models {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::LlmModelsMismatch,
            subject: "llm_models".into(),
            detail: format!(
                "runtime ids {:?} != export ids {:?}",
                runtime.llm_models.keys().collect::<Vec<_>>(),
                expected.llm_models.keys().collect::<Vec<_>>()
            ),
        });
    }
    if runtime.trainers != expected.trainers {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::TrainersMismatch,
            subject: "trainers".into(),
            detail: format!(
                "runtime ids {:?} != export ids {:?}",
                runtime.trainers.keys().collect::<Vec<_>>(),
                expected.trainers.keys().collect::<Vec<_>>()
            ),
        });
    }
    if runtime.captioners != expected.captioners {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::CaptionersMismatch,
            subject: "captioners".into(),
            detail: "captioner descriptors differ".into(),
        });
    }
    if runtime.transcribers != expected.transcribers {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::TranscribersMismatch,
            subject: "transcribers".into(),
            detail: "transcriber descriptors differ".into(),
        });
    }
    if runtime.embedders != expected.embedders {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::EmbeddersMismatch,
            subject: "embedders".into(),
            detail: "embedder descriptors differ".into(),
        });
    }
    if runtime.transforms != expected.transforms {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::TransformsMismatch,
            subject: "transforms".into(),
            detail: "transform descriptors differ".into(),
        });
    }
    if runtime.imported_models != expected.imported_models {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::ImportedModelsMismatch,
            subject: "imported_models".into(),
            detail: "imported model routes differ".into(),
        });
    }
    if runtime.checkpoint_adapters != expected.checkpoint_adapters {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::CheckpointAdaptersMismatch,
            subject: "checkpoint_adapters".into(),
            detail: "checkpoint adapters differ".into(),
        });
    }
    if runtime.bespoke_waivers != expected.bespoke_waivers {
        disagreements.push(ExportDisagreement {
            kind: ExportDisagreementKind::BespokeWaiversMismatch,
            subject: "bespoke_waivers".into(),
            detail: "bespoke utility crates or waivers differ".into(),
        });
    }

    if disagreements.is_empty() {
        Ok(())
    } else {
        Err(ExportDisagreementError { disagreements })
    }
}

fn diff_route_maps(
    runtime: &BTreeMap<String, ExportedRouteV2>,
    expected: &BTreeMap<String, ExportedRouteV2>,
    missing_kind: ExportDisagreementKind,
    extra_kind: ExportDisagreementKind,
    feature_kind: ExportDisagreementKind,
    definition_kind: ExportDisagreementKind,
    out: &mut Vec<ExportDisagreement>,
) {
    let all_keys: BTreeSet<&String> = runtime.keys().chain(expected.keys()).collect();
    for key in all_keys {
        match (runtime.get(key), expected.get(key)) {
            (None, Some(_)) => out.push(ExportDisagreement {
                kind: missing_kind,
                subject: key.clone(),
                detail: format!("route '{key}' is present in export but absent in linked runtime"),
            }),
            (Some(_), None) => out.push(ExportDisagreement {
                kind: extra_kind,
                subject: key.clone(),
                detail: format!("route '{key}' is present in linked runtime but absent in export"),
            }),
            (Some(rt), Some(exp)) if rt != exp => {
                if rt.features != exp.features {
                    out.push(ExportDisagreement {
                        kind: feature_kind,
                        subject: key.clone(),
                        detail: format!(
                            "route '{key}' computed HOOK-4 feature applicability differs between linked runtime and export"
                        ),
                    });
                }
                let mut rt_without_features = rt.clone();
                let mut exp_without_features = exp.clone();
                rt_without_features.features.clear();
                exp_without_features.features.clear();
                if rt_without_features != exp_without_features {
                    out.push(ExportDisagreement {
                        kind: definition_kind,
                        subject: key.clone(),
                        detail: format!(
                            "route '{key}' capabilities, defaults, limits, or modes differ between linked runtime and export"
                        ),
                    });
                }
            }
            _ => {}
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Exported Route Schema (`ExportedRouteV2`)
// -------------------------------------------------------------------------------------------------

/// Complete exported specification for one generation route (media or audio).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedRouteV2 {
    pub route_id: String,
    pub family: String,
    pub backend: String,
    pub modality: String,
    pub mac_only: bool,
    pub modes: Vec<String>,
    pub denoiser: ExportedModelBindingV2,
    pub defaults: ExportedRouteDefaultsV2,
    pub limits: ExportedProductLimitsV2,
    pub solvers: Vec<String>,
    pub schedules: Vec<String>,
    pub guidance: ExportedGuidanceSurfaceV2,
    pub conditioning: ExportedConditioningSurfaceV2,
    pub adapters: ExportedAdapterSurfaceV2,
    pub quant: ExportedQuantSurfaceV2,
    pub preview: String,
    pub decoders: ExportedDecoderSurfaceV2,
    pub seed_and_batch: ExportedSeedAndBatchV2,
    /// Computed `HOOK-4` Feature applicability, refusal reasons, and parameter limits, sorted by
    /// `feature_id`.
    pub features: Vec<ExportedFeatureApplicabilityV2>,
    pub memory: ExportedMemorySurfaceV2,
    pub weight_plan: ExportedWeightPlanV2,
    /// Complete computed `Capabilities` surface so every SceneWorks consumer can migrate without
    /// losing any field.
    pub capabilities: ExportedCapabilitiesV2,
}

impl ExportedRouteV2 {
    /// Construct an [`ExportedRouteV2`] purely from a primary [`RouteRecord`] and its owning
    /// [`ProviderRegistry`] context (for memory/plan registrations and `HOOK-4` evaluation).
    pub fn from_route_record(
        record: &RouteRecord,
        registry: Option<&ProviderRegistry>,
        control_required: bool,
    ) -> Self {
        let descriptor = record.compute_descriptor();
        let features_map = match registry {
            Some(reg) => reg.generator_feature_applicability(&descriptor),
            None => crate::FeatureRegistry::standard().evaluate_route_record(record),
        };
        let features: Vec<ExportedFeatureApplicabilityV2> = features_map
            .values()
            .map(ExportedFeatureApplicabilityV2::from_feature_applicability)
            .collect();

        let has_memory_contract = record.bindings.memory.contract.is_some()
            || registry.is_some_and(|reg| {
                reg.memory_strategy_registrations()
                    .any(|m| m.provider_id == record.route_id)
            });
        let has_memory_fixture = record.bindings.memory.fixture.is_some()
            || registry.is_some_and(|reg| {
                reg.memory_contract_fixture_registrations()
                    .any(|m| m.provider_id == record.route_id)
            });
        let has_memory_behavior = record.bindings.memory.behavior.is_some()
            || registry.is_some_and(|reg| {
                reg.memory_behavior_registrations()
                    .any(|m| m.provider_id == record.route_id)
            });
        let has_render_plan = record.bindings.render_plan.is_some()
            || registry.is_some_and(|reg| reg.render_plan_registration(record.route_id).is_some());
        let has_video_plan = record.bindings.video_plan.is_some()
            || registry.is_some_and(|reg| reg.has_video_plan(record.route_id));

        let capabilities = &descriptor.capabilities;
        let modes = derive_route_modes(record.modality, capabilities, control_required);
        let alternate_decoder_ids = derive_alternate_decoders(
            record.backend,
            record.route_id,
            record.structure.latent_codec.output_latent_space,
        );

        Self {
            route_id: record.route_id.to_string(),
            family: record.family.to_string(),
            backend: record.backend.to_string(),
            modality: modality_name(record.modality).to_string(),
            mac_only: record.mac_only,
            modes,
            denoiser: ExportedModelBindingV2::from_binding(&record.structure.denoiser),
            defaults: ExportedRouteDefaultsV2::from_defaults(&record.defaults),
            limits: ExportedProductLimitsV2::from_limits(&record.limits),
            solvers: capabilities
                .samplers
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            schedules: capabilities
                .schedulers
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            guidance: ExportedGuidanceSurfaceV2 {
                methods: capabilities
                    .supported_guidance_methods
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                supports_guidance: capabilities.supports_guidance,
                supports_true_cfg: capabilities.supports_true_cfg,
                supports_cfg_interval: capabilities.supports_cfg_interval,
                requires_sigma_shift: capabilities.requires_sigma_shift,
                supports_h3_image_sampler: capabilities.supports_h3_image_sampler,
                pag: capabilities.pag.clone(),
            },
            conditioning: ExportedConditioningSurfaceV2 {
                kinds: capabilities
                    .conditioning
                    .iter()
                    .copied()
                    .map(conditioning_name)
                    .map(str::to_string)
                    .collect(),
                control_kinds: record
                    .structure
                    .conditioning
                    .control_kinds
                    .as_ref()
                    .map(ExportedControlKindsV2::from_accepted),
                pass0_requirement: pass0_requirement_name(
                    record.structure.conditioning.pass0_requirement,
                )
                .to_string(),
                aligned: derive_conditioning_alignment(
                    &capabilities.conditioning,
                    record.structure.conditioning.pass0_requirement,
                ),
                supports_negative_prompt: capabilities.supports_negative_prompt,
                modules: record
                    .structure
                    .conditioning
                    .modules
                    .iter()
                    .map(|m| (*m).to_string())
                    .collect(),
            },
            adapters: ExportedAdapterSurfaceV2 {
                supports_lora: capabilities.supports_lora,
                supports_lokr: capabilities.supports_lokr,
                install_policy: adapter_install_policy_name(
                    record.structure.adapters.install_policy,
                )
                .to_string(),
                supports_per_phase: capabilities.supports_phases
                    && (capabilities.supports_lora || capabilities.supports_lokr),
            },
            quant: ExportedQuantSurfaceV2 {
                supported_tiers: capabilities
                    .supported_quants
                    .iter()
                    .copied()
                    .map(quant_name)
                    .map(str::to_string)
                    .collect(),
                component_precision_floors: capabilities
                    .component_precision_floors
                    .iter()
                    .copied()
                    .map(ExportedComponentPrecisionFloorV2::from_floor)
                    .collect(),
            },
            preview: preview_support_name(record.structure.output.preview).to_string(),
            decoders: ExportedDecoderSurfaceV2 {
                codec_id: record.structure.latent_codec.codec_id.map(str::to_string),
                output_latent_space: record
                    .structure
                    .latent_codec
                    .output_latent_space
                    .map(ExportedLatentSpaceV2::from_space),
                pixel_stride: record
                    .structure
                    .latent_codec
                    .effective_pixel_stride(record.limits.size_floor),
                token_width: record.structure.latent_codec.effective_token_width(),
                supports_diffusion_decoder: capabilities.supports_diffusion_decoder,
                alternate_decoder_ids,
            },
            seed_and_batch: ExportedSeedAndBatchV2::from_semantics(&record.seed_and_batch),
            features,
            memory: ExportedMemorySurfaceV2 {
                has_memory_contract,
                has_memory_fixture,
                has_memory_behavior,
                has_render_plan,
                render_plan_scope: has_render_plan
                    .then(|| render_plan_scope_name(record.bindings.render_plan_scope).to_string()),
                has_video_plan,
                supports_sequential_offload: capabilities.supports_sequential_offload,
                unconditionally_engages_staged_residency: capabilities
                    .unconditionally_engages_staged_residency,
                staged_residency_availability: staged_residency_name(
                    capabilities.staged_residency_availability(),
                )
                .to_string(),
                supports_kv_cache: capabilities.supports_kv_cache,
                request_memory_strategies: capabilities
                    .request_memory_strategies
                    .iter()
                    .copied()
                    .map(memory_strategy_name)
                    .map(str::to_string)
                    .collect(),
                residency_phases: record
                    .structure
                    .residency
                    .lifetime_graph
                    .iter()
                    .map(ExportedResidencyPhaseV2::from_node)
                    .collect(),
            },
            weight_plan: ExportedWeightPlanV2 {
                source_kind: weight_source_kind_name(record.structure.weight_plan.source_kind)
                    .to_string(),
                required_components: record
                    .structure
                    .weight_plan
                    .required_components
                    .iter()
                    .map(|c| (*c).to_string())
                    .collect(),
                requires_control_overlay: record.structure.weight_plan.requires_control_overlay,
                requires_ip_adapter_overlay: record
                    .structure
                    .weight_plan
                    .requires_ip_adapter_overlay,
                requires_identity_overlay: record.structure.weight_plan.requires_identity_overlay,
                text_encoders: record
                    .structure
                    .text_encoders
                    .encoders
                    .iter()
                    .map(|e| ExportedTextEncoderV2 {
                        name: e.name.to_string(),
                        max_sequence_length: e.max_sequence_length,
                    })
                    .collect(),
                upsamplers: record
                    .structure
                    .upsamplers
                    .components
                    .iter()
                    .map(|u| (*u).to_string())
                    .collect(),
                has_encoder_contract: record.structure.text_encoders.encoder_contract.is_some(),
            },
            capabilities: ExportedCapabilitiesV2::from_capabilities(capabilities),
        }
    }

    /// Fallback constructor when a generator exists without a `RouteRecord` (used only by
    /// synthetic test registries; production catalogs enforce 100% `RouteRecord` registration).
    pub fn from_descriptor(
        descriptor: &ModelDescriptor,
        registry: Option<&ProviderRegistry>,
        control_required: bool,
    ) -> Self {
        let capabilities = &descriptor.capabilities;
        let features_map = match registry {
            Some(reg) => reg.generator_feature_applicability(descriptor),
            None => {
                let route =
                    crate::FeatureRouteStructure::from_descriptor_and_plan(descriptor, None);
                crate::FeatureRegistry::standard().evaluate_route(&route)
            }
        };
        let features: Vec<ExportedFeatureApplicabilityV2> = features_map
            .values()
            .map(ExportedFeatureApplicabilityV2::from_feature_applicability)
            .collect();
        let has_memory_contract = registry.is_some_and(|reg| {
            reg.memory_strategy_registrations()
                .any(|m| m.provider_id == descriptor.id)
        });
        let has_memory_fixture = registry.is_some_and(|reg| {
            reg.memory_contract_fixture_registrations()
                .any(|m| m.provider_id == descriptor.id)
        });
        let has_memory_behavior = registry.is_some_and(|reg| {
            reg.memory_behavior_registrations()
                .any(|m| m.provider_id == descriptor.id)
        });
        let has_render_plan =
            registry.is_some_and(|reg| reg.render_plan_registration(descriptor.id).is_some());
        let has_video_plan = registry.is_some_and(|reg| reg.has_video_plan(descriptor.id));
        let modes = derive_route_modes(descriptor.modality, capabilities, control_required);
        let alternate_decoder_ids = derive_alternate_decoders(
            descriptor.backend,
            descriptor.id,
            descriptor.denoiser_output_latent_space,
        );

        Self {
            route_id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            modality: modality_name(descriptor.modality).to_string(),
            mac_only: capabilities.mac_only,
            modes,
            denoiser: ExportedModelBindingV2 {
                kind: "unspecified".into(),
                name: descriptor.family.to_string(),
                supports_geometry_change: capabilities.supports_low_res_start,
                supports_renoise_from_clean: capabilities.supports_low_res_start,
                supports_read_only_boundary_estimate: capabilities.supports_low_res_start,
                supports_multiphase_denoise: capabilities.supports_phases,
                supports_expert_switching: false,
                pag: capabilities.pag.clone(),
            },
            defaults: ExportedRouteDefaultsV2::default(),
            limits: ExportedProductLimitsV2 {
                min_size: capabilities.min_size,
                max_size: capabilities.max_size,
                size_floor: ExportedSizeFloorV2::from_size_floor(capabilities.size_floor),
                supported_steps: ExportedStepSupportV2::from_step_support(
                    &capabilities.supported_steps,
                ),
                max_audio_duration_secs: capabilities.max_audio_duration_secs,
            },
            solvers: capabilities
                .samplers
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            schedules: capabilities
                .schedulers
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            guidance: ExportedGuidanceSurfaceV2 {
                methods: capabilities
                    .supported_guidance_methods
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                supports_guidance: capabilities.supports_guidance,
                supports_true_cfg: capabilities.supports_true_cfg,
                supports_cfg_interval: capabilities.supports_cfg_interval,
                requires_sigma_shift: capabilities.requires_sigma_shift,
                supports_h3_image_sampler: capabilities.supports_h3_image_sampler,
                pag: capabilities.pag.clone(),
            },
            conditioning: ExportedConditioningSurfaceV2 {
                kinds: capabilities
                    .conditioning
                    .iter()
                    .copied()
                    .map(conditioning_name)
                    .map(str::to_string)
                    .collect(),
                control_kinds: descriptor
                    .control_kinds
                    .as_ref()
                    .map(ExportedControlKindsV2::from_accepted),
                pass0_requirement: pass0_requirement_name(
                    Pass0ConditioningRequirement::NoiseStartSupported,
                )
                .to_string(),
                aligned: derive_conditioning_alignment(
                    &capabilities.conditioning,
                    Pass0ConditioningRequirement::NoiseStartSupported,
                ),
                supports_negative_prompt: capabilities.supports_negative_prompt,
                modules: Vec::new(),
            },
            adapters: ExportedAdapterSurfaceV2 {
                supports_lora: capabilities.supports_lora,
                supports_lokr: capabilities.supports_lokr,
                install_policy: if capabilities.supports_lora || capabilities.supports_lokr {
                    "forward_residual".into()
                } else {
                    "unsupported".into()
                },
                supports_per_phase: capabilities.supports_phases
                    && (capabilities.supports_lora || capabilities.supports_lokr),
            },
            quant: ExportedQuantSurfaceV2 {
                supported_tiers: capabilities
                    .supported_quants
                    .iter()
                    .copied()
                    .map(quant_name)
                    .map(str::to_string)
                    .collect(),
                component_precision_floors: capabilities
                    .component_precision_floors
                    .iter()
                    .copied()
                    .map(ExportedComponentPrecisionFloorV2::from_floor)
                    .collect(),
            },
            preview: if capabilities.supports_preview {
                "latent_projection".into()
            } else {
                "unsupported".into()
            },
            decoders: ExportedDecoderSurfaceV2 {
                codec_id: None,
                output_latent_space: descriptor
                    .denoiser_output_latent_space
                    .map(ExportedLatentSpaceV2::from_space),
                pixel_stride: capabilities.size_floor.explicit_size_multiple(),
                token_width: None,
                supports_diffusion_decoder: capabilities.supports_diffusion_decoder,
                alternate_decoder_ids,
            },
            seed_and_batch: ExportedSeedAndBatchV2 {
                seed_scheme: seed_scheme_name(SeedScheme::DeterministicOffset).to_string(),
                image_index_seed_relation: seed_index_relation(SeedScheme::DeterministicOffset)
                    .to_string(),
                max_count: capabilities.max_count,
                supports_prompt_batch: capabilities.supports_prompt_batch,
            },
            features,
            memory: ExportedMemorySurfaceV2 {
                has_memory_contract,
                has_memory_fixture,
                has_memory_behavior,
                has_render_plan,
                render_plan_scope: has_render_plan.then(|| "single_and_multi_pass".into()),
                has_video_plan,
                supports_sequential_offload: capabilities.supports_sequential_offload,
                unconditionally_engages_staged_residency: capabilities
                    .unconditionally_engages_staged_residency,
                staged_residency_availability: staged_residency_name(
                    capabilities.staged_residency_availability(),
                )
                .to_string(),
                supports_kv_cache: capabilities.supports_kv_cache,
                request_memory_strategies: capabilities
                    .request_memory_strategies
                    .iter()
                    .copied()
                    .map(memory_strategy_name)
                    .map(str::to_string)
                    .collect(),
                residency_phases: Vec::new(),
            },
            weight_plan: ExportedWeightPlanV2 {
                source_kind: weight_source_kind_name(WeightSourceKind::SnapshotDirectory)
                    .to_string(),
                required_components: descriptor
                    .required_components
                    .iter()
                    .map(|c| (*c).to_string())
                    .collect(),
                requires_control_overlay: false,
                requires_ip_adapter_overlay: false,
                requires_identity_overlay: false,
                text_encoders: Vec::new(),
                upsamplers: Vec::new(),
                has_encoder_contract: descriptor.encoder_contract.is_some(),
            },
            capabilities: ExportedCapabilitiesV2::from_capabilities(capabilities),
        }
    }
}

/// Conditioned / composed feature sub-route exported alongside standalone generators.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedFeatureRouteV2 {
    pub route_id: String,
    pub parent_provider_id: Option<String>,
    pub family: String,
    pub backend: String,
    pub modality: String,
    pub has_render_plan: bool,
    pub features: Vec<ExportedFeatureApplicabilityV2>,
    pub capabilities: ExportedCapabilitiesV2,
}

/// Exported denoiser binding summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedModelBindingV2 {
    pub kind: String,
    pub name: String,
    pub supports_geometry_change: bool,
    pub supports_renoise_from_clean: bool,
    pub supports_read_only_boundary_estimate: bool,
    pub supports_multiphase_denoise: bool,
    pub supports_expert_switching: bool,
    pub pag: Option<PagCapability>,
}

impl ExportedModelBindingV2 {
    pub fn from_binding(binding: &ModelBinding) -> Self {
        let name = match binding {
            ModelBinding::FamilyModel(spec) => spec.model_id.to_string(),
            ModelBinding::FamilyDriver(spec) => spec.support.driver_id.to_string(),
            ModelBinding::NonDiffusion { architecture } => (*architecture).to_string(),
        };
        Self {
            kind: binding.kind_str().to_string(),
            name,
            supports_geometry_change: binding.supports_geometry_change(),
            supports_renoise_from_clean: binding.supports_renoise_from_clean(),
            supports_read_only_boundary_estimate: binding.supports_read_only_boundary_estimate(),
            supports_multiphase_denoise: binding.supports_multiphase_denoise(),
            supports_expert_switching: binding.supports_expert_switching(),
            pag: binding.pag().cloned(),
        }
    }
}

/// Owned, serializable form of [`RouteDefaults`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportedRouteDefaultsV2 {
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub sampler: Option<String>,
    pub schedule: Option<String>,
    pub shift: Option<f32>,
    pub strength: Option<f32>,
    pub dimensions: Option<(u32, u32)>,
    pub audio_voice: Option<String>,
    pub audio_sample_rate: Option<u32>,
    pub audio_language: Option<String>,
}

impl ExportedRouteDefaultsV2 {
    pub fn from_defaults(defaults: &RouteDefaults) -> Self {
        Self {
            steps: defaults.steps,
            guidance: defaults.guidance,
            sampler: defaults.sampler.map(str::to_string),
            schedule: defaults.schedule.map(str::to_string),
            shift: defaults.shift,
            strength: defaults.strength,
            dimensions: defaults.dimensions,
            audio_voice: defaults.audio_voice.map(str::to_string),
            audio_sample_rate: defaults.audio_sample_rate,
            audio_language: defaults.audio_language.map(str::to_string),
        }
    }
}

/// Owned, serializable form of [`ProductLimits`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedProductLimitsV2 {
    pub min_size: u32,
    pub max_size: u32,
    pub size_floor: ExportedSizeFloorV2,
    pub supported_steps: ExportedStepSupportV2,
    pub max_audio_duration_secs: Option<f32>,
}

impl ExportedProductLimitsV2 {
    pub fn from_limits(limits: &ProductLimits) -> Self {
        Self {
            min_size: limits.min_size,
            max_size: limits.max_size,
            size_floor: ExportedSizeFloorV2::from_size_floor(limits.size_floor),
            supported_steps: ExportedStepSupportV2::from_step_support(&limits.supported_steps),
            max_audio_duration_secs: limits.max_audio_duration_secs,
        }
    }
}

/// Serializable representation of [`SizeFloor`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedSizeFloorV2 {
    RangeChecked,
    RangeCheckedOnGrid { multiple: u32 },
    RangeCheckedOnGridArea { multiple: u32, max_cells: u32 },
    ResolvedDownstream,
    ResolvedDownstreamExplicitGrid { multiple: u32 },
}

impl ExportedSizeFloorV2 {
    pub fn from_size_floor(floor: SizeFloor) -> Self {
        match floor {
            SizeFloor::RangeChecked => Self::RangeChecked,
            SizeFloor::RangeCheckedOnGrid { multiple } => Self::RangeCheckedOnGrid { multiple },
            SizeFloor::RangeCheckedOnGridArea {
                multiple,
                max_cells,
            } => Self::RangeCheckedOnGridArea {
                multiple,
                max_cells,
            },
            SizeFloor::ResolvedDownstream => Self::ResolvedDownstream,
            SizeFloor::ResolvedDownstreamExplicitGrid { multiple } => {
                Self::ResolvedDownstreamExplicitGrid { multiple }
            }
        }
    }
}

/// Serializable representation of [`StepSupport`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedStepSupportV2 {
    Unconstrained,
    Exact { values: Vec<u32> },
    Range { min: u32, max: u32 },
}

impl ExportedStepSupportV2 {
    pub fn from_step_support(support: &StepSupport) -> Self {
        match support {
            StepSupport::Unconstrained => Self::Unconstrained,
            StepSupport::Exact(values) => Self::Exact {
                values: values.to_vec(),
            },
            StepSupport::Range { min, max } => Self::Range {
                min: *min,
                max: *max,
            },
        }
    }
}

/// Exported guidance surface for a route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedGuidanceSurfaceV2 {
    pub methods: Vec<String>,
    pub supports_guidance: bool,
    pub supports_true_cfg: bool,
    pub supports_cfg_interval: bool,
    pub requires_sigma_shift: bool,
    pub supports_h3_image_sampler: bool,
    pub pag: Option<PagCapability>,
}

/// Whether a route's conditioning requires pixel-grid alignment with pass-0 geometry or is
/// geometry-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningAlignment {
    GridAligned,
    GeometryFree,
}

/// Exported control-kind acceptance policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedControlKindsV2 {
    Any,
    Only { kinds: Vec<String> },
}

impl ExportedControlKindsV2 {
    pub fn from_accepted(accepted: &AcceptedControlKinds) -> Self {
        match accepted {
            AcceptedControlKinds::Any => Self::Any,
            AcceptedControlKinds::Only(kinds) => Self::Only {
                kinds: kinds.iter().map(control_kind_name).collect(),
            },
        }
    }
}

/// Exported conditioning surface for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedConditioningSurfaceV2 {
    pub kinds: Vec<String>,
    pub control_kinds: Option<ExportedControlKindsV2>,
    pub pass0_requirement: String,
    pub aligned: ConditioningAlignment,
    pub supports_negative_prompt: bool,
    pub modules: Vec<String>,
}

/// Exported runtime adapter surface for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedAdapterSurfaceV2 {
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub install_policy: String,
    pub supports_per_phase: bool,
}

/// Exported component precision floor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedComponentPrecisionFloorV2 {
    pub component: String,
    pub selected_tier: String,
    pub resident_tier: String,
}

impl ExportedComponentPrecisionFloorV2 {
    pub fn from_floor(floor: ComponentPrecisionFloor) -> Self {
        Self {
            component: floor.component.as_str().to_string(),
            selected_tier: quant_name(floor.selected_tier).to_string(),
            resident_tier: quant_name(floor.resident_tier).to_string(),
        }
    }
}

/// Exported quantization surface for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedQuantSurfaceV2 {
    pub supported_tiers: Vec<String>,
    pub component_precision_floors: Vec<ExportedComponentPrecisionFloorV2>,
}

/// Exported latent space descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedLatentSpaceV2 {
    pub channels: u16,
    pub spatial_compression_height: u16,
    pub spatial_compression_width: u16,
    pub patch_layout: String,
    pub temporal_law: String,
    pub normalization: String,
}

impl ExportedLatentSpaceV2 {
    pub fn from_space(space: &LatentSpace) -> Self {
        let patch_layout = match space.patch_layout {
            LatentPatchLayout::Unpacked => "unpacked".to_string(),
            LatentPatchLayout::Packed {
                patch_height,
                patch_width,
            } => format!("packed_{patch_height}x{patch_width}"),
        };
        let temporal_law = match space.temporal_law {
            LatentTemporalLaw::None => "none",
            LatentTemporalLaw::Causal4x => "causal_4x",
            LatentTemporalLaw::Causal6x => "causal_6x",
            LatentTemporalLaw::Causal8x => "causal_8x",
        };
        let normalization = match space.normalization {
            LatentNormalization::Identity => "identity".to_string(),
            LatentNormalization::Affine {
                scale_bits,
                shift_bits,
            } => format!("affine:{scale_bits:08x}:{shift_bits:08x}"),
            LatentNormalization::PerChannel(stats) => {
                format!("per_channel:{}:ch={}", stats.identity, stats.channels)
            }
            LatentNormalization::LearnedPerChannel { identity } => {
                format!("learned_per_channel:{identity}")
            }
        };
        Self {
            channels: space.channels,
            spatial_compression_height: space.spatial_compression.height,
            spatial_compression_width: space.spatial_compression.width,
            patch_layout,
            temporal_law: temporal_law.to_string(),
            normalization,
        }
    }
}

/// Exported decoder / latent-codec surface for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedDecoderSurfaceV2 {
    pub codec_id: Option<String>,
    pub output_latent_space: Option<ExportedLatentSpaceV2>,
    pub pixel_stride: Option<u32>,
    pub token_width: Option<u32>,
    pub supports_diffusion_decoder: bool,
    pub alternate_decoder_ids: Vec<String>,
}

/// Exported seed derivation & batch count semantics for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedSeedAndBatchV2 {
    pub seed_scheme: String,
    pub image_index_seed_relation: String,
    pub max_count: u32,
    pub supports_prompt_batch: bool,
}

impl ExportedSeedAndBatchV2 {
    pub fn from_semantics(sem: &SeedAndBatchSemantics) -> Self {
        Self {
            seed_scheme: seed_scheme_name(sem.seed_scheme).to_string(),
            image_index_seed_relation: seed_index_relation(sem.seed_scheme).to_string(),
            max_count: sem.max_count,
            supports_prompt_batch: sem.supports_prompt_batch,
        }
    }
}

/// Typed `HOOK-4` refusal reason for a non-applicable Feature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedFeatureRefusalV2 {
    pub code: String,
    pub message: String,
}

/// Owned representation of `FeatureRelations`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedFeatureRelationsV2 {
    pub conflicts_with_features: Vec<String>,
    pub conflicts_with_axes: Vec<String>,
    pub requires_features: Vec<String>,
    pub requires_capabilities: Vec<String>,
    pub before: Vec<String>,
    pub after: Vec<String>,
}

/// Owned representation of `FeatureMemoryEffect`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedFeatureMemoryEffectV2 {
    pub extra_resident_weight_bytes: u64,
    pub transition_residency: String,
    pub introduces_denoise_phase_boundary: bool,
    pub introduces_pass_boundary: bool,
    pub retains_single_pass_schedule: bool,
}

impl ExportedFeatureMemoryEffectV2 {
    pub fn from_effect(effect: &FeatureMemoryEffect) -> Self {
        let transition_residency = match effect.transition_residency {
            TransitionResidency::None => "none",
            TransitionResidency::InDeviceLatentUnpackResizeRepack => {
                "in_device_latent_unpack_resize_repack"
            }
            TransitionResidency::PixelRoundTripProducedReference => {
                "pixel_round_trip_produced_reference"
            }
            TransitionResidency::StagedExpertWeightHandoff => "staged_expert_weight_handoff",
        };
        Self {
            extra_resident_weight_bytes: effect.extra_resident_weight_bytes,
            transition_residency: transition_residency.to_string(),
            introduces_denoise_phase_boundary: effect.introduces_denoise_phase_boundary,
            introduces_pass_boundary: effect.introduces_pass_boundary,
            retains_single_pass_schedule: effect.retains_single_pass_schedule,
        }
    }
}

/// Exported `HOOK-4` Feature applicability, refusal reasons, and parameter limits for one route.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedFeatureApplicabilityV2 {
    pub feature_id: String,
    pub version: u32,
    pub class: String,
    pub applicable: bool,
    pub parameter_schema: serde_json::Value,
    pub relations: ExportedFeatureRelationsV2,
    pub hook_points: Vec<String>,
    pub memory_effect: ExportedFeatureMemoryEffectV2,
    pub reasons: Vec<ExportedFeatureRefusalV2>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub conditioned_routes: BTreeMap<String, ExportedFeatureApplicabilityV2>,
}

impl ExportedFeatureApplicabilityV2 {
    pub fn from_feature_applicability(app: &FeatureApplicability) -> Self {
        Self {
            feature_id: app.feature_id.to_string(),
            version: app.version,
            class: app.class.as_str().to_string(),
            applicable: app.applicable,
            parameter_schema: canonicalize_json_value(app.parameter_schema.to_json_map()),
            relations: ExportedFeatureRelationsV2 {
                conflicts_with_features: app
                    .relations
                    .conflicts_with_features
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                conflicts_with_axes: app
                    .relations
                    .conflicts_with_axes
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                requires_features: app
                    .relations
                    .requires_features
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                requires_capabilities: app
                    .relations
                    .requires_capabilities
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                before: app
                    .relations
                    .before
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
                after: app
                    .relations
                    .after
                    .iter()
                    .map(|s| (*s).to_string())
                    .collect(),
            },
            hook_points: app
                .hook_points
                .iter()
                .map(|p| p.as_str().to_string())
                .collect(),
            memory_effect: ExportedFeatureMemoryEffectV2::from_effect(&app.memory_effect),
            reasons: app
                .reasons
                .iter()
                .map(|r| ExportedFeatureRefusalV2 {
                    code: r.code.as_str().to_string(),
                    message: r.message.clone(),
                })
                .collect(),
            conditioned_routes: app
                .conditioned_routes
                .iter()
                .map(|(k, v)| (k.clone(), Self::from_feature_applicability(v)))
                .collect(),
        }
    }
}

/// Exported residency lifetime node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedResidencyPhaseV2 {
    pub phase: String,
    pub components: Vec<String>,
    pub droppable_after_phase: bool,
}

impl ExportedResidencyPhaseV2 {
    pub fn from_node(node: &ResidencyPhaseNode) -> Self {
        let phase = match node.phase {
            ResidencyPhaseKind::PromptEncode => "prompt_encode",
            ResidencyPhaseKind::ConditioningEncode => "conditioning_encode",
            ResidencyPhaseKind::Denoise => "denoise",
            ResidencyPhaseKind::Upsample => "upsample",
            ResidencyPhaseKind::Decode => "decode",
            ResidencyPhaseKind::AudioSynthesize => "audio_synthesize",
        };
        Self {
            phase: phase.to_string(),
            components: node.components.iter().map(|c| (*c).to_string()).collect(),
            droppable_after_phase: node.droppable_after_phase,
        }
    }
}

/// Exported memory contract & residency strategy surface for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedMemorySurfaceV2 {
    pub has_memory_contract: bool,
    pub has_memory_fixture: bool,
    pub has_memory_behavior: bool,
    pub has_render_plan: bool,
    pub render_plan_scope: Option<String>,
    pub has_video_plan: bool,
    pub supports_sequential_offload: bool,
    pub unconditionally_engages_staged_residency: bool,
    pub staged_residency_availability: String,
    pub supports_kv_cache: bool,
    pub request_memory_strategies: Vec<String>,
    pub residency_phases: Vec<ExportedResidencyPhaseV2>,
}

/// Exported text encoder component declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedTextEncoderV2 {
    pub name: String,
    pub max_sequence_length: Option<u32>,
}

/// Exported weight plan and component requirements for a route.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedWeightPlanV2 {
    pub source_kind: String,
    pub required_components: Vec<String>,
    pub requires_control_overlay: bool,
    pub requires_ip_adapter_overlay: bool,
    pub requires_identity_overlay: bool,
    pub text_encoders: Vec<ExportedTextEncoderV2>,
    pub upsamplers: Vec<String>,
    pub has_encoder_contract: bool,
}

// -------------------------------------------------------------------------------------------------
// Complete Owned `Capabilities` Mirror (`ExportedCapabilitiesV2`)
// -------------------------------------------------------------------------------------------------

/// Complete owned, serializable representation of [`Capabilities`], preserving every field so
/// SceneWorks consumers can read the full capability contract from `sga.capabilities/2` (`SW-6983`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedCapabilitiesV2 {
    pub supports_negative_prompt: bool,
    pub supports_guidance: bool,
    pub supports_true_cfg: bool,
    pub conditioning: Vec<String>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub samplers: Vec<String>,
    pub schedulers: Vec<String>,
    pub supported_guidance_methods: Vec<String>,
    pub min_size: u32,
    pub max_size: u32,
    pub max_count: u32,
    pub supported_steps: ExportedStepSupportV2,
    pub mac_only: bool,
    pub size_floor: ExportedSizeFloorV2,
    pub audio_sample_rates: Vec<u32>,
    pub max_audio_duration_secs: Option<f32>,
    pub audio_voices: Vec<String>,
    pub audio_languages: Vec<String>,
    pub audio_edit_modes: Vec<String>,
    pub supports_preview: bool,
    pub supports_hdr: bool,
    pub supports_cfg_interval: bool,
    pub supports_prompt_batch: bool,
    pub supports_caption_overflow_policy: bool,
    pub supports_h3_image_sampler: bool,
    pub max_prompt_tokens: Option<u32>,
    pub supports_alpha_output: bool,
    pub supports_prompt_enhancement: bool,
    pub supports_streaming: bool,
    pub supports_multi_speaker: bool,
    pub max_speakers: Option<u32>,
    pub supports_segmented_lyrics: bool,
    pub supports_repetition_penalty: bool,
    pub supports_reference_region: bool,
    pub supports_output_limiter: bool,
    pub supports_symbolic_song: bool,
    pub supports_audio_artifacts: bool,
    pub supports_song_plan_only: bool,
    pub supports_song_cover: bool,
    pub request_memory_strategies: Vec<String>,
    pub supports_auto_duration: bool,
    pub supports_generated_keyframes: bool,
    pub max_temporal_upsample_rounds: u32,
    pub supports_diffusion_decoder: bool,
    pub supports_conversation_history: bool,
    pub supports_conversation_session: bool,
    pub supported_quants: Vec<String>,
    pub component_precision_floors: Vec<ExportedComponentPrecisionFloorV2>,
    pub supports_kv_cache: bool,
    pub requires_sigma_shift: bool,
    pub supports_sequential_offload: bool,
    pub unconditionally_engages_staged_residency: bool,
    pub execution: ExportedExecutionSurfaceV2,
    pub approximation: ExportedApproximationSurfaceV2,
    pub pag: Option<PagCapability>,
    pub supports_phases: bool,
    pub supports_low_res_start: bool,
}

impl ExportedCapabilitiesV2 {
    pub fn from_capabilities(caps: &Capabilities) -> Self {
        Self {
            supports_negative_prompt: caps.supports_negative_prompt,
            supports_guidance: caps.supports_guidance,
            supports_true_cfg: caps.supports_true_cfg,
            conditioning: caps
                .conditioning
                .iter()
                .copied()
                .map(conditioning_name)
                .map(str::to_string)
                .collect(),
            supports_lora: caps.supports_lora,
            supports_lokr: caps.supports_lokr,
            samplers: caps.samplers.iter().map(|s| (*s).to_string()).collect(),
            schedulers: caps.schedulers.iter().map(|s| (*s).to_string()).collect(),
            supported_guidance_methods: caps
                .supported_guidance_methods
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
            min_size: caps.min_size,
            max_size: caps.max_size,
            max_count: caps.max_count,
            supported_steps: ExportedStepSupportV2::from_step_support(&caps.supported_steps),
            mac_only: caps.mac_only,
            size_floor: ExportedSizeFloorV2::from_size_floor(caps.size_floor),
            audio_sample_rates: caps.audio_sample_rates.clone(),
            max_audio_duration_secs: caps.max_audio_duration_secs,
            audio_voices: caps.audio_voices.iter().map(|v| (*v).to_string()).collect(),
            audio_languages: caps
                .audio_languages
                .iter()
                .map(|l| (*l).to_string())
                .collect(),
            audio_edit_modes: caps
                .audio_edit_modes
                .iter()
                .copied()
                .map(audio_edit_mode_name)
                .map(str::to_string)
                .collect(),
            supports_preview: caps.supports_preview,
            supports_hdr: caps.supports_hdr,
            supports_cfg_interval: caps.supports_cfg_interval,
            supports_prompt_batch: caps.supports_prompt_batch,
            supports_caption_overflow_policy: caps.supports_caption_overflow_policy,
            supports_h3_image_sampler: caps.supports_h3_image_sampler,
            max_prompt_tokens: caps.max_prompt_tokens,
            supports_alpha_output: caps.supports_alpha_output,
            supports_prompt_enhancement: caps.supports_prompt_enhancement,
            supports_streaming: caps.supports_streaming,
            supports_multi_speaker: caps.supports_multi_speaker,
            max_speakers: caps.max_speakers,
            supports_segmented_lyrics: caps.supports_segmented_lyrics,
            supports_repetition_penalty: caps.supports_repetition_penalty,
            supports_reference_region: caps.supports_reference_region,
            supports_output_limiter: caps.supports_output_limiter,
            supports_symbolic_song: caps.supports_symbolic_song,
            supports_audio_artifacts: caps.supports_audio_artifacts,
            supports_song_plan_only: caps.supports_song_plan_only,
            supports_song_cover: caps.supports_song_cover,
            request_memory_strategies: caps
                .request_memory_strategies
                .iter()
                .copied()
                .map(memory_strategy_name)
                .map(str::to_string)
                .collect(),
            supports_auto_duration: caps.supports_auto_duration,
            supports_generated_keyframes: caps.supports_generated_keyframes,
            max_temporal_upsample_rounds: caps.max_temporal_upsample_rounds,
            supports_diffusion_decoder: caps.supports_diffusion_decoder,
            supports_conversation_history: caps.supports_conversation_history,
            supports_conversation_session: caps.supports_conversation_session,
            supported_quants: caps
                .supported_quants
                .iter()
                .copied()
                .map(quant_name)
                .map(str::to_string)
                .collect(),
            component_precision_floors: caps
                .component_precision_floors
                .iter()
                .copied()
                .map(ExportedComponentPrecisionFloorV2::from_floor)
                .collect(),
            supports_kv_cache: caps.supports_kv_cache,
            requires_sigma_shift: caps.requires_sigma_shift,
            supports_sequential_offload: caps.supports_sequential_offload,
            unconditionally_engages_staged_residency: caps.unconditionally_engages_staged_residency,
            execution: ExportedExecutionSurfaceV2::from_surface(&caps.execution),
            approximation: ExportedApproximationSurfaceV2::from_surface(&caps.approximation),
            pag: caps.pag.clone(),
            supports_phases: caps.supports_phases,
            supports_low_res_start: caps.supports_low_res_start,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedExecutionValueDomainV2 {
    Unsupported,
    AtLeast { floor: u32 },
    Candidates { values: Vec<u32> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedCfgBatchingDomainV2 {
    Unsupported,
    Modes { modes: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedPrefixKvCacheDomainV2 {
    Unsupported,
    Modes { modes: Vec<String> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedExecutionSurfaceV2 {
    pub graph_eval_cadence_blocks: ExportedExecutionValueDomainV2,
    pub ffn_chunk_rows: ExportedExecutionValueDomainV2,
    pub cfg_batching: ExportedCfgBatchingDomainV2,
    pub prefix_kv_cache: ExportedPrefixKvCacheDomainV2,
}

impl ExportedExecutionSurfaceV2 {
    pub fn from_surface(surface: &ExecutionSurface) -> Self {
        let convert_val = |d: &ExecutionValueDomain| match d {
            ExecutionValueDomain::Unsupported => ExportedExecutionValueDomainV2::Unsupported,
            ExecutionValueDomain::AtLeast(floor) => {
                ExportedExecutionValueDomainV2::AtLeast { floor: floor.get() }
            }
            ExecutionValueDomain::Candidates(values) => {
                ExportedExecutionValueDomainV2::Candidates {
                    values: values.clone(),
                }
            }
        };
        let cfg_batching = match &surface.cfg_batching {
            CfgBatchingDomain::Unsupported => ExportedCfgBatchingDomainV2::Unsupported,
            CfgBatchingDomain::Modes(modes) => ExportedCfgBatchingDomainV2::Modes {
                modes: modes
                    .iter()
                    .map(|m| match m {
                        CfgBatching::Batched => "batched".into(),
                        CfgBatching::Sequential => "sequential".into(),
                    })
                    .collect(),
            },
        };
        let prefix_kv_cache = match &surface.prefix_kv_cache {
            PrefixKvCacheDomain::Unsupported => ExportedPrefixKvCacheDomainV2::Unsupported,
            PrefixKvCacheDomain::Modes(modes) => ExportedPrefixKvCacheDomainV2::Modes {
                modes: modes
                    .iter()
                    .map(|m| match m {
                        PrefixKvCacheMode::Cached => "cached".into(),
                        PrefixKvCacheMode::Prefill => "prefill".into(),
                    })
                    .collect(),
            },
        };
        Self {
            graph_eval_cadence_blocks: convert_val(&surface.graph_eval_cadence_blocks),
            ffn_chunk_rows: convert_val(&surface.ffn_chunk_rows),
            cfg_batching,
            prefix_kv_cache,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedFeatureCacheDomainV2 {
    Unsupported,
    Implemented {
        intervals: Vec<u32>,
        max_warmup_steps: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExportedTokenPruningDomainV2 {
    Unsupported,
    Implemented {
        drop_strides: Vec<u32>,
        max_warmup_steps: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedApproximationSurfaceV2 {
    pub denoise_feature_cache: ExportedFeatureCacheDomainV2,
    pub token_pruning: ExportedTokenPruningDomainV2,
    pub characterization_bound: bool,
}

impl ExportedApproximationSurfaceV2 {
    pub fn from_surface(surface: &ApproximationSurface) -> Self {
        let denoise_feature_cache = match &surface.denoise_feature_cache {
            FeatureCacheDomain::Unsupported => ExportedFeatureCacheDomainV2::Unsupported,
            FeatureCacheDomain::Implemented {
                intervals,
                max_warmup_steps,
            } => ExportedFeatureCacheDomainV2::Implemented {
                intervals: intervals.clone(),
                max_warmup_steps: *max_warmup_steps,
            },
        };
        let token_pruning = match &surface.token_pruning {
            TokenPruningDomain::Unsupported => ExportedTokenPruningDomainV2::Unsupported,
            TokenPruningDomain::Implemented {
                drop_strides,
                max_warmup_steps,
            } => ExportedTokenPruningDomainV2::Implemented {
                drop_strides: drop_strides.clone(),
                max_warmup_steps: *max_warmup_steps,
            },
        };
        let characterization_bound =
            matches!(surface.characterization, CharacterizationBinding::Bound(_));
        Self {
            denoise_feature_cache,
            token_pruning,
            characterization_bound,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Independent Entrypoints (LLM/Vector, Trainers, Captioners, Transcribers, Embedders, Transforms,
// Imported Models, Checkpoint Adapters, Bespoke Waivers)
// -------------------------------------------------------------------------------------------------

/// Exported text-LLM or vector-generation (StarVector) entrypoint (`M8`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedLlmEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    /// Distinguishes `"text_llm"` from `"vector"` (StarVector SVG generators).
    pub entrypoint_kind: String,
    pub modality: String,
    pub capabilities: ExportedLlmCapabilitiesV2,
}

impl ExportedLlmEntry {
    pub fn from_descriptor(descriptor: &TextLlmDescriptor) -> Self {
        let is_vector =
            descriptor.family.starts_with("starvector") || descriptor.id.contains("starvector");
        let entrypoint_kind = if is_vector { "vector" } else { "text_llm" };
        let modality = if is_vector {
            "image_to_svg"
        } else if descriptor.capabilities.supports_vision && descriptor.capabilities.supports_audio
        {
            "multimodal_text_vision_audio"
        } else if descriptor.capabilities.supports_vision {
            "vision_language"
        } else {
            "text"
        };
        Self {
            id: descriptor.id.clone(),
            family: descriptor.family.clone(),
            backend: descriptor.backend.clone(),
            entrypoint_kind: entrypoint_kind.to_string(),
            modality: modality.to_string(),
            capabilities: ExportedLlmCapabilitiesV2::from_capabilities(&descriptor.capabilities),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedMtpCapabilitiesV2 {
    pub max_draft_tokens: u32,
    pub recommended_draft_tokens: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedLlmCapabilitiesV2 {
    pub max_context_tokens: usize,
    pub max_new_tokens: u32,
    pub supports_system_prompt: bool,
    pub supports_vision: bool,
    pub supports_video: bool,
    pub supports_audio: bool,
    pub supports_thinking: bool,
    pub supports_reasoning_effort: bool,
    pub reasoning_efforts: Vec<String>,
    pub supports_preserve_thinking: bool,
    pub supports_tools: bool,
    pub mtp: Option<ExportedMtpCapabilitiesV2>,
    pub speculative: Vec<core_llm::ProposerCapabilities>,
    pub tree_verify: Option<core_llm::TreeVerifyCapabilities>,
    pub prefix_checkpoints: Option<core_llm::PrefixCheckpointCapabilities>,
    pub supported_constraints: Vec<String>,
}

impl ExportedLlmCapabilitiesV2 {
    pub fn from_capabilities(caps: &core_llm::TextLlmCapabilities) -> Self {
        Self {
            max_context_tokens: caps.max_context_tokens,
            max_new_tokens: caps.max_new_tokens,
            supports_system_prompt: caps.supports_system_prompt,
            supports_vision: caps.supports_vision,
            supports_video: caps.supports_video,
            supports_audio: caps.supports_audio,
            supports_thinking: caps.supports_thinking,
            supports_reasoning_effort: caps.supports_reasoning_effort,
            reasoning_efforts: caps
                .reasoning_efforts
                .iter()
                .map(|r| r.as_str().to_string())
                .collect(),
            supports_preserve_thinking: caps.supports_preserve_thinking,
            supports_tools: caps.supports_tools,
            mtp: caps.mtp.map(|m| ExportedMtpCapabilitiesV2 {
                max_draft_tokens: m.max_draft_tokens,
                recommended_draft_tokens: m.recommended_draft_tokens,
            }),
            speculative: caps.speculative.clone(),
            tree_verify: caps.tree_verify,
            prefix_checkpoints: caps.prefix_checkpoints,
            supported_constraints: caps
                .supported_constraints
                .iter()
                .map(|c| match c {
                    ConstraintKind::Json => "json".into(),
                    ConstraintKind::JsonSchema => "json_schema".into(),
                })
                .collect(),
        }
    }
}

/// Exported media trainer entrypoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedTrainerEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub modality: String,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub supports_control: bool,
    pub supports_full_finetune: bool,
    pub max_reference_images: u32,
    pub supports_weight_noise: bool,
    pub supports_depth_anchoring: bool,
    pub supports_gradient_noise: bool,
    pub supports_resolution_buckets: bool,
    pub supports_subject_mask_loss: bool,
    pub supports_identity_loss: bool,
    pub supports_face_landmark_loss: bool,
    pub supports_body_proportion_loss: bool,
    pub supports_body_shape_loss: bool,
    pub supports_normal_loss: bool,
    pub supports_vae_anchor_loss: bool,
    pub supports_latent_lpips_loss: bool,
    pub supports_training_adapter: bool,
}

impl ExportedTrainerEntry {
    pub fn from_descriptor(descriptor: &TrainerDescriptor) -> Self {
        Self {
            id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            modality: modality_name(descriptor.modality).to_string(),
            supports_lora: descriptor.supports_lora,
            supports_lokr: descriptor.supports_lokr,
            supports_control: descriptor.supports_control,
            supports_full_finetune: descriptor.supports_full_finetune,
            max_reference_images: descriptor.max_reference_images,
            supports_weight_noise: descriptor.techniques.weight_noise,
            supports_depth_anchoring: descriptor.techniques.depth_anchoring,
            supports_gradient_noise: descriptor.techniques.gradient_noise,
            supports_resolution_buckets: descriptor.techniques.resolution_buckets,
            supports_subject_mask_loss: descriptor.techniques.subject_mask_loss,
            supports_identity_loss: descriptor.techniques.identity_loss,
            supports_face_landmark_loss: descriptor.techniques.face_landmark_loss,
            supports_body_proportion_loss: descriptor.techniques.body_proportion_loss,
            supports_body_shape_loss: descriptor.techniques.body_shape_loss,
            supports_normal_loss: descriptor.techniques.normal_loss,
            supports_vae_anchor_loss: descriptor.techniques.vae_anchor_loss,
            supports_latent_lpips_loss: descriptor.techniques.latent_lpips_loss,
            supports_training_adapter: descriptor.techniques.training_adapter,
        }
    }
}

/// Exported media captioner (image → text) entrypoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedCaptionerEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub caption_types: Vec<String>,
    pub caption_lengths: Vec<String>,
    pub supports_custom_prompt: bool,
    pub supports_low_vram: bool,
    pub min_image_size: u32,
    pub max_image_size: u32,
    pub max_prompt_chars: usize,
    pub max_name_chars: usize,
    pub max_extra_options: usize,
    pub max_extra_option_chars: usize,
    pub max_trigger_words: usize,
    pub max_trigger_word_chars: usize,
    pub max_new_tokens: u32,
    pub mac_only: bool,
}

impl ExportedCaptionerEntry {
    pub fn from_descriptor(descriptor: &CaptionerDescriptor) -> Self {
        let c = &descriptor.capabilities;
        Self {
            id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            caption_types: c.caption_types.iter().map(|s| (*s).to_string()).collect(),
            caption_lengths: c.caption_lengths.iter().map(|s| (*s).to_string()).collect(),
            supports_custom_prompt: c.supports_custom_prompt,
            supports_low_vram: c.supports_low_vram,
            min_image_size: c.min_image_size,
            max_image_size: c.max_image_size,
            max_prompt_chars: c.max_prompt_chars,
            max_name_chars: c.max_name_chars,
            max_extra_options: c.max_extra_options,
            max_extra_option_chars: c.max_extra_option_chars,
            max_trigger_words: c.max_trigger_words,
            max_trigger_word_chars: c.max_trigger_word_chars,
            max_new_tokens: c.max_new_tokens,
            mac_only: c.mac_only,
        }
    }
}

/// Exported audio transcriber (audio → text) entrypoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedTranscriberEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub languages: Vec<String>,
    pub supports_translate: bool,
    pub supports_segment_timestamps: bool,
    pub supports_word_timestamps: bool,
    pub max_audio_seconds: f32,
    pub max_new_tokens: u32,
    pub mac_only: bool,
}

impl ExportedTranscriberEntry {
    pub fn from_descriptor(descriptor: &TranscriberDescriptor) -> Self {
        let c = &descriptor.capabilities;
        Self {
            id: descriptor.id.to_string(),
            family: descriptor.family.to_string(),
            backend: descriptor.backend.to_string(),
            languages: c.languages.iter().map(|s| (*s).to_string()).collect(),
            supports_translate: c.supports_translate,
            supports_segment_timestamps: c.supports_segment_timestamps,
            supports_word_timestamps: c.supports_word_timestamps,
            max_audio_seconds: c.max_audio_seconds,
            max_new_tokens: c.max_new_tokens,
            mac_only: c.mac_only,
        }
    }
}

/// Exported embedders grouped by modality (`M8`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedEmbeddersV2 {
    pub image: BTreeMap<String, ExportedImageEmbedderEntry>,
    pub text: BTreeMap<String, ExportedTextEmbedderEntry>,
    pub voice: BTreeMap<String, ExportedVoiceEmbedderEntry>,
    pub audio: BTreeMap<String, ExportedAudioEmbedderEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedImageEmbedderEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub embedding_dim: usize,
    pub space: String,
    pub mac_only: bool,
}

impl ExportedImageEmbedderEntry {
    pub fn from_descriptor(d: &ImageEmbedderDescriptor) -> Self {
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            embedding_dim: d.embedding_dim,
            space: d.space.to_string(),
            mac_only: d.mac_only,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedTextEmbedderEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub embedding_dim: usize,
    pub space: String,
    pub mac_only: bool,
}

impl ExportedTextEmbedderEntry {
    pub fn from_descriptor(d: &TextEmbedderDescriptor) -> Self {
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            embedding_dim: d.embedding_dim,
            space: d.space.to_string(),
            mac_only: d.mac_only,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedVoiceEmbedderEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub embedding_dim: usize,
    pub mac_only: bool,
}

impl ExportedVoiceEmbedderEntry {
    pub fn from_descriptor(d: &VoiceEmbedderDescriptor) -> Self {
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            embedding_dim: d.embedding_dim,
            mac_only: d.mac_only,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedAudioEmbedderEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub embedding_dim: usize,
    pub space: String,
    pub mac_only: bool,
}

impl ExportedAudioEmbedderEntry {
    pub fn from_descriptor(d: &AudioEmbedderDescriptor) -> Self {
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            embedding_dim: d.embedding_dim,
            space: d.space.to_string(),
            mac_only: d.mac_only,
        }
    }
}

/// Exported non-prompt transforms grouped by modality (`M8`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportedTransformsV2 {
    pub image: BTreeMap<String, ExportedTransformEntry>,
    pub audio: BTreeMap<String, ExportedAudioTransformEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedInputBudgetV2 {
    pub short_side: u32,
    pub long_side: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedTransformEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub scale: bool,
    pub min_edge: bool,
    pub resolution: bool,
    pub max_scale: f32,
    pub is_diffusion: bool,
    pub supports_strength: bool,
    pub mac_only: bool,
    pub default_scale: f32,
    pub input_budget: Option<ExportedInputBudgetV2>,
    pub supports_color_fix: bool,
}

impl ExportedTransformEntry {
    pub fn from_descriptor(d: &TransformDescriptor) -> Self {
        let c = &d.capabilities;
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            scale: c.scale,
            min_edge: c.min_edge,
            resolution: c.resolution,
            max_scale: c.max_scale,
            is_diffusion: c.is_diffusion,
            supports_strength: c.supports_strength,
            mac_only: c.mac_only,
            default_scale: c.default_scale,
            input_budget: c.input_budget.map(|b| ExportedInputBudgetV2 {
                short_side: b.short_side,
                long_side: b.long_side,
            }),
            supports_color_fix: c.supports_color_fix,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedAudioTransformEntry {
    pub id: String,
    pub family: String,
    pub backend: String,
    pub kind: String,
    pub stem_count: u16,
    pub is_diffusion: bool,
    pub supports_strength: bool,
    pub supports_resample: bool,
    pub mac_only: bool,
}

impl ExportedAudioTransformEntry {
    pub fn from_descriptor(d: &AudioTransformDescriptor) -> Self {
        let c = &d.capabilities;
        let kind = match c.kind {
            AudioTransformKind::VoiceConversion => "voice_conversion",
            AudioTransformKind::StemSeparation => "stem_separation",
            AudioTransformKind::SuperResolution => "super_resolution",
        };
        Self {
            id: d.id.to_string(),
            family: d.family.to_string(),
            backend: d.backend.to_string(),
            kind: kind.to_string(),
            stem_count: c.stem_count,
            is_diffusion: c.is_diffusion,
            supports_strength: c.supports_strength,
            supports_resample: c.supports_resample,
            mac_only: c.mac_only,
        }
    }
}

/// Exported imported-model route projection.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExportedImportedModelEntry {
    pub family: String,
    pub source: String,
    pub operation: String,
    pub provider_id: String,
    pub required_components: Option<Vec<String>>,
    pub inherit_adapters: bool,
}

impl ExportedImportedModelEntry {
    pub fn from_registration(reg: &ImportedModelRegistration) -> Self {
        Self {
            family: reg.family.to_string(),
            source: imported_source_name(reg.source).to_string(),
            operation: imported_operation_name(reg.operation).to_string(),
            provider_id: reg.provider_id.to_string(),
            required_components: reg
                .required_components
                .map(|slice| slice.iter().map(|c| (*c).to_string()).collect()),
            inherit_adapters: reg.inherit_adapters,
        }
    }
}

/// Exported portable checkpoint adapter entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedCheckpointAdapterEntry {
    pub adapter_id: String,
    pub family: String,
    pub compatibility_family: String,
    pub eligible_backends: Vec<String>,
    pub dialects: Vec<ExportedCheckpointDialectV2>,
    pub operations: Vec<String>,
    pub backend_bindings: Vec<ExportedCheckpointBindingV2>,
    pub has_loader_reuse: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedCheckpointDialectV2 {
    pub id: String,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedCheckpointBindingV2 {
    pub backend: String,
    pub source: String,
    pub operation: String,
    pub provider_id: String,
    pub required_components: Option<Vec<String>>,
    pub inherit_adapters: bool,
}

impl ExportedCheckpointAdapterEntry {
    pub fn from_registration(reg: &CheckpointAdapterRegistration) -> Self {
        Self {
            adapter_id: reg.adapter_id.to_string(),
            family: reg.family.to_string(),
            compatibility_family: reg.compatibility_projection.family.to_string(),
            eligible_backends: reg
                .eligible_backends
                .iter()
                .copied()
                .map(checkpoint_backend_name)
                .map(str::to_string)
                .collect(),
            dialects: reg
                .dialects
                .iter()
                .map(|d| ExportedCheckpointDialectV2 {
                    id: d.id.to_string(),
                    source: imported_source_name(d.source).to_string(),
                })
                .collect(),
            operations: reg
                .operations
                .iter()
                .copied()
                .map(imported_operation_name)
                .map(str::to_string)
                .collect(),
            backend_bindings: reg
                .backend_bindings
                .iter()
                .map(|b| ExportedCheckpointBindingV2 {
                    backend: checkpoint_backend_name(b.backend).to_string(),
                    source: imported_source_name(b.source).to_string(),
                    operation: imported_operation_name(b.operation).to_string(),
                    provider_id: b.provider_id.to_string(),
                    required_components: b
                        .required_components
                        .map(|slice| slice.iter().map(|c| (*c).to_string()).collect()),
                    inherit_adapters: b.inherit_adapters,
                })
                .collect(),
            has_loader_reuse: reg.loader_reuse.is_some(),
        }
    }
}

/// Platform-owned bespoke utility crates, crate-API providers, and descriptor-less memory waivers.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedBespokeSurfaceV2 {
    /// Platform-owned utility crates consumed outside `ProviderRegistry::load` (e.g. `depth`,
    /// `face`, `instantid`, `pid`, `pulid`, `sam2`, `sam3`).
    pub utility_crates: Vec<String>,
    /// Crate-API providers not registered in `ProviderRegistry` (e.g. `sheetsage2`).
    pub crate_api_providers: Vec<String>,
    /// Explicit descriptor-less memory route waivers (e.g. Candle PuLID).
    pub memory_route_waivers: Vec<ExportedBespokeMemoryWaiverV2>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedBespokeMemoryWaiverV2 {
    pub provider_id: String,
    pub crate_name: String,
    pub owner: String,
    pub reason: String,
}

// -------------------------------------------------------------------------------------------------
// Pure Derivation & Name Helpers
// -------------------------------------------------------------------------------------------------

fn derive_conditioning_alignment(
    kinds: &[ConditioningKind],
    pass0: Pass0ConditioningRequirement,
) -> ConditioningAlignment {
    if matches!(
        pass0,
        Pass0ConditioningRequirement::RequiresControl | Pass0ConditioningRequirement::RequiresEdit
    ) {
        return ConditioningAlignment::GridAligned;
    }
    let has_spatial_grid_input = kinds.iter().any(|k| {
        matches!(
            k,
            ConditioningKind::Control
                | ConditioningKind::Depth
                | ConditioningKind::Mask
                | ConditioningKind::ControlClip
                | ConditioningKind::Keyframe
                | ConditioningKind::VideoClip
        )
    });
    if has_spatial_grid_input {
        ConditioningAlignment::GridAligned
    } else {
        ConditioningAlignment::GeometryFree
    }
}

fn derive_route_modes(
    modality: Modality,
    caps: &Capabilities,
    control_required: bool,
) -> Vec<String> {
    let mut modes = Vec::new();
    let mut push = |mode: &str| {
        if !modes.iter().any(|m| m == mode) {
            modes.push(mode.to_string());
        }
    };

    match modality {
        Modality::Image => {
            if !control_required {
                push("text_to_image");
            }
            if caps.accepts(ConditioningKind::Reference) {
                push("image_to_image");
            }
            if caps.accepts(ConditioningKind::ReferenceRgba) {
                push("reference_rgba");
            }
            if caps.accepts(ConditioningKind::MultiReference) {
                push("multi_reference");
            }
            if caps.accepts(ConditioningKind::ReduxRefs) {
                push("redux");
            }
            if caps.accepts(ConditioningKind::Mask) {
                push("inpaint");
            }
            if caps.accepts(ConditioningKind::Control) {
                push("control");
            }
            if caps.accepts(ConditioningKind::Depth) {
                push("depth");
            }
        }
        Modality::Video => {
            push("text_to_video");
            if caps.accepts(ConditioningKind::Reference) || caps.accepts(ConditioningKind::Keyframe)
            {
                push("image_to_video");
            }
            if caps.accepts(ConditioningKind::VideoClip) {
                push("video_to_video");
            }
            if caps.accepts(ConditioningKind::ControlClip)
                || caps.accepts(ConditioningKind::Control)
            {
                push("control_video");
            }
            if caps.accepts(ConditioningKind::ReferenceAudio) {
                push("audio_to_video");
            }
            if caps.accepts(ConditioningKind::Mask) {
                push("inpaint_video");
            }
        }
        Modality::Both => {
            if !control_required {
                push("text_to_image");
                push("text_to_video");
            }
            if caps.accepts(ConditioningKind::Reference) || caps.accepts(ConditioningKind::Keyframe)
            {
                push("image_to_image");
                push("image_to_video");
            }
            if caps.accepts(ConditioningKind::VideoClip) {
                push("video_to_video");
            }
            if caps.accepts(ConditioningKind::Control)
                || caps.accepts(ConditioningKind::ControlClip)
            {
                push("control");
            }
        }
        Modality::Audio => {
            push("text_to_audio");
            if caps.accepts(ConditioningKind::VideoSync) {
                push("video_to_audio");
            }
            if caps.accepts(ConditioningKind::ReferenceAudio)
                || caps.accepts(ConditioningKind::VoiceEmbedding)
            {
                push("voice_clone");
            }
            if caps.accepts(ConditioningKind::AudioEdit)
                || caps.accepts(ConditioningKind::AudioEditRegions)
                || !caps.audio_edit_modes.is_empty()
            {
                push("audio_edit");
            }
            if caps.supports_multi_speaker {
                push("dialogue");
            }
            if caps.supports_conversation_history || caps.supports_conversation_session {
                push("conversation");
            }
            if caps.supports_symbolic_song {
                push("song");
            }
            if caps.supports_song_cover {
                push("song_cover");
            }
        }
    }

    modes
}

fn derive_alternate_decoders(
    backend: &str,
    route_id: &str,
    output_space: Option<&'static LatentSpace>,
) -> Vec<String> {
    let mut ids: Vec<String> = DECODER_OPTIONS
        .iter()
        .filter(|option| {
            option.eligible_backends.contains(&backend)
                && option.eligible_provider_ids.contains(&route_id)
                && output_space.is_none_or(|space| {
                    latent_spaces_compatible(Some(space), Some(option.input_latent_space))
                })
        })
        .map(|option| option.id.to_string())
        .collect();
    ids.sort();
    ids.dedup();
    ids
}

pub fn modality_name(modality: Modality) -> &'static str {
    match modality {
        Modality::Image => "image",
        Modality::Video => "video",
        Modality::Both => "both",
        Modality::Audio => "audio",
    }
}

pub fn quant_name(quant: Quant) -> &'static str {
    match quant {
        Quant::Q4 => "q4",
        Quant::Q8 => "q8",
        Quant::Nvfp4 => "nvfp4",
        Quant::Fp8Rowwise => "fp8r",
    }
}

pub fn conditioning_name(kind: ConditioningKind) -> &'static str {
    match kind {
        ConditioningKind::Reference => "reference",
        ConditioningKind::ReferenceRgba => "referenceRgba",
        ConditioningKind::ReferenceAudio => "referenceAudio",
        ConditioningKind::ReferenceVideo => "referenceVideo",
        ConditioningKind::AudioEdit => "audioEdit",
        ConditioningKind::AudioEditRegions => "audioEditRegions",
        ConditioningKind::VoiceEmbedding => "voiceEmbedding",
        ConditioningKind::MultiReference => "multiReference",
        ConditioningKind::ReduxRefs => "reduxRefs",
        ConditioningKind::Control => "control",
        ConditioningKind::Depth => "depth",
        ConditioningKind::Mask => "mask",
        ConditioningKind::Keyframe => "keyframe",
        ConditioningKind::VideoClip => "videoClip",
        ConditioningKind::ControlClip => "controlClip",
        ConditioningKind::VideoSync => "videoSync",
        ConditioningKind::ConversationHistory => "conversationHistory",
    }
}

fn control_kind_name(kind: &ControlKind) -> String {
    match kind {
        ControlKind::Pose => "pose".into(),
        ControlKind::Canny => "canny".into(),
        ControlKind::Depth => "depth".into(),
        ControlKind::Other(other) => other.clone(),
    }
}

fn pass0_requirement_name(req: Pass0ConditioningRequirement) -> &'static str {
    match req {
        Pass0ConditioningRequirement::NoiseStartSupported => "noise_start_supported",
        Pass0ConditioningRequirement::RequiresReference => "requires_reference",
        Pass0ConditioningRequirement::RequiresControl => "requires_control",
        Pass0ConditioningRequirement::RequiresEdit => "requires_edit",
    }
}

fn adapter_install_policy_name(policy: AdapterInstallPolicy) -> &'static str {
    match policy {
        AdapterInstallPolicy::Unsupported => "unsupported",
        AdapterInstallPolicy::ForwardResidual => "forward_residual",
        AdapterInstallPolicy::FusedMerge => "fused_merge",
        AdapterInstallPolicy::BaseDenoiserOnly => "base_denoiser_only",
    }
}

fn preview_support_name(preview: PreviewSupport) -> &'static str {
    match preview {
        PreviewSupport::Unsupported => "unsupported",
        PreviewSupport::LatentProjection => "latent_projection",
        PreviewSupport::PixelSpace => "pixel_space",
    }
}

fn render_plan_scope_name(scope: RenderPlanScope) -> &'static str {
    match scope {
        RenderPlanScope::SingleAndMultiPass => "single_and_multi_pass",
        RenderPlanScope::MultiphaseOnly => "multiphase_only",
    }
}

fn staged_residency_name(avail: StagedResidencyAvailability) -> &'static str {
    match avail {
        StagedResidencyAvailability::Absent => "absent",
        StagedResidencyAvailability::Selectable => "selectable",
        StagedResidencyAvailability::UnconditionallyEngaged => "unconditionally_engaged",
    }
}

fn memory_strategy_name(strategy: MemoryStrategy) -> &'static str {
    match strategy {
        MemoryStrategy::Resident => "resident",
        MemoryStrategy::StagedResidency => "staged_residency",
        MemoryStrategy::BoundedDecode => "bounded_decode",
        MemoryStrategy::BoundedAttention => "bounded_attention",
        MemoryStrategy::BoundedTransformerResidency => "bounded_transformer_residency",
    }
}

fn weight_source_kind_name(kind: WeightSourceKind) -> &'static str {
    match kind {
        WeightSourceKind::SnapshotDirectory => "snapshot_directory",
        WeightSourceKind::SingleFile => "single_file",
        WeightSourceKind::DirectoryOrFile => "directory_or_file",
    }
}

fn seed_scheme_name(scheme: SeedScheme) -> &'static str {
    match scheme {
        SeedScheme::DeterministicOffset => "deterministic_offset",
        SeedScheme::BackendGlobalSeeded => "backend_global_seeded",
        SeedScheme::DeterministicSynthesis => "deterministic_synthesis",
    }
}

fn seed_index_relation(scheme: SeedScheme) -> &'static str {
    match scheme {
        SeedScheme::DeterministicOffset => "base_seed_plus_index",
        SeedScheme::BackendGlobalSeeded => "sequential_backend_rng",
        SeedScheme::DeterministicSynthesis => "deterministic_no_latent_noise",
    }
}

fn audio_edit_mode_name(mode: AudioEditMode) -> &'static str {
    match mode {
        AudioEditMode::Inpaint => "inpaint",
        AudioEditMode::Repaint => "repaint",
        AudioEditMode::Extend => "extend",
        AudioEditMode::Cover => "cover",
    }
}

fn imported_source_name(source: ImportedModelSource) -> &'static str {
    match source {
        ImportedModelSource::TransformerFile => "transformer_file",
        ImportedModelSource::FusedCheckpoint => "fused_checkpoint",
        ImportedModelSource::TransformerDirectory => "transformer_directory",
        ImportedModelSource::ComfyUiTree => "comfyui_tree",
    }
}

fn imported_operation_name(op: ImportedModelOperation) -> &'static str {
    match op {
        ImportedModelOperation::Generate => "generate",
        ImportedModelOperation::Edit => "edit",
        ImportedModelOperation::Pose => "pose",
        ImportedModelOperation::MultiPhase => "multi_phase",
    }
}

fn checkpoint_backend_name(backend: CheckpointBackend) -> &'static str {
    match backend {
        CheckpointBackend::Mlx => "mlx",
        CheckpointBackend::Candle => "candle",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latent::FLUX1_LATENT_SPACE;
    use crate::route::{
        AdapterSurface, ConditioningSurface, FamilyModelSpec, LatentCodecDeclaration,
        OutputSurface, ResidencySurface, RouteBindings, RouteStructure, SamplingSurface,
        TextEncoderDeclaration, TextEncoderSurface, WeightPlanDeclaration,
    };

    fn dummy_descriptor() -> ModelDescriptor {
        sample_low_res_route().compute_descriptor()
    }

    fn dummy_load(_: &crate::LoadSpec) -> Result<Box<dyn crate::Generator>> {
        Err(Error::Msg("unused".into()))
    }

    fn dummy_plan(
        _: &crate::LoadSpec,
        _: &crate::GenerationRequest,
    ) -> Result<Option<crate::RenderPlan>> {
        Ok(None)
    }

    fn sample_low_res_route() -> RouteRecord {
        RouteRecord {
            route_id: "sample_flux",
            family: "flux",
            backend: "candle",
            modality: Modality::Image,
            mac_only: false,
            structure: RouteStructure {
                text_encoders: TextEncoderSurface {
                    encoders: vec![TextEncoderDeclaration::new("clip_l", Some(77))],
                    ..Default::default()
                },
                conditioning: ConditioningSurface {
                    kinds: vec![ConditioningKind::Reference],
                    pass0_requirement: Pass0ConditioningRequirement::NoiseStartSupported,
                    supports_negative_prompt: false,
                    ..Default::default()
                },
                sampling: SamplingSurface {
                    samplers: vec!["euler"],
                    schedulers: vec!["linear"],
                    supports_guidance: true,
                    ..Default::default()
                },
                denoiser: ModelBinding::family_model(
                    FamilyModelSpec::new("flux1_mmdit").with_low_res_start(),
                ),
                latent_codec: LatentCodecDeclaration {
                    codec_id: Some("flux1_ae"),
                    output_latent_space: Some(&FLUX1_LATENT_SPACE),
                    pixel_stride: Some(16),
                    token_width: Some(64),
                    supports_diffusion_decoder: false,
                },
                adapters: AdapterSurface::lora_and_lokr(AdapterInstallPolicy::ForwardResidual),
                residency: ResidencySurface {
                    supports_sequential_offload: true,
                    ..Default::default()
                },
                weight_plan: WeightPlanDeclaration {
                    source_kind: WeightSourceKind::SnapshotDirectory,
                    supported_quants: &[Quant::Q4, Quant::Q8],
                    ..Default::default()
                },
                output: OutputSurface {
                    preview: PreviewSupport::LatentProjection,
                    supports_hdr: false,
                    supports_alpha_output: false,
                },
                ..Default::default()
            },
            limits: ProductLimits {
                min_size: 256,
                max_size: 2048,
                size_floor: SizeFloor::RangeCheckedOnGrid { multiple: 16 },
                supported_steps: StepSupport::Unconstrained,
                max_audio_duration_secs: None,
            },
            seed_and_batch: SeedAndBatchSemantics {
                seed_scheme: SeedScheme::DeterministicOffset,
                max_count: 4,
                supports_prompt_batch: false,
            },
            defaults: RouteDefaults {
                steps: Some(25),
                guidance: Some(3.5),
                sampler: Some("euler"),
                schedule: Some("linear"),
                dimensions: Some((1024, 1024)),
                ..Default::default()
            },
            bindings: RouteBindings::generator(dummy_descriptor, dummy_load)
                .with_render_plan(dummy_plan),
        }
    }

    fn sample_export(route: ExportedRouteV2) -> CapabilitiesExportV2 {
        let mut routes = BTreeMap::new();
        routes.insert(route.route_id.clone(), route);
        CapabilitiesExportV2 {
            schema_version: CAPABILITIES_EXPORT_SCHEMA_VERSION.to_string(),
            backend: "candle".into(),
            platform: "cpu".into(),
            feature_set: FEATURE_SET_CANDLE_CPU.into(),
            audio_backend: None,
            advanced_quant_tiers: Vec::new(),
            rejected_quants: BTreeMap::new(),
            snapshot_preparer_backends: vec!["candle".into()],
            audio_snapshot_preparer_backends: Vec::new(),
            routes,
            audio_routes: BTreeMap::new(),
            feature_routes: BTreeMap::new(),
            llm_models: BTreeMap::new(),
            trainers: BTreeMap::new(),
            captioners: BTreeMap::new(),
            transcribers: BTreeMap::new(),
            embedders: ExportedEmbeddersV2::default(),
            transforms: ExportedTransformsV2::default(),
            imported_models: Vec::new(),
            checkpoint_adapters: BTreeMap::new(),
            bespoke_waivers: ExportedBespokeSurfaceV2::default(),
        }
    }

    #[test]
    fn canonical_json_is_deterministic_roundtrips_and_forbids_pin_metadata() {
        let export = sample_export(ExportedRouteV2::from_route_record(
            &sample_low_res_route(),
            None,
            false,
        ));
        let bytes_a = export.to_canonical_json_bytes().unwrap();
        let bytes_b = export.to_canonical_json_bytes().unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "canonical bytes must be byte-deterministic"
        );
        assert!(canonical_bytes_equal(&bytes_a, &bytes_b).unwrap());

        let reparsed = CapabilitiesExportV2::from_json_bytes(&bytes_a).unwrap();
        assert!(semantic_equal(&export, &reparsed));
        assert_eq!(
            export.semantic_digest_hex().unwrap(),
            reparsed.semantic_digest_hex().unwrap()
        );

        let text = String::from_utf8(bytes_a).unwrap();
        for forbidden in [
            "git_sha",
            "commit_sha",
            "timestamp",
            "generated_at",
            "pin_hash",
        ] {
            assert!(
                !text.contains(forbidden),
                "canonical export bytes must not contain {forbidden} (M8)"
            );
        }
    }

    #[test]
    fn hook4_removing_structural_precondition_flips_exported_feature_applicability() {
        let mut route = sample_low_res_route();
        let before = ExportedRouteV2::from_route_record(&route, None, false);
        let before_lrs = before
            .features
            .iter()
            .find(|f| f.feature_id == "low_res_start")
            .expect("low_res_start present in export");
        assert!(
            before_lrs.applicable,
            "route with all structural preconditions must export low_res_start applicable=true"
        );
        assert!(before_lrs.reasons.is_empty());

        // Remove ONE structural precondition (geometry change on denoiser) with zero table edits.
        route.structure.denoiser = ModelBinding::family_model(FamilyModelSpec::new("flux1_mmdit"));
        let after = ExportedRouteV2::from_route_record(&route, None, false);
        let after_lrs = after
            .features
            .iter()
            .find(|f| f.feature_id == "low_res_start")
            .expect("low_res_start present in export");
        assert!(
            !after_lrs.applicable,
            "removing denoiser geometry_change precondition must flip exported low_res_start to false"
        );
        assert!(
            !after_lrs.reasons.is_empty(),
            "non-applicable exported Feature must carry typed refusal reasons"
        );
    }

    #[test]
    fn verify_runtime_against_export_refuses_disagreements_with_typed_reasons() {
        let route_a = ExportedRouteV2::from_route_record(&sample_low_res_route(), None, false);
        let export_a = sample_export(route_a.clone());
        verify_runtime_against_export(&export_a, &export_a)
            .expect("matching runtime and export must succeed");

        // 1. Missing route in runtime
        let mut missing_runtime = export_a.clone();
        missing_runtime.routes.clear();
        let err = verify_runtime_against_export(&missing_runtime, &export_a).unwrap_err();
        assert!(err
            .disagreements
            .iter()
            .any(|d| d.kind == ExportDisagreementKind::MissingRouteInRuntime));

        // 2. Changed capability/limit on route
        let mut changed_limit = export_a.clone();
        changed_limit
            .routes
            .get_mut("sample_flux")
            .unwrap()
            .limits
            .max_size = 1024;
        let err = verify_runtime_against_export(&changed_limit, &export_a).unwrap_err();
        assert!(err
            .disagreements
            .iter()
            .any(|d| d.kind == ExportDisagreementKind::RouteDefinitionMismatch));

        // 3. Changed HOOK-4 feature applicability
        let mut changed_route = sample_low_res_route();
        changed_route.structure.denoiser =
            ModelBinding::family_model(FamilyModelSpec::new("flux1_mmdit"));
        let export_b = sample_export(ExportedRouteV2::from_route_record(
            &changed_route,
            None,
            false,
        ));
        let err = verify_runtime_against_export(&export_b, &export_a).unwrap_err();
        assert!(err
            .disagreements
            .iter()
            .any(|d| d.kind == ExportDisagreementKind::RouteFeatureApplicabilityMismatch));
    }

    #[test]
    fn conditioning_alignment_distinguishes_grid_aligned_and_geometry_free() {
        let mut route = sample_low_res_route();
        let free = ExportedRouteV2::from_route_record(&route, None, false);
        assert_eq!(
            free.conditioning.aligned,
            ConditioningAlignment::GeometryFree
        );

        route
            .structure
            .conditioning
            .kinds
            .push(ConditioningKind::Control);
        let aligned = ExportedRouteV2::from_route_record(&route, None, false);
        assert_eq!(
            aligned.conditioning.aligned,
            ConditioningAlignment::GridAligned
        );
    }
}

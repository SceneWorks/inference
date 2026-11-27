//! Hardware-free `sga.capabilities/2` export assembly from a validated [`RuntimeCatalog`] and
//! shared CLI harness for all platform export binaries (`SW-6980`: `REG-3`, `HOOK-4`, `M8`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use gen_core::export::{
    canonical_bytes_equal, modality_name, quant_name, semantic_equal,
    verify_runtime_against_export, CapabilitiesExportV2, ExportDisagreementError,
    ExportedAudioEmbedderEntry, ExportedAudioTransformEntry, ExportedBespokeMemoryWaiverV2,
    ExportedBespokeSurfaceV2, ExportedCapabilitiesV2, ExportedCaptionerEntry,
    ExportedCheckpointAdapterEntry, ExportedEmbeddersV2, ExportedFeatureApplicabilityV2,
    ExportedFeatureRouteV2, ExportedImageEmbedderEntry, ExportedImportedModelEntry,
    ExportedLlmEntry, ExportedRouteV2, ExportedTextEmbedderEntry, ExportedTrainerEntry,
    ExportedTranscriberEntry, ExportedTransformEntry, ExportedTransformsV2,
    ExportedVoiceEmbedderEntry, CAPABILITIES_EXPORT_SCHEMA_VERSION,
};
use gen_core::{FeatureRegistry, FeatureRouteStructure, ProviderRegistry, Quant};

use crate::RuntimeCatalog;

/// Platform-specific metadata supplied by a runtime bundle when assembling a
/// [`CapabilitiesExportV2`] from its validated [`RuntimeCatalog`].
#[derive(Clone, Debug, Default)]
pub struct ExportAssemblyOptions {
    /// Shipping feature-set label (e.g. `"macos-mlx"`, `"ios-media"`, `"ios-llm-only"`,
    /// `"candle-cuda"`, `"candle-cpu"`).
    pub feature_set: String,
    /// Advanced quantization tiers surfaced beyond standard Q4/Q8 (e.g. `[Quant::Nvfp4]` on
    /// `candle-cuda`).
    pub advanced_quant_tiers: Vec<Quant>,
    /// Route IDs whose control input is mandatory on every request (e.g. `z_image_turbo_control`
    /// on `ios-media`).
    pub control_required_routes: Vec<String>,
    /// Platform-owned utility crates consumed outside `ProviderRegistry::load`.
    pub bespoke_utility_crates: Vec<String>,
    /// Crate-API providers exposed directly by the bundle outside `ProviderRegistry`.
    pub crate_api_providers: Vec<String>,
    /// Explicit descriptor-less memory route waivers.
    pub memory_route_waivers: Vec<ExportedBespokeMemoryWaiverV2>,
}

impl ExportAssemblyOptions {
    pub fn new(feature_set: impl Into<String>) -> Self {
        Self {
            feature_set: feature_set.into(),
            ..Default::default()
        }
    }
}

/// Assemble a canonical [`CapabilitiesExportV2`] purely from a validated [`RuntimeCatalog`]
/// without loading weights or touching GPU/host runtime state (`REG-3`, `M8`).
pub fn assemble_capabilities_export(
    catalog: &RuntimeCatalog,
    options: &ExportAssemblyOptions,
) -> CapabilitiesExportV2 {
    let control_required: BTreeSet<&str> = options
        .control_required_routes
        .iter()
        .map(String::as_str)
        .collect();

    let routes = export_routes_from_registry(catalog.media(), &control_required);
    let audio_routes = catalog
        .audio()
        .map(|audio| export_routes_from_registry(audio, &control_required))
        .unwrap_or_default();

    let feature_routes = export_feature_routes(catalog.media());

    let mut llm_models = BTreeMap::new();
    for reg in catalog.text().registrations() {
        let descriptor = (reg.descriptor)();
        llm_models.insert(
            descriptor.id.clone(),
            ExportedLlmEntry::from_descriptor(&descriptor),
        );
    }

    let mut trainers = BTreeMap::new();
    for reg in catalog.media().trainers() {
        let descriptor = (reg.descriptor)();
        trainers.insert(
            descriptor.id.to_string(),
            ExportedTrainerEntry::from_descriptor(&descriptor),
        );
    }

    let mut captioners = BTreeMap::new();
    for reg in catalog.media().captioners() {
        let descriptor = (reg.descriptor)();
        captioners.insert(
            descriptor.id.to_string(),
            ExportedCaptionerEntry::from_descriptor(&descriptor),
        );
    }

    let mut transcribers = BTreeMap::new();
    if let Some(audio) = catalog.audio() {
        for reg in audio.transcribers() {
            let descriptor = (reg.descriptor)();
            transcribers.insert(
                descriptor.id.to_string(),
                ExportedTranscriberEntry::from_descriptor(&descriptor),
            );
        }
    }

    let mut embedders = ExportedEmbeddersV2::default();
    for reg in catalog.media().image_embedders() {
        let descriptor = (reg.descriptor)();
        embedders.image.insert(
            descriptor.id.to_string(),
            ExportedImageEmbedderEntry::from_descriptor(&descriptor),
        );
    }
    for reg in catalog.media().text_embedders() {
        let descriptor = (reg.descriptor)();
        embedders.text.insert(
            descriptor.id.to_string(),
            ExportedTextEmbedderEntry::from_descriptor(&descriptor),
        );
    }
    if let Some(audio) = catalog.audio() {
        for reg in audio.voice_embedders() {
            let descriptor = (reg.descriptor)();
            embedders.voice.insert(
                descriptor.id.to_string(),
                ExportedVoiceEmbedderEntry::from_descriptor(&descriptor),
            );
        }
        for reg in audio.audio_embedders() {
            let descriptor = (reg.descriptor)();
            embedders.audio.insert(
                descriptor.id.to_string(),
                ExportedAudioEmbedderEntry::from_descriptor(&descriptor),
            );
        }
    }

    let mut transforms = ExportedTransformsV2::default();
    for reg in catalog.media().transforms() {
        let descriptor = (reg.descriptor)();
        transforms.image.insert(
            descriptor.id.to_string(),
            ExportedTransformEntry::from_descriptor(&descriptor),
        );
    }
    if let Some(audio) = catalog.audio() {
        for reg in audio.audio_transforms() {
            let descriptor = (reg.descriptor)();
            transforms.audio.insert(
                descriptor.id.to_string(),
                ExportedAudioTransformEntry::from_descriptor(&descriptor),
            );
        }
    }

    let mut imported_models: Vec<ExportedImportedModelEntry> = catalog
        .media()
        .imported_models()
        .map(ExportedImportedModelEntry::from_registration)
        .collect();
    imported_models.sort();

    let mut checkpoint_adapters = BTreeMap::new();
    for adapter in catalog.media().checkpoint_adapters() {
        checkpoint_adapters.insert(
            adapter.adapter_id.to_string(),
            ExportedCheckpointAdapterEntry::from_registration(adapter),
        );
    }

    let mut advanced_quant_tiers: Vec<String> = options
        .advanced_quant_tiers
        .iter()
        .copied()
        .map(quant_name)
        .map(str::to_string)
        .collect();
    advanced_quant_tiers.sort();
    advanced_quant_tiers.dedup();

    let mut rejected_quants = BTreeMap::new();
    for &(quant, reason) in catalog.media().rejected_quants() {
        rejected_quants.insert(quant_name(quant).to_string(), reason.to_string());
    }

    let snapshot_preparer_backends: Vec<String> = catalog
        .preparers()
        .registrations()
        .map(|reg| (reg.backend)().to_string())
        .collect();

    let audio_snapshot_preparer_backends: Vec<String> = catalog
        .audio_preparers()
        .map(|prep| {
            prep.registrations()
                .map(|reg| (reg.backend)().to_string())
                .collect()
        })
        .unwrap_or_default();

    let mut utility_crates = options.bespoke_utility_crates.clone();
    utility_crates.sort();
    utility_crates.dedup();

    let mut crate_api_providers = options.crate_api_providers.clone();
    crate_api_providers.sort();
    crate_api_providers.dedup();

    let mut memory_route_waivers = options.memory_route_waivers.clone();
    memory_route_waivers.sort_by(|a, b| a.provider_id.cmp(&b.provider_id));

    CapabilitiesExportV2 {
        schema_version: CAPABILITIES_EXPORT_SCHEMA_VERSION.to_string(),
        backend: catalog.backend().to_string(),
        platform: catalog.platform().to_string(),
        feature_set: options.feature_set.clone(),
        audio_backend: catalog.audio_backend().map(str::to_string),
        advanced_quant_tiers,
        rejected_quants,
        snapshot_preparer_backends,
        audio_snapshot_preparer_backends,
        routes,
        audio_routes,
        feature_routes,
        llm_models,
        trainers,
        captioners,
        transcribers,
        embedders,
        transforms,
        imported_models,
        checkpoint_adapters,
        bespoke_waivers: ExportedBespokeSurfaceV2 {
            utility_crates,
            crate_api_providers,
            memory_route_waivers,
        },
    }
}

fn export_routes_from_registry(
    registry: &ProviderRegistry,
    control_required: &BTreeSet<&str>,
) -> BTreeMap<String, ExportedRouteV2> {
    let mut routes = BTreeMap::new();
    for descriptor in registry.generator_descriptors() {
        let is_control_required = control_required.contains(descriptor.id);
        let exported = if let Some(record) = registry.route_record(descriptor.id) {
            ExportedRouteV2::from_route_record(record, Some(registry), is_control_required)
        } else {
            ExportedRouteV2::from_descriptor(descriptor, Some(registry), is_control_required)
        };
        routes.insert(descriptor.id.to_string(), exported);
    }
    routes
}

fn export_feature_routes(registry: &ProviderRegistry) -> BTreeMap<String, ExportedFeatureRouteV2> {
    let mut out = BTreeMap::new();
    let standard = FeatureRegistry::standard();
    for fr in registry.feature_route_registrations() {
        let mut descriptor = (fr.descriptor)();
        descriptor.id = fr.route_id;
        let route_struct = if let Some(record) = registry.route_record(fr.route_id) {
            FeatureRouteStructure::from_route_record(record)
        } else {
            FeatureRouteStructure::from_descriptor_and_plan(&descriptor, fr.render_plan.as_ref())
        };
        let features: Vec<ExportedFeatureApplicabilityV2> = standard
            .evaluate_route(&route_struct)
            .values()
            .map(ExportedFeatureApplicabilityV2::from_feature_applicability)
            .collect();
        out.insert(
            fr.route_id.to_string(),
            ExportedFeatureRouteV2 {
                route_id: fr.route_id.to_string(),
                parent_provider_id: fr.parent_provider_id.map(str::to_string),
                family: descriptor.family.to_string(),
                backend: descriptor.backend.to_string(),
                modality: modality_name(descriptor.modality).to_string(),
                has_render_plan: fr.render_plan.is_some(),
                features,
                capabilities: ExportedCapabilitiesV2::from_capabilities(&descriptor.capabilities),
            },
        );
    }
    out
}

/// Verify a live linked [`RuntimeCatalog`] against a deserialized [`CapabilitiesExportV2`]
/// document, returning a typed [`ExportDisagreementError`] on any mismatch (`REG-3`).
pub fn verify_catalog_against_export(
    catalog: &RuntimeCatalog,
    expected: &CapabilitiesExportV2,
    options: &ExportAssemblyOptions,
) -> std::result::Result<(), ExportDisagreementError> {
    let actual = assemble_capabilities_export(catalog, options);
    verify_runtime_against_export(&actual, expected)
}

// -------------------------------------------------------------------------------------------------
// Shared Export CLI Harness (used by runtime-cpu, runtime-macos, runtime-ios binaries)
// -------------------------------------------------------------------------------------------------

/// One named shipping feature-set target exposed by an export binary.
pub struct ExportCliTarget {
    pub feature_set: &'static str,
    pub build: fn() -> crate::Result<CapabilitiesExportV2>,
}

/// Run the shared `sga.capabilities/2` descriptor-only export CLI.
///
/// Supported flags:
/// - `--list`: list the shipping feature sets compiled into this binary.
/// - `--feature-set <name>`: select one shipping feature set (default when only one target exists).
/// - `--out <path>`: write canonical `sga.capabilities/2` JSON to `<path>` (defaults to stdout).
/// - `--all --out-dir <dir>`: write `<dir>/<feature_set>.json` for every target in this binary.
/// - `--check <path>`: fail if the generated canonical export differs semantically from `<path>`.
/// - `--self-check`: verify byte-for-byte determinism, JSON round-trip, and runtime parity for
///   every target compiled into this binary without writing any files.
pub fn run_export_cli(
    tool_name: &str,
    targets: &[ExportCliTarget],
) -> std::result::Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    run_export_cli_with_args(tool_name, targets, &args)
}

pub fn run_export_cli_with_args(
    tool_name: &str,
    targets: &[ExportCliTarget],
    args: &[String],
) -> std::result::Result<(), String> {
    let mut selected_feature_set: Option<String> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut out_dir: Option<PathBuf> = None;
    let mut check_path: Option<PathBuf> = None;
    let mut all = false;
    let mut list = false;
    let mut self_check = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: {tool_name} [--list] [--self-check] [--all --out-dir <DIR>] \
                     [--feature-set <NAME>] [--out <FILE> | --check <FILE>]"
                );
                return Ok(());
            }
            "--list" => list = true,
            "--all" => all = true,
            "--self-check" => self_check = true,
            "--feature-set" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--feature-set requires a value".to_string())?;
                selected_feature_set = Some(value.clone());
            }
            "--out" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--out requires a path".to_string())?;
                out_path = Some(PathBuf::from(value));
            }
            "--out-dir" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--out-dir requires a directory".to_string())?;
                out_dir = Some(PathBuf::from(value));
            }
            "--check" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| "--check requires an existing JSON file path".to_string())?;
                check_path = Some(PathBuf::from(value));
            }
            other => {
                return Err(format!(
                    "{tool_name}: unknown argument {other:?} (try --help)"
                ));
            }
        }
        i += 1;
    }

    if list {
        for target in targets {
            println!("{}", target.feature_set);
        }
        return Ok(());
    }

    if self_check {
        let to_check: Vec<&ExportCliTarget> = if let Some(name) = &selected_feature_set {
            vec![find_target(tool_name, targets, name)?]
        } else {
            targets.iter().collect()
        };
        for target in to_check {
            let first = (target.build)()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let second = (target.build)()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let bytes_a = first
                .to_canonical_json_bytes()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let bytes_b = second
                .to_canonical_json_bytes()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            if bytes_a != bytes_b {
                return Err(format!(
                    "{tool_name} ({}): canonical export is not byte-deterministic across repeated builds",
                    target.feature_set
                ));
            }
            let parsed = CapabilitiesExportV2::from_json_bytes(&bytes_a)
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            if !semantic_equal(&first, &parsed) {
                return Err(format!(
                    "{tool_name} ({}): round-tripped JSON is not semantically equal",
                    target.feature_set
                ));
            }
            verify_runtime_against_export(&first, &parsed)
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let digest = first
                .semantic_digest_hex()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            println!(
                "ok: {} (platform={}, backend={}, routes={}, audio_routes={}, llm_models={}, sha256={})",
                first.feature_set,
                first.platform,
                first.backend,
                first.routes.len(),
                first.audio_routes.len(),
                first.llm_models.len(),
                &digest[..16],
            );
        }
        return Ok(());
    }

    if all {
        let dir = out_dir.ok_or_else(|| format!("{tool_name}: --all requires --out-dir <DIR>"))?;
        std::fs::create_dir_all(&dir)
            .map_err(|err| format!("{tool_name}: failed to create {}: {err}", dir.display()))?;
        for target in targets {
            let export = (target.build)()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let bytes = export
                .to_canonical_json_bytes()
                .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
            let path = dir.join(format!("{}.json", target.feature_set));
            write_bytes(&path, &bytes)?;
        }
        return Ok(());
    }

    let target = match selected_feature_set {
        Some(ref name) => find_target(tool_name, targets, name)?,
        None if targets.len() == 1 => &targets[0],
        None => {
            let available: Vec<&str> = targets.iter().map(|t| t.feature_set).collect();
            return Err(format!(
                "{tool_name}: pass --feature-set <NAME> (available: {}) or --all --out-dir <DIR>",
                available.join(", ")
            ));
        }
    };

    let export =
        (target.build)().map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;
    let bytes = export
        .to_canonical_json_bytes()
        .map_err(|err| format!("{tool_name} ({}): {err}", target.feature_set))?;

    if let Some(existing_path) = check_path {
        let existing = std::fs::read(&existing_path).map_err(|err| {
            format!(
                "{tool_name}: failed to read {}: {err}",
                existing_path.display()
            )
        })?;
        let equal = canonical_bytes_equal(&bytes, &existing)
            .map_err(|err| format!("{tool_name}: {err}"))?;
        if !equal {
            let parsed_existing = CapabilitiesExportV2::from_json_bytes(&existing)
                .map_err(|err| format!("{tool_name}: {err}"))?;
            if let Err(disagreement) = verify_runtime_against_export(&export, &parsed_existing) {
                return Err(format!(
                    "{tool_name}: {} differs from {}: {disagreement}",
                    target.feature_set,
                    existing_path.display()
                ));
            }
            return Err(format!(
                "{tool_name}: {} canonical bytes differ from {}",
                target.feature_set,
                existing_path.display()
            ));
        }
        return Ok(());
    }

    if let Some(path) = out_path {
        write_bytes(&path, &bytes)?;
    } else if let Some(dir) = out_dir {
        std::fs::create_dir_all(&dir)
            .map_err(|err| format!("{tool_name}: failed to create {}: {err}", dir.display()))?;
        let path = dir.join(format!("{}.json", target.feature_set));
        write_bytes(&path, &bytes)?;
    } else {
        std::io::stdout()
            .write_all(&bytes)
            .map_err(|err| format!("{tool_name}: failed to write stdout: {err}"))?;
    }
    Ok(())
}

fn find_target<'a>(
    tool_name: &str,
    targets: &'a [ExportCliTarget],
    name: &str,
) -> std::result::Result<&'a ExportCliTarget, String> {
    targets
        .iter()
        .find(|t| t.feature_set == name)
        .ok_or_else(|| {
            let available: Vec<&str> = targets.iter().map(|t| t.feature_set).collect();
            format!(
                "{tool_name}: feature set {name:?} is not built into this binary (available: {})",
                available.join(", ")
            )
        })
}

fn write_bytes(path: &Path, bytes: &[u8]) -> std::result::Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "failed to create parent directory {}: {err}",
                    parent.display()
                )
            })?;
        }
    }
    std::fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

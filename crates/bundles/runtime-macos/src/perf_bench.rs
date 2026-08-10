//! Versioned contracts for the cross-family MLX performance gate (sc-18321).
//!
//! A campaign is frozen before any child starts: source/build provenance, host identity, the full
//! workload matrix, selected variants, provider capability declarations, exact artifact revisions,
//! and deterministic content inventories all contribute to one campaign id. Children consume only
//! that frozen envelope. Run and summary validation is deliberately fail-closed so legacy, stale,
//! partial, mixed-host, or silently-fallback evidence cannot be promoted into an acceptance result.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub const MATRIX_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-matrix.v2";
pub const ARTIFACT_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-artifacts.v2";
pub const CAMPAIGN_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-campaign.v1";
pub const RUN_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-run.v2";
pub const SUMMARY_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-summary.v2";
pub const INVENTORY_ALGORITHM: &str = "sha256-tree-content-v1";
pub const MEMORY_SAMPLE_INTERVAL_MICROS: u64 = 50_000;
pub const MEMORY_MAX_GAP_MULTIPLIER: u64 = 8;
pub const MEMORY_BINDING_RULE: &str = "median_peak_same_sample_active_plus_cache";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkFamily {
    WanVideo,
    ImageDit,
    SdxlUnet,
}

impl BenchmarkFamily {
    pub const ALL: [Self; 3] = [Self::WanVideo, Self::ImageDit, Self::SdxlUnet];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WanVideo => "wan_video",
            Self::ImageDit => "image_dit",
            Self::SdxlUnet => "sdxl_unet",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    Bf16,
    Q4,
    Q8,
}

impl ModelTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Q4 => "q4",
            Self::Q8 => "q8",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationToggle {
    RetainedCompilation,
    ExactEpilogues,
    FusedAttentionPrimitives,
    IndexedDecodeAccumulator,
    GeometryAwareDecode,
}

impl OptimizationToggle {
    pub const ALL: [Self; 5] = [
        Self::RetainedCompilation,
        Self::ExactEpilogues,
        Self::FusedAttentionPrimitives,
        Self::IndexedDecodeAccumulator,
        Self::GeometryAwareDecode,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RetainedCompilation => "retained_compilation",
            Self::ExactEpilogues => "exact_epilogues",
            Self::FusedAttentionPrimitives => "fused_attention_primitives",
            Self::IndexedDecodeAccumulator => "indexed_decode_accumulator",
            Self::GeometryAwareDecode => "geometry_aware_decode",
        }
    }

    pub fn from_name(value: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|toggle| toggle.as_str() == value)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignMode {
    BaselineOnly,
    Partial,
    RequiredAll,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VariantPlan {
    pub id: String,
    pub toggles: Vec<OptimizationToggle>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactContract {
    pub key: String,
    pub repository: String,
    pub resolved_revision: String,
    pub tier: ModelTier,
    pub inventory_sha256: String,
    pub inventory_file_count: u64,
    pub inventory_total_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WorkloadCase {
    pub id: String,
    pub family: BenchmarkFamily,
    pub provider: String,
    pub artifact_key: String,
    pub repository: String,
    pub tier: ModelTier,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub steps: u32,
    pub seed: u64,
    pub prompt: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BenchmarkMatrix {
    pub schema_version: String,
    pub benchmark_id: String,
    pub warmup_runs: u32,
    pub measured_runs: u32,
    pub variants: Vec<VariantPlan>,
    pub artifacts: Vec<ArtifactContract>,
    pub cases: Vec<WorkloadCase>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactBinding {
    pub key: String,
    pub repository: String,
    pub resolved_revision: String,
    pub tier: ModelTier,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: String,
    pub benchmark_id: String,
    pub artifacts: Vec<ArtifactBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractError {
    pub(crate) messages: Vec<String>,
}

impl ContractError {
    fn new(messages: Vec<String>) -> Self {
        Self { messages }
    }

    pub fn messages(&self) -> &[String] {
        &self.messages
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.messages.join("; "))
    }
}

impl std::error::Error for ContractError {}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_lower_hex_revision(value: &str) -> bool {
    is_lower_hex(value, 40)
}

fn is_sha256(value: &str) -> bool {
    is_lower_hex(value, 64)
}

pub fn sha256_json(value: &impl Serialize) -> Result<String, ContractError> {
    serde_json::to_vec(value)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| ContractError::new(vec![format!("serialize fingerprint input: {error}")]))
}

impl BenchmarkMatrix {
    pub fn validate(&self) -> Result<(), ContractError> {
        let mut errors = Vec::new();
        if self.schema_version != MATRIX_SCHEMA_VERSION {
            errors.push(format!(
                "matrix.schemaVersion must be {MATRIX_SCHEMA_VERSION:?}"
            ));
        }
        if self.benchmark_id.trim().is_empty() {
            errors.push("matrix.benchmarkId must not be empty".to_owned());
        }
        if self.warmup_runs == 0 || self.measured_runs < 2 {
            errors.push("matrix requires at least one warmup and two measured runs".to_owned());
        }

        let mut variant_ids = BTreeSet::new();
        let mut singleton_toggles = BTreeSet::new();
        let mut baseline_count = 0;
        let mut all_on_count = 0;
        let all_toggles: BTreeSet<_> = OptimizationToggle::ALL.into_iter().collect();
        for variant in &self.variants {
            if variant.id.trim().is_empty() || !variant_ids.insert(variant.id.as_str()) {
                errors.push(format!(
                    "matrix variant ids must be non-empty and unique (got {:?})",
                    variant.id
                ));
            }
            let unique: BTreeSet<_> = variant.toggles.iter().copied().collect();
            if unique.len() != variant.toggles.len() {
                errors.push(format!("variant {:?} repeats a toggle", variant.id));
            }
            if variant.id == "baseline" {
                baseline_count += 1;
                if !variant.toggles.is_empty() {
                    errors.push("baseline must not request any toggle".to_owned());
                }
            } else if variant.id == "all_on" {
                all_on_count += 1;
                if unique != all_toggles {
                    errors
                        .push("all_on must contain every benchmark toggle exactly once".to_owned());
                }
            } else if variant.toggles.len() == 1 {
                if !singleton_toggles.insert(variant.toggles[0]) {
                    errors.push(format!(
                        "matrix repeats the independent {:?} toggle variant",
                        variant.toggles[0]
                    ));
                }
            } else {
                errors.push(format!(
                    "variant {:?} must be baseline, one independent toggle, or all_on",
                    variant.id
                ));
            }
        }
        if self.variants.len() != 7
            || baseline_count != 1
            || all_on_count != 1
            || singleton_toggles != all_toggles
        {
            errors.push(
                "matrix must contain baseline, one independent variant per toggle, and all_on"
                    .to_owned(),
            );
        }

        let mut artifact_keys = BTreeSet::new();
        for artifact in &self.artifacts {
            if artifact.key.trim().is_empty() || !artifact_keys.insert(artifact.key.as_str()) {
                errors.push(format!(
                    "matrix artifact keys must be non-empty and unique (got {:?})",
                    artifact.key
                ));
            }
            if artifact.repository.trim().is_empty()
                || !is_lower_hex_revision(&artifact.resolved_revision)
                || !is_sha256(&artifact.inventory_sha256)
                || artifact.inventory_file_count == 0
                || artifact.inventory_total_bytes == 0
            {
                errors.push(format!(
                    "artifact contract {:?} must pin repository, revision, and nonempty inventory",
                    artifact.key
                ));
            }
        }

        let mut case_ids = BTreeSet::new();
        let mut counts = BTreeMap::new();
        let mut decode_bound = BTreeSet::new();
        let mut referenced_artifacts = BTreeSet::new();
        for case in &self.cases {
            if case.id.trim().is_empty() || !case_ids.insert(case.id.as_str()) {
                errors.push(format!(
                    "matrix case ids must be non-empty and unique (got {:?})",
                    case.id
                ));
            }
            *counts.entry(case.family).or_insert(0usize) += 1;
            referenced_artifacts.insert(case.artifact_key.as_str());
            if case.width == 0 || case.height == 0 || case.steps == 0 || case.frames == 0 {
                errors.push(format!(
                    "case {:?} must have nonzero geometry, frames, and steps",
                    case.id
                ));
            }
            if case.prompt.trim().is_empty()
                || case.provider.trim().is_empty()
                || case.artifact_key.trim().is_empty()
                || case.repository.trim().is_empty()
            {
                errors.push(format!("case {:?} has an empty identity field", case.id));
            }
            match self.artifact(&case.artifact_key) {
                Some(artifact)
                    if artifact.repository == case.repository && artifact.tier == case.tier => {}
                _ => errors.push(format!(
                    "case {:?} does not match its exact artifact contract",
                    case.id
                )),
            }
            match case.family {
                BenchmarkFamily::WanVideo => {
                    if case.provider != "wan2_2_ti2v_5b"
                        || case.frames <= 1
                        || !(480..=1280).contains(&case.width)
                        || !(480..=1280).contains(&case.height)
                        || !case.width.is_multiple_of(32)
                        || !case.height.is_multiple_of(32)
                        || u64::from(case.width) * u64::from(case.height) > 1280 * 704
                        || case.frames % 4 != 1
                        || case.frames > 1025
                    {
                        errors.push(format!(
                            "Wan case {:?} must satisfy the TI2V-5B geometry/frame contract",
                            case.id
                        ));
                    }
                }
                BenchmarkFamily::ImageDit => {
                    if !(case.provider.starts_with("qwen_image")
                        || case.provider.starts_with("krea_2"))
                        || case.frames != 1
                    {
                        errors.push(format!(
                            "image-DiT case {:?} must use Qwen/Krea with frames=1",
                            case.id
                        ));
                    }
                    if case.width >= 2048 && case.height >= 2048 {
                        decode_bound.insert(case.family);
                    }
                }
                BenchmarkFamily::SdxlUnet => {
                    if case.provider != "sdxl" || case.frames != 1 {
                        errors.push(format!(
                            "SDXL case {:?} must use provider sdxl with frames=1",
                            case.id
                        ));
                    }
                    if case.width >= 2048 && case.height >= 2048 {
                        decode_bound.insert(case.family);
                    }
                }
            }
        }
        if referenced_artifacts != artifact_keys {
            errors.push("matrix artifact contracts must exactly cover referenced keys".to_owned());
        }
        for family in BenchmarkFamily::ALL {
            let count = counts.get(&family).copied().unwrap_or(0);
            if !(2..=3).contains(&count) {
                errors.push(format!(
                    "family {} must contain 2-3 fixed workload cases (got {count})",
                    family.as_str()
                ));
            }
        }
        for family in [BenchmarkFamily::ImageDit, BenchmarkFamily::SdxlUnet] {
            if !decode_bound.contains(&family) {
                errors.push(format!(
                    "family {} must contain a decode-bound geometry at least 2048x2048",
                    family.as_str()
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ContractError::new(errors))
        }
    }

    pub fn case(&self, id: &str) -> Option<&WorkloadCase> {
        self.cases.iter().find(|case| case.id == id)
    }

    pub fn variant(&self, id: &str) -> Option<&VariantPlan> {
        self.variants.iter().find(|variant| variant.id == id)
    }

    pub fn artifact(&self, key: &str) -> Option<&ArtifactContract> {
        self.artifacts.iter().find(|artifact| artifact.key == key)
    }
}

impl ArtifactManifest {
    pub fn validate_against(&self, matrix: &BenchmarkMatrix) -> Result<(), ContractError> {
        let mut errors = Vec::new();
        if self.schema_version != ARTIFACT_SCHEMA_VERSION {
            errors.push(format!(
                "artifacts.schemaVersion must be {ARTIFACT_SCHEMA_VERSION:?}"
            ));
        }
        if self.benchmark_id != matrix.benchmark_id {
            errors.push("artifact benchmarkId does not match the matrix".to_owned());
        }
        let mut actual = BTreeSet::new();
        for binding in &self.artifacts {
            if !actual.insert(binding.key.as_str()) {
                errors.push(format!("artifact key {:?} is duplicated", binding.key));
            }
            let Some(contract) = matrix.artifact(&binding.key) else {
                errors.push(format!("artifact {:?} is not in the matrix", binding.key));
                continue;
            };
            if binding.repository != contract.repository
                || binding.resolved_revision != contract.resolved_revision
                || binding.tier != contract.tier
            {
                errors.push(format!(
                    "artifact {:?} does not match the matrix repository/revision/tier",
                    binding.key
                ));
            }
            if !binding.path.is_absolute() || !binding.path.is_dir() {
                errors.push(format!(
                    "artifact {:?} path must be an existing absolute directory: {}",
                    binding.key,
                    binding.path.display()
                ));
                continue;
            }
            if binding.path.file_name().and_then(|name| name.to_str())
                != Some(binding.tier.as_str())
                || binding
                    .path
                    .parent()
                    .and_then(|parent| parent.file_name())
                    .and_then(|name| name.to_str())
                    != Some(binding.resolved_revision.as_str())
            {
                errors.push(format!(
                    "artifact {:?} path must be the exact tier under its resolved revision",
                    binding.key
                ));
            }
        }
        let expected: BTreeSet<_> = matrix
            .artifacts
            .iter()
            .map(|artifact| artifact.key.as_str())
            .collect();
        if actual != expected {
            errors.push("artifact manifest must exactly cover the matrix artifacts".to_owned());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ContractError::new(errors))
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactInventoryReceipt {
    pub algorithm: String,
    pub file_count: u64,
    pub total_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactReceipt {
    pub key: String,
    pub repository: String,
    pub resolved_revision: String,
    pub tier: ModelTier,
    pub input_path: PathBuf,
    pub canonical_path: PathBuf,
    pub inventory: ArtifactInventoryReceipt,
}

fn collect_inventory_paths(
    root: &Path,
    current: &Path,
    paths: &mut Vec<(String, PathBuf)>,
) -> Result<(), ContractError> {
    let entries = fs::read_dir(current).map_err(|error| {
        ContractError::new(vec![format!(
            "read artifact directory {}: {error}",
            current.display()
        )])
    })?;
    for entry in entries {
        let entry = entry
            .map_err(|error| ContractError::new(vec![format!("read artifact entry: {error}")]))?;
        let path = entry.path();
        let link_meta = fs::symlink_metadata(&path).map_err(|error| {
            ContractError::new(vec![format!(
                "inspect artifact path {}: {error}",
                path.display()
            )])
        })?;
        let meta = fs::metadata(&path).map_err(|error| {
            ContractError::new(vec![format!(
                "resolve artifact path {}: {error}",
                path.display()
            )])
        })?;
        if meta.is_dir() {
            if link_meta.file_type().is_symlink() {
                return Err(ContractError::new(vec![format!(
                    "artifact inventory refuses symlinked directory {}",
                    path.display()
                )]));
            }
            collect_inventory_paths(root, &path, paths)?;
        } else if meta.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("inventory traversal stays below its root")
                .to_str()
                .ok_or_else(|| {
                    ContractError::new(vec![format!(
                        "artifact inventory path is not UTF-8: {}",
                        path.display()
                    )])
                })?
                .replace(std::path::MAIN_SEPARATOR, "/");
            paths.push((relative, path));
        } else {
            return Err(ContractError::new(vec![format!(
                "artifact inventory refuses non-file path {}",
                path.display()
            )]));
        }
    }
    Ok(())
}

pub fn inventory_artifact(path: &Path) -> Result<ArtifactInventoryReceipt, ContractError> {
    let mut paths = Vec::new();
    collect_inventory_paths(path, path, &mut paths)?;
    paths.sort_by(|left, right| left.0.cmp(&right.0));
    if paths.is_empty() {
        return Err(ContractError::new(vec![
            "artifact inventory must contain at least one file".to_owned(),
        ]));
    }
    let mut inventory = Sha256::new();
    let mut total_bytes = 0u64;
    for (relative, file_path) in &paths {
        let mut file = fs::File::open(file_path).map_err(|error| {
            ContractError::new(vec![format!(
                "open artifact file {}: {error}",
                file_path.display()
            )])
        })?;
        let size = file
            .metadata()
            .map_err(|error| {
                ContractError::new(vec![format!(
                    "stat artifact file {}: {error}",
                    file_path.display()
                )])
            })?
            .len();
        let mut content = Sha256::new();
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        loop {
            let read = file.read(&mut buffer).map_err(|error| {
                ContractError::new(vec![format!(
                    "read artifact file {}: {error}",
                    file_path.display()
                )])
            })?;
            if read == 0 {
                break;
            }
            content.update(&buffer[..read]);
        }
        inventory.update(relative.as_bytes());
        inventory.update([0]);
        inventory.update(size.to_le_bytes());
        inventory.update([0]);
        inventory.update(content.finalize());
        total_bytes = total_bytes.checked_add(size).ok_or_else(|| {
            ContractError::new(vec!["artifact inventory byte count overflowed".to_owned()])
        })?;
    }
    Ok(ArtifactInventoryReceipt {
        algorithm: INVENTORY_ALGORITHM.to_owned(),
        file_count: paths.len() as u64,
        total_bytes,
        sha256: format!("{:x}", inventory.finalize()),
    })
}

pub fn freeze_artifacts(
    matrix: &BenchmarkMatrix,
    manifest: &ArtifactManifest,
) -> Result<Vec<ArtifactReceipt>, ContractError> {
    matrix.validate()?;
    manifest.validate_against(matrix)?;
    let mut receipts = Vec::new();
    let mut errors = Vec::new();
    for contract in &matrix.artifacts {
        let binding = manifest
            .artifacts
            .iter()
            .find(|binding| binding.key == contract.key)
            .expect("validated manifest covers every artifact");
        let canonical_path = match binding.path.canonicalize() {
            Ok(path) => path,
            Err(error) => {
                errors.push(format!("canonicalize artifact {:?}: {error}", binding.key));
                continue;
            }
        };
        match inventory_artifact(&canonical_path) {
            Ok(inventory)
                if inventory.algorithm == INVENTORY_ALGORITHM
                    && inventory.sha256 == contract.inventory_sha256
                    && inventory.file_count == contract.inventory_file_count
                    && inventory.total_bytes == contract.inventory_total_bytes =>
            {
                receipts.push(ArtifactReceipt {
                    key: contract.key.clone(),
                    repository: contract.repository.clone(),
                    resolved_revision: contract.resolved_revision.clone(),
                    tier: contract.tier,
                    input_path: binding.path.clone(),
                    canonical_path,
                    inventory,
                });
            }
            Ok(inventory) => errors.push(format!(
                "artifact {:?} inventory does not match revision {}: actual sha256={}, files={}, bytes={}",
                contract.key,
                contract.resolved_revision,
                inventory.sha256,
                inventory.file_count,
                inventory.total_bytes
            )),
            Err(error) => errors.extend(error.messages),
        }
    }
    if errors.is_empty() {
        Ok(receipts)
    } else {
        Err(ContractError::new(errors))
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BuildProvenance {
    pub source_revision: String,
    pub mlx_revision: String,
    pub source_dirty: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HostIdentity {
    pub rustc_version: String,
    pub os_version: String,
    pub hardware_model: String,
    pub metal_device: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProviderCapabilityReceipt {
    pub provider: String,
    pub available_toggles: Vec<OptimizationToggle>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct CampaignIdentityMaterial<'a> {
    created_at_unix_millis: u128,
    mode: CampaignMode,
    selected_variants: &'a [String],
    matrix_sha256: &'a str,
    input_artifact_manifest_sha256: &'a str,
    artifact_set_sha256: &'a str,
    build: &'a BuildProvenance,
    host: &'a HostIdentity,
    provider_capabilities: &'a [ProviderCapabilityReceipt],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FrozenCampaign {
    pub schema_version: String,
    pub campaign_id: String,
    pub created_at_unix_millis: u128,
    pub mode: CampaignMode,
    pub selected_variants: Vec<String>,
    pub matrix: BenchmarkMatrix,
    pub matrix_sha256: String,
    pub input_artifact_manifest: ArtifactManifest,
    pub input_artifact_manifest_sha256: String,
    pub artifacts: Vec<ArtifactReceipt>,
    pub artifact_set_sha256: String,
    pub build: BuildProvenance,
    pub host: HostIdentity,
    pub provider_capabilities: Vec<ProviderCapabilityReceipt>,
}

fn validate_selection(
    matrix: &BenchmarkMatrix,
    selected: &[String],
) -> Result<CampaignMode, ContractError> {
    let mut errors = Vec::new();
    let unique: BTreeSet<_> = selected.iter().map(String::as_str).collect();
    if selected.is_empty() || unique.len() != selected.len() || !unique.contains("baseline") {
        errors.push("selected variants must be nonempty, unique, and include baseline".to_owned());
    }
    for id in selected {
        if matrix.variant(id).is_none() {
            errors.push(format!("selected unknown variant {id:?}"));
        }
    }
    if !errors.is_empty() {
        return Err(ContractError::new(errors));
    }
    let all: BTreeSet<_> = matrix
        .variants
        .iter()
        .map(|variant| variant.id.as_str())
        .collect();
    Ok(if unique == all {
        CampaignMode::RequiredAll
    } else if selected.len() == 1 && selected[0] == "baseline" {
        CampaignMode::BaselineOnly
    } else {
        CampaignMode::Partial
    })
}

impl FrozenCampaign {
    #[allow(clippy::too_many_arguments)]
    pub fn freeze(
        matrix: BenchmarkMatrix,
        manifest: &ArtifactManifest,
        selected_variants: Vec<String>,
        build: BuildProvenance,
        host: HostIdentity,
        mut provider_capabilities: Vec<ProviderCapabilityReceipt>,
        created_at_unix_millis: u128,
    ) -> Result<Self, ContractError> {
        matrix.validate()?;
        let mode = validate_selection(&matrix, &selected_variants)?;
        let selected_set: BTreeSet<_> = selected_variants.iter().map(String::as_str).collect();
        let selected_variants: Vec<_> = matrix
            .variants
            .iter()
            .filter(|variant| selected_set.contains(variant.id.as_str()))
            .map(|variant| variant.id.clone())
            .collect();
        provider_capabilities.sort_by(|left, right| left.provider.cmp(&right.provider));
        for receipt in &mut provider_capabilities {
            receipt.available_toggles.sort_unstable();
        }
        let input_artifact_manifest = ArtifactManifest {
            schema_version: manifest.schema_version.clone(),
            benchmark_id: manifest.benchmark_id.clone(),
            artifacts: matrix
                .artifacts
                .iter()
                .filter_map(|contract| {
                    manifest
                        .artifacts
                        .iter()
                        .find(|binding| binding.key == contract.key)
                        .cloned()
                })
                .collect(),
        };
        let matrix_sha256 = sha256_json(&matrix)?;
        let input_artifact_manifest_sha256 = sha256_json(&input_artifact_manifest)?;
        let artifacts = freeze_artifacts(&matrix, manifest)?;
        let artifact_set_sha256 = sha256_json(&artifacts)?;
        let material = CampaignIdentityMaterial {
            created_at_unix_millis,
            mode,
            selected_variants: &selected_variants,
            matrix_sha256: &matrix_sha256,
            input_artifact_manifest_sha256: &input_artifact_manifest_sha256,
            artifact_set_sha256: &artifact_set_sha256,
            build: &build,
            host: &host,
            provider_capabilities: &provider_capabilities,
        };
        let campaign_id = sha256_json(&material)?;
        let campaign = Self {
            schema_version: CAMPAIGN_SCHEMA_VERSION.to_owned(),
            campaign_id,
            created_at_unix_millis,
            mode,
            selected_variants,
            matrix,
            matrix_sha256,
            input_artifact_manifest,
            input_artifact_manifest_sha256,
            artifacts,
            artifact_set_sha256,
            build,
            host,
            provider_capabilities,
        };
        campaign.validate()?;
        Ok(campaign)
    }

    pub fn validate(&self) -> Result<(), ContractError> {
        let mut errors = Vec::new();
        if self.schema_version != CAMPAIGN_SCHEMA_VERSION {
            errors.push(format!(
                "campaign.schemaVersion must be {CAMPAIGN_SCHEMA_VERSION:?}; legacy evidence is unbound"
            ));
        }
        if let Err(error) = self.matrix.validate() {
            errors.extend(error.messages);
        }
        if sha256_json(&self.matrix).as_deref() != Ok(self.matrix_sha256.as_str()) {
            errors.push("campaign matrix fingerprint mismatch".to_owned());
        }
        if self.input_artifact_manifest.schema_version != ARTIFACT_SCHEMA_VERSION
            || self.input_artifact_manifest.benchmark_id != self.matrix.benchmark_id
        {
            errors.push("campaign contains an invalid frozen input artifact manifest".to_owned());
        }
        if sha256_json(&self.input_artifact_manifest).as_deref()
            != Ok(self.input_artifact_manifest_sha256.as_str())
        {
            errors.push("campaign input artifact-manifest fingerprint mismatch".to_owned());
        }
        if sha256_json(&self.artifacts).as_deref() != Ok(self.artifact_set_sha256.as_str()) {
            errors.push("campaign artifact-set fingerprint mismatch".to_owned());
        }
        match validate_selection(&self.matrix, &self.selected_variants) {
            Ok(mode) if mode == self.mode => {}
            _ => errors.push("campaign mode does not match selected variants".to_owned()),
        }
        let selected_set: BTreeSet<_> = self.selected_variants.iter().map(String::as_str).collect();
        let canonical_selection: Vec<_> = self
            .matrix
            .variants
            .iter()
            .filter(|variant| selected_set.contains(variant.id.as_str()))
            .map(|variant| variant.id.as_str())
            .collect();
        if self
            .selected_variants
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != canonical_selection
        {
            errors.push("campaign selected variants are not in canonical matrix order".to_owned());
        }
        if self.created_at_unix_millis == 0
            || !is_lower_hex_revision(&self.build.source_revision)
            || !is_lower_hex_revision(&self.build.mlx_revision)
            || self.build.source_dirty
        {
            errors.push("campaign requires a clean, exact source and mlx build receipt".to_owned());
        }
        if self.host.rustc_version.trim().is_empty()
            || self.host.os_version.trim().is_empty()
            || self.host.hardware_model.trim().is_empty()
            || self.host.metal_device.trim().is_empty()
        {
            errors.push("campaign host identity must be complete".to_owned());
        }

        let contracts: BTreeMap<_, _> = self
            .matrix
            .artifacts
            .iter()
            .map(|contract| (contract.key.as_str(), contract))
            .collect();
        if self
            .input_artifact_manifest
            .artifacts
            .iter()
            .map(|binding| binding.key.as_str())
            .ne(self
                .matrix
                .artifacts
                .iter()
                .map(|contract| contract.key.as_str()))
        {
            errors.push(
                "campaign frozen input manifest must exactly follow matrix artifact order"
                    .to_owned(),
            );
        }
        if self
            .artifacts
            .iter()
            .map(|receipt| receipt.key.as_str())
            .ne(self
                .matrix
                .artifacts
                .iter()
                .map(|contract| contract.key.as_str()))
        {
            errors.push("campaign artifact receipts are not in canonical matrix order".to_owned());
        }
        let mut receipt_keys = BTreeSet::new();
        for receipt in &self.artifacts {
            if !receipt_keys.insert(receipt.key.as_str()) {
                errors.push(format!(
                    "campaign repeats artifact receipt {:?}",
                    receipt.key
                ));
                continue;
            }
            match contracts.get(receipt.key.as_str()) {
                Some(contract)
                    if receipt.repository == contract.repository
                        && receipt.resolved_revision == contract.resolved_revision
                        && receipt.tier == contract.tier
                        && receipt.inventory.algorithm == INVENTORY_ALGORITHM
                        && receipt.inventory.sha256 == contract.inventory_sha256
                        && receipt.inventory.file_count == contract.inventory_file_count
                        && receipt.inventory.total_bytes == contract.inventory_total_bytes
                        && receipt.input_path.is_absolute()
                        && receipt.canonical_path.is_absolute() => {}
                _ => errors.push(format!(
                    "campaign artifact receipt {:?} does not match its immutable contract",
                    receipt.key
                )),
            }
            match self
                .input_artifact_manifest
                .artifacts
                .iter()
                .find(|binding| binding.key == receipt.key)
            {
                Some(binding)
                    if binding.repository == receipt.repository
                        && binding.resolved_revision == receipt.resolved_revision
                        && binding.tier == receipt.tier
                        && binding.path == receipt.input_path => {}
                _ => errors.push(format!(
                    "campaign artifact receipt {:?} does not match its frozen input binding",
                    receipt.key
                )),
            }
        }
        if receipt_keys != contracts.keys().copied().collect() {
            errors.push("campaign artifact receipts must exactly cover contracts".to_owned());
        }

        let expected_providers: BTreeSet<_> = self
            .matrix
            .cases
            .iter()
            .map(|case| case.provider.as_str())
            .collect();
        let mut actual_providers = BTreeSet::new();
        let all_toggles: BTreeSet<_> = OptimizationToggle::ALL.into_iter().collect();
        let mut capabilities = BTreeMap::new();
        for receipt in &self.provider_capabilities {
            if !actual_providers.insert(receipt.provider.as_str()) {
                errors.push(format!(
                    "campaign repeats provider capability {:?}",
                    receipt.provider
                ));
            }
            let available: BTreeSet<_> = receipt.available_toggles.iter().copied().collect();
            if available.len() != receipt.available_toggles.len()
                || !available.is_subset(&all_toggles)
                || receipt
                    .available_toggles
                    .windows(2)
                    .any(|window| window[0] >= window[1])
            {
                errors.push(format!(
                    "provider {:?} capability set must be unique and known",
                    receipt.provider
                ));
            }
            capabilities.insert(receipt.provider.as_str(), available);
        }
        if self
            .provider_capabilities
            .windows(2)
            .any(|window| window[0].provider >= window[1].provider)
        {
            errors.push("campaign provider capabilities are not in canonical order".to_owned());
        }
        if actual_providers != expected_providers {
            errors.push("campaign capabilities must exactly cover matrix providers".to_owned());
        }
        for case in &self.matrix.cases {
            let available = capabilities.get(case.provider.as_str());
            for variant_id in &self.selected_variants {
                let variant = self
                    .matrix
                    .variant(variant_id)
                    .expect("validated selection names a variant");
                if !variant.toggles.is_empty()
                    && !variant
                        .toggles
                        .iter()
                        .all(|toggle| available.is_some_and(|set| set.contains(toggle)))
                {
                    errors.push(format!(
                        "provider {:?} cannot run requested variant {:?}; capability is unavailable",
                        case.provider, variant.id
                    ));
                }
            }
        }

        let material = CampaignIdentityMaterial {
            created_at_unix_millis: self.created_at_unix_millis,
            mode: self.mode,
            selected_variants: &self.selected_variants,
            matrix_sha256: &self.matrix_sha256,
            input_artifact_manifest_sha256: &self.input_artifact_manifest_sha256,
            artifact_set_sha256: &self.artifact_set_sha256,
            build: &self.build,
            host: &self.host,
            provider_capabilities: &self.provider_capabilities,
        };
        if sha256_json(&material).as_deref() != Ok(self.campaign_id.as_str()) {
            errors.push("campaign identity fingerprint mismatch".to_owned());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ContractError::new(errors))
        }
    }

    pub fn artifact(&self, key: &str) -> Option<&ArtifactReceipt> {
        self.artifacts.iter().find(|artifact| artifact.key == key)
    }

    pub fn capabilities(&self, provider: &str) -> Option<&[OptimizationToggle]> {
        self.provider_capabilities
            .iter()
            .find(|receipt| receipt.provider == provider)
            .map(|receipt| receipt.available_toggles.as_slice())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RequestReceipt {
    pub case: WorkloadCase,
    pub artifact_inventory_sha256: String,
    pub sha256: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RequestIdentityMaterial<'a> {
    campaign_id: &'a str,
    case: &'a WorkloadCase,
    artifact: &'a ArtifactReceipt,
}

pub fn request_receipt(
    campaign: &FrozenCampaign,
    case: &WorkloadCase,
    artifact: &ArtifactReceipt,
) -> Result<RequestReceipt, ContractError> {
    let sha256 = sha256_json(&RequestIdentityMaterial {
        campaign_id: &campaign.campaign_id,
        case,
        artifact,
    })?;
    Ok(RequestReceipt {
        case: case.clone(),
        artifact_inventory_sha256: artifact.inventory.sha256.clone(),
        sha256,
    })
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PhaseBoundary {
    DenoiseStart,
    DecodeStart,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhaseBoundaryReceipt {
    pub boundary: PhaseBoundary,
    pub elapsed_nanos: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StepReceipt {
    pub current: u32,
    pub total: u32,
    pub elapsed_nanos: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ProgressReceipt {
    pub steps: Vec<StepReceipt>,
    pub decoding_elapsed_nanos: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryCoverageReceipt {
    pub interval_micros: u64,
    pub sample_count: u64,
    pub periodic_sample_count: u64,
    pub sampling_span_micros: u64,
    pub max_gap_micros: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhaseMetrics {
    pub seconds: f64,
    pub native_active_peak_bytes: u64,
    pub sampled_active_peak_bytes: u64,
    pub sampled_cache_peak_bytes: u64,
    pub sampled_footprint_peak_bytes: u64,
    pub footprint_peak_active_bytes: u64,
    pub footprint_peak_cache_bytes: u64,
    pub boundary_active_bytes: u64,
    pub boundary_cache_bytes: u64,
    pub coverage: MemoryCoverageReceipt,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhaseSet {
    pub encode: PhaseMetrics,
    pub denoise: PhaseMetrics,
    pub decode: PhaseMetrics,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OutputFingerprint {
    pub kind: String,
    pub items: u32,
    pub width: u32,
    pub height: u32,
    pub payload_bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiagnosticRecord {
    pub domain: String,
    pub site: String,
    pub outcome: String,
    pub count: u64,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MeasurementRecord {
    pub repetition: u32,
    pub total_elapsed_nanos: u64,
    pub total_seconds: f64,
    pub denoise_steps_per_second: f64,
    pub progress: ProgressReceipt,
    pub phase_boundaries: Vec<PhaseBoundaryReceipt>,
    pub phases: PhaseSet,
    pub output: OutputFingerprint,
    pub diagnostics: Vec<DiagnosticRecord>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunRecord {
    pub schema_version: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub case_id: String,
    pub family: BenchmarkFamily,
    pub provider: String,
    pub artifact: ArtifactReceipt,
    pub variant: VariantPlan,
    pub request: RequestReceipt,
    pub build: BuildProvenance,
    pub host: HostIdentity,
    pub available_toggles: Vec<OptimizationToggle>,
    pub started_at_unix_millis: u128,
    pub load_seconds: f64,
    pub load_active_peak_bytes: u64,
    pub load_cache_bytes_after_load: u64,
    pub warmup_runs_completed: u32,
    pub measurements: Vec<MeasurementRecord>,
}

fn relative_close(actual: f64, expected: f64) -> bool {
    let scale = expected.abs().max(1.0);
    (actual - expected).abs() <= scale * 1e-6
}

fn validate_phase(name: &str, phase: &PhaseMetrics, errors: &mut Vec<String>) {
    let coverage = &phase.coverage;
    if !(phase.seconds.is_finite() && phase.seconds > 0.0)
        || phase.native_active_peak_bytes == 0
        || phase.sampled_footprint_peak_bytes == 0
        || phase.sampled_active_peak_bytes > phase.native_active_peak_bytes
        || phase.boundary_active_bytes > phase.sampled_active_peak_bytes
        || phase.boundary_cache_bytes > phase.sampled_cache_peak_bytes
        || phase.sampled_footprint_peak_bytes < phase.sampled_active_peak_bytes
        || phase.sampled_footprint_peak_bytes < phase.sampled_cache_peak_bytes
        || phase.sampled_footprint_peak_bytes
            > phase
                .sampled_active_peak_bytes
                .saturating_add(phase.sampled_cache_peak_bytes)
        || phase.sampled_footprint_peak_bytes
            != phase
                .footprint_peak_active_bytes
                .saturating_add(phase.footprint_peak_cache_bytes)
        || phase.footprint_peak_active_bytes > phase.sampled_active_peak_bytes
        || phase.footprint_peak_cache_bytes > phase.sampled_cache_peak_bytes
    {
        errors.push(format!(
            "measurement {name} phase has invalid active/cache/paired-footprint witness evidence"
        ));
    }
    if coverage.interval_micros != MEMORY_SAMPLE_INTERVAL_MICROS
        || coverage.sample_count != coverage.periodic_sample_count.saturating_add(2)
        || coverage.sample_count < 2
    {
        errors.push(format!(
            "measurement {name} phase has an invalid sampling-count/cadence receipt"
        ));
    }
    let phase_micros = (phase.seconds * 1_000_000.0).round() as u64;
    if coverage.sampling_span_micros.abs_diff(phase_micros) > MEMORY_SAMPLE_INTERVAL_MICROS {
        errors.push(format!(
            "measurement {name} phase sampling span does not cover its duration"
        ));
    }
    if coverage.sampling_span_micros > 4 * MEMORY_SAMPLE_INTERVAL_MICROS
        && (coverage.periodic_sample_count == 0
            || coverage.max_gap_micros > MEMORY_MAX_GAP_MULTIPLIER * MEMORY_SAMPLE_INTERVAL_MICROS)
    {
        errors.push(format!(
            "measurement {name} phase lacks fixed-cadence sampling coverage"
        ));
    }
}

fn validate_toggle_receipts(
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
    errors: &mut Vec<String>,
) {
    let requested: BTreeSet<_> = variant
        .toggles
        .iter()
        .map(|toggle| toggle.as_str())
        .collect();
    let mut seen = BTreeSet::new();
    for diagnostic in diagnostics {
        if diagnostic.count == 0 {
            errors.push("diagnostic receipts must have positive counts".to_owned());
        }
        if !seen.insert((
            diagnostic.domain.as_str(),
            diagnostic.site.as_str(),
            diagnostic.outcome.as_str(),
            diagnostic.reason.as_deref(),
        )) {
            errors
                .push("diagnostic records must be aggregated into an exact unique set".to_owned());
        }
    }
    let toggle_records: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.domain == "toggle")
        .collect();
    if requested.is_empty() {
        if !toggle_records.is_empty() {
            errors.push("baseline forbids every toggle terminal receipt".to_owned());
        }
        return;
    }
    if diagnostics
        .iter()
        .any(|diagnostic| diagnostic.domain == "fallback")
    {
        errors.push("requested variants forbid fallback diagnostics".to_owned());
    }
    for record in &toggle_records {
        if !requested.contains(record.site.as_str()) {
            errors.push(format!(
                "variant {:?} emitted an unrequested toggle receipt for {}",
                variant.id, record.site
            ));
        }
    }
    for toggle in requested {
        let records: Vec<_> = toggle_records
            .iter()
            .filter(|record| record.site == toggle)
            .collect();
        if records.len() != 1
            || records[0].outcome != "applied"
            || records[0].count == 0
            || records[0].reason.is_some()
        {
            errors.push(format!(
                "variant {:?} requires exactly one terminal Applied record and no fallback/unavailable for {toggle}",
                variant.id
            ));
        }
    }
}

/// Validate one request's complete, aggregated diagnostic set against its selected variant.
///
/// The runner applies this to warmups as well as measured repetitions so an unavailable or
/// silently-fallback toggle aborts the campaign before it can consume the remaining matrix.
pub fn validate_toggle_diagnostics(
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
) -> Result<(), ContractError> {
    let mut errors = Vec::new();
    validate_toggle_receipts(variant, diagnostics, &mut errors);
    if errors.is_empty() {
        Ok(())
    } else {
        Err(ContractError::new(errors))
    }
}

impl RunRecord {
    pub fn validate_against(&self, campaign: &FrozenCampaign) -> Result<(), ContractError> {
        let mut errors = Vec::new();
        if let Err(error) = campaign.validate() {
            errors.extend(error.messages);
        }
        if self.schema_version != RUN_SCHEMA_VERSION {
            errors.push(format!(
                "run.schemaVersion must be {RUN_SCHEMA_VERSION:?}; legacy evidence is unbound"
            ));
        }
        if self.campaign_id != campaign.campaign_id
            || self.benchmark_id != campaign.matrix.benchmark_id
            || self.build != campaign.build
            || self.host != campaign.host
        {
            errors.push(
                "run campaign/build/host identity does not match the frozen campaign".to_owned(),
            );
        }
        let Some(case) = campaign.matrix.case(&self.case_id) else {
            return Err(ContractError::new(vec![format!(
                "run references unknown case {:?}",
                self.case_id
            )]));
        };
        let expected_artifact = campaign
            .artifact(&case.artifact_key)
            .expect("validated campaign covers every case artifact");
        if self.family != case.family
            || self.provider != case.provider
            || self.artifact != *expected_artifact
        {
            errors.push("run family/provider/artifact identity does not match the case".to_owned());
        }
        if campaign.matrix.variant(&self.variant.id) != Some(&self.variant)
            || !campaign.selected_variants.contains(&self.variant.id)
        {
            errors.push("run variant is not an exact selected matrix variant".to_owned());
        }
        let expected_capabilities = campaign.capabilities(&self.provider).unwrap_or(&[]);
        if self.available_toggles != expected_capabilities {
            errors.push("run provider capability receipt does not match the campaign".to_owned());
        }
        if !self
            .variant
            .toggles
            .iter()
            .all(|toggle| self.available_toggles.contains(toggle))
        {
            errors.push("run requested a provider-unavailable toggle".to_owned());
        }
        match request_receipt(campaign, case, expected_artifact) {
            Ok(expected) if self.request == expected => {}
            _ => errors.push(
                "run request receipt/fingerprint does not bind prompt, seed, steps, geometry, model, and campaign"
                    .to_owned(),
            ),
        }
        if self.started_at_unix_millis == 0
            || !(self.load_seconds.is_finite() && self.load_seconds > 0.0)
        {
            errors.push("run must record a timestamp and positive cold-load timing".to_owned());
        }
        if self.warmup_runs_completed != campaign.matrix.warmup_runs {
            errors.push(format!(
                "run completed {} warmups, expected {}",
                self.warmup_runs_completed, campaign.matrix.warmup_runs
            ));
        }
        if self.measurements.len() != campaign.matrix.measured_runs as usize {
            errors.push(format!(
                "run has {} measured repetitions, expected {}",
                self.measurements.len(),
                campaign.matrix.measured_runs
            ));
        }

        let mut repetitions = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for measurement in &self.measurements {
            if !repetitions.insert(measurement.repetition) {
                errors.push(format!(
                    "run repeats measurement index {}",
                    measurement.repetition
                ));
            }
            let expected_total_seconds = measurement.total_elapsed_nanos as f64 / 1e9;
            if measurement.total_elapsed_nanos == 0
                || !relative_close(measurement.total_seconds, expected_total_seconds)
            {
                errors.push("measurement total timing receipt is inconsistent".to_owned());
            }
            if measurement.progress.steps.len() != case.steps as usize {
                errors.push(format!(
                    "measurement must preserve exactly {} Step receipts",
                    case.steps
                ));
            }
            let mut previous_elapsed = 0;
            for (index, step) in measurement.progress.steps.iter().enumerate() {
                let expected_current = index as u32 + 1;
                if step.current != expected_current
                    || step.total != case.steps
                    || step.elapsed_nanos < previous_elapsed
                    || step.elapsed_nanos >= measurement.progress.decoding_elapsed_nanos
                {
                    errors.push(
                        "measurement Step receipts must be exact monotone 1..N before Decoding"
                            .to_owned(),
                    );
                    break;
                }
                previous_elapsed = step.elapsed_nanos;
            }
            if measurement.progress.decoding_elapsed_nanos == 0
                || measurement.progress.decoding_elapsed_nanos >= measurement.total_elapsed_nanos
            {
                errors.push("measurement must preserve one in-order Decoding receipt".to_owned());
            }
            let expected_boundaries = [PhaseBoundary::DenoiseStart, PhaseBoundary::DecodeStart];
            if measurement.phase_boundaries.len() != 2
                || measurement
                    .phase_boundaries
                    .iter()
                    .map(|receipt| receipt.boundary)
                    .ne(expected_boundaries)
                || measurement.phase_boundaries[0].elapsed_nanos == 0
                || measurement.phase_boundaries[0].elapsed_nanos
                    >= measurement.phase_boundaries[1].elapsed_nanos
                || measurement.phase_boundaries[1].elapsed_nanos
                    > measurement.progress.decoding_elapsed_nanos
            {
                errors.push(
                    "measurement requires explicit ordered DenoiseStart and DecodeStart boundaries"
                        .to_owned(),
                );
            } else {
                let denoise_start = measurement.phase_boundaries[0].elapsed_nanos;
                let decode_start = measurement.phase_boundaries[1].elapsed_nanos;
                if measurement.progress.steps.iter().any(|step| {
                    step.elapsed_nanos < denoise_start || step.elapsed_nanos > decode_start
                }) {
                    errors.push(
                        "measurement Step receipts must lie inside the explicit denoise phase"
                            .to_owned(),
                    );
                }
                let encode_seconds = denoise_start as f64 / 1e9;
                let denoise_seconds = decode_start.saturating_sub(denoise_start) as f64 / 1e9;
                let decode_seconds =
                    measurement.total_elapsed_nanos.saturating_sub(decode_start) as f64 / 1e9;
                if !relative_close(measurement.phases.encode.seconds, encode_seconds)
                    || !relative_close(measurement.phases.denoise.seconds, denoise_seconds)
                    || !relative_close(measurement.phases.decode.seconds, decode_seconds)
                {
                    errors.push(
                        "measurement phase durations do not match explicit boundaries".to_owned(),
                    );
                }
                let expected_rate = case.steps as f64 / denoise_seconds;
                if !relative_close(measurement.denoise_steps_per_second, expected_rate) {
                    errors.push(
                        "measurement denoise rate does not cover all configured steps".to_owned(),
                    );
                }
            }
            validate_phase("encode", &measurement.phases.encode, &mut errors);
            validate_phase("denoise", &measurement.phases.denoise, &mut errors);
            validate_phase("decode", &measurement.phases.decode, &mut errors);
            if measurement.output.items == 0
                || measurement.output.payload_bytes == 0
                || !is_sha256(&measurement.output.sha256)
            {
                errors.push("measurement output must be nonempty with a SHA-256 digest".to_owned());
            }
            let (expected_kind, expected_items) = match case.family {
                BenchmarkFamily::WanVideo => ("video", case.frames),
                BenchmarkFamily::ImageDit | BenchmarkFamily::SdxlUnet => ("images", 1),
            };
            if measurement.output.kind != expected_kind
                || measurement.output.items != expected_items
                || measurement.output.width != case.width
                || measurement.output.height != case.height
            {
                errors.push(
                    "measurement output kind/count/geometry do not match the case".to_owned(),
                );
            }
            digests.insert(measurement.output.sha256.as_str());
            validate_toggle_receipts(&self.variant, &measurement.diagnostics, &mut errors);
        }
        if digests.len() > 1 {
            errors.push("measured outputs are not byte-stable for this case/variant".to_owned());
        }
        let expected_repetitions: BTreeSet<_> = (0..campaign.matrix.measured_runs).collect();
        if repetitions != expected_repetitions {
            errors.push("measurement indexes must exactly cover 0..measuredRuns".to_owned());
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ContractError::new(errors))
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BindingPhase {
    Encode,
    Denoise,
    Decode,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ComparisonRow {
    pub case_id: String,
    pub family: BenchmarkFamily,
    pub tier: ModelTier,
    pub variant_id: String,
    pub median_total_seconds: f64,
    pub median_encode_seconds: f64,
    pub median_denoise_seconds: f64,
    pub median_decode_seconds: f64,
    pub median_denoise_steps_per_second: f64,
    pub median_encode_native_active_peak_bytes: u64,
    pub median_denoise_native_active_peak_bytes: u64,
    pub median_decode_native_active_peak_bytes: u64,
    pub median_encode_sampled_cache_peak_bytes: u64,
    pub median_denoise_sampled_cache_peak_bytes: u64,
    pub median_decode_sampled_cache_peak_bytes: u64,
    pub median_encode_sampled_footprint_peak_bytes: u64,
    pub median_denoise_sampled_footprint_peak_bytes: u64,
    pub median_decode_sampled_footprint_peak_bytes: u64,
    pub binding_phase: BindingPhase,
    pub speedup_vs_baseline: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BenchmarkSummary {
    pub schema_version: String,
    pub campaign_id: String,
    pub benchmark_id: String,
    pub mode: CampaignMode,
    pub acceptance_complete: bool,
    pub matrix_sha256: String,
    pub artifact_set_sha256: String,
    pub build: BuildProvenance,
    pub host: HostIdentity,
    pub binding_rule: String,
    pub rows: Vec<ComparisonRow>,
}

pub fn build_summary(
    campaign: &FrozenCampaign,
    records: &[RunRecord],
) -> Result<BenchmarkSummary, ContractError> {
    campaign.validate()?;
    let mut errors = Vec::new();
    let mut by_key = BTreeMap::new();
    for record in records {
        if let Err(error) = record.validate_against(campaign) {
            errors.extend(error.messages);
            continue;
        }
        let key = (record.case_id.as_str(), record.variant.id.as_str());
        if by_key.insert(key, record).is_some() {
            errors.push(format!("duplicate run record for {} / {}", key.0, key.1));
        }
    }
    for case in &campaign.matrix.cases {
        for variant in &campaign.selected_variants {
            if !by_key.contains_key(&(case.id.as_str(), variant.as_str())) {
                errors.push(format!("missing run record for {} / {}", case.id, variant));
            }
        }
    }
    for case in &campaign.matrix.cases {
        let Some(baseline) = by_key.get(&(case.id.as_str(), "baseline")) else {
            continue;
        };
        let baseline_digest = &baseline.measurements[0].output.sha256;
        for variant in &campaign.selected_variants {
            let Some(record) = by_key.get(&(case.id.as_str(), variant.as_str())) else {
                continue;
            };
            if record
                .measurements
                .iter()
                .any(|measurement| measurement.output.sha256 != *baseline_digest)
            {
                errors.push(format!(
                    "case {} variant {} output digest differs from baseline",
                    case.id, variant
                ));
            }
        }
    }
    if !errors.is_empty() {
        return Err(ContractError::new(errors));
    }

    let mut rows = Vec::new();
    for case in &campaign.matrix.cases {
        let baseline = by_key[&(case.id.as_str(), "baseline")];
        let baseline_total = median_f64(
            baseline
                .measurements
                .iter()
                .map(|measurement| measurement.total_seconds)
                .collect(),
        );
        for variant_id in &campaign.selected_variants {
            let record = by_key[&(case.id.as_str(), variant_id.as_str())];
            let median_phase_f64 = |pick: fn(&PhaseSet) -> &PhaseMetrics| {
                median_f64(
                    record
                        .measurements
                        .iter()
                        .map(|measurement| pick(&measurement.phases).seconds)
                        .collect(),
                )
            };
            let median_phase_u64 =
                |pick: fn(&PhaseSet) -> &PhaseMetrics, field: fn(&PhaseMetrics) -> u64| {
                    median_u64(
                        record
                            .measurements
                            .iter()
                            .map(|measurement| field(pick(&measurement.phases)))
                            .collect(),
                    )
                };
            let total = median_f64(
                record
                    .measurements
                    .iter()
                    .map(|measurement| measurement.total_seconds)
                    .collect(),
            );
            let encode_active = median_phase_u64(
                |phases| &phases.encode,
                |phase| phase.native_active_peak_bytes,
            );
            let denoise_active = median_phase_u64(
                |phases| &phases.denoise,
                |phase| phase.native_active_peak_bytes,
            );
            let decode_active = median_phase_u64(
                |phases| &phases.decode,
                |phase| phase.native_active_peak_bytes,
            );
            let encode_cache = median_phase_u64(
                |phases| &phases.encode,
                |phase| phase.sampled_cache_peak_bytes,
            );
            let denoise_cache = median_phase_u64(
                |phases| &phases.denoise,
                |phase| phase.sampled_cache_peak_bytes,
            );
            let decode_cache = median_phase_u64(
                |phases| &phases.decode,
                |phase| phase.sampled_cache_peak_bytes,
            );
            let encode_footprint = median_phase_u64(
                |phases| &phases.encode,
                |phase| phase.sampled_footprint_peak_bytes,
            );
            let denoise_footprint = median_phase_u64(
                |phases| &phases.denoise,
                |phase| phase.sampled_footprint_peak_bytes,
            );
            let decode_footprint = median_phase_u64(
                |phases| &phases.decode,
                |phase| phase.sampled_footprint_peak_bytes,
            );
            let binding_phase = [
                (BindingPhase::Encode, encode_footprint),
                (BindingPhase::Denoise, denoise_footprint),
                (BindingPhase::Decode, decode_footprint),
            ]
            .into_iter()
            .max_by_key(|(_, bytes)| *bytes)
            .expect("three phases")
            .0;
            rows.push(ComparisonRow {
                case_id: case.id.clone(),
                family: case.family,
                tier: case.tier,
                variant_id: variant_id.clone(),
                median_total_seconds: total,
                median_encode_seconds: median_phase_f64(|phases| &phases.encode),
                median_denoise_seconds: median_phase_f64(|phases| &phases.denoise),
                median_decode_seconds: median_phase_f64(|phases| &phases.decode),
                median_denoise_steps_per_second: median_f64(
                    record
                        .measurements
                        .iter()
                        .map(|measurement| measurement.denoise_steps_per_second)
                        .collect(),
                ),
                median_encode_native_active_peak_bytes: encode_active,
                median_denoise_native_active_peak_bytes: denoise_active,
                median_decode_native_active_peak_bytes: decode_active,
                median_encode_sampled_cache_peak_bytes: encode_cache,
                median_denoise_sampled_cache_peak_bytes: denoise_cache,
                median_decode_sampled_cache_peak_bytes: decode_cache,
                median_encode_sampled_footprint_peak_bytes: encode_footprint,
                median_denoise_sampled_footprint_peak_bytes: denoise_footprint,
                median_decode_sampled_footprint_peak_bytes: decode_footprint,
                binding_phase,
                speedup_vs_baseline: baseline_total / total,
            });
        }
    }
    Ok(BenchmarkSummary {
        schema_version: SUMMARY_SCHEMA_VERSION.to_owned(),
        campaign_id: campaign.campaign_id.clone(),
        benchmark_id: campaign.matrix.benchmark_id.clone(),
        mode: campaign.mode,
        acceptance_complete: campaign.mode == CampaignMode::RequiredAll,
        matrix_sha256: campaign.matrix_sha256.clone(),
        artifact_set_sha256: campaign.artifact_set_sha256.clone(),
        build: campaign.build.clone(),
        host: campaign.host.clone(),
        binding_rule: MEMORY_BINDING_RULE.to_owned(),
        rows,
    })
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn median_u64(mut values: Vec<u64>) -> u64 {
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1]
            .saturating_add(values[middle])
            .saturating_div(2)
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn committed_matrix() -> BenchmarkMatrix {
        serde_json::from_str(include_str!("../benchmarks/mlx-perf-matrix-v1.json")).unwrap()
    }

    fn phase(seconds: f64, footprint: u64) -> PhaseMetrics {
        let sampling_span_micros = (seconds * 1_000_000.0) as u64;
        let periodic_sample_count = sampling_span_micros / MEMORY_SAMPLE_INTERVAL_MICROS;
        PhaseMetrics {
            seconds,
            native_active_peak_bytes: footprint,
            sampled_active_peak_bytes: footprint,
            sampled_cache_peak_bytes: 0,
            sampled_footprint_peak_bytes: footprint,
            footprint_peak_active_bytes: footprint,
            footprint_peak_cache_bytes: 0,
            boundary_active_bytes: footprint,
            boundary_cache_bytes: 0,
            coverage: MemoryCoverageReceipt {
                interval_micros: MEMORY_SAMPLE_INTERVAL_MICROS,
                sample_count: periodic_sample_count + 2,
                periodic_sample_count,
                sampling_span_micros,
                max_gap_micros: MEMORY_SAMPLE_INTERVAL_MICROS,
            },
        }
    }

    fn fixture_campaign() -> (FrozenCampaign, tempfile::TempDir) {
        let mut matrix = committed_matrix();
        let root = tempfile::tempdir().unwrap();
        let mut bindings = Vec::new();
        for contract in &mut matrix.artifacts {
            let path = root
                .path()
                .join(&contract.key)
                .join(&contract.resolved_revision)
                .join(contract.tier.as_str());
            fs::create_dir_all(&path).unwrap();
            fs::write(path.join("weights.bin"), contract.key.as_bytes()).unwrap();
            let inventory = inventory_artifact(&path).unwrap();
            contract.inventory_sha256 = inventory.sha256;
            contract.inventory_file_count = inventory.file_count;
            contract.inventory_total_bytes = inventory.total_bytes;
            bindings.push(ArtifactBinding {
                key: contract.key.clone(),
                repository: contract.repository.clone(),
                resolved_revision: contract.resolved_revision.clone(),
                tier: contract.tier,
                path,
            });
        }
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
            benchmark_id: matrix.benchmark_id.clone(),
            artifacts: bindings,
        };
        let providers: BTreeSet<_> = matrix
            .cases
            .iter()
            .map(|case| case.provider.clone())
            .collect();
        let provider_capabilities = providers
            .into_iter()
            .map(|provider| ProviderCapabilityReceipt {
                provider,
                available_toggles: OptimizationToggle::ALL.to_vec(),
            })
            .collect();
        let selected_variants = matrix
            .variants
            .iter()
            .map(|variant| variant.id.clone())
            .collect();
        let campaign = FrozenCampaign::freeze(
            matrix,
            &manifest,
            selected_variants,
            BuildProvenance {
                source_revision: "a".repeat(40),
                mlx_revision: "b".repeat(40),
                source_dirty: false,
            },
            HostIdentity {
                rustc_version: "rustc test".to_owned(),
                os_version: "macOS test".to_owned(),
                hardware_model: "Mac test".to_owned(),
                metal_device: "Apple test".to_owned(),
            },
            provider_capabilities,
            1,
        )
        .unwrap();
        (campaign, root)
    }

    fn valid_run(campaign: &FrozenCampaign, case_id: &str, variant_id: &str) -> RunRecord {
        let case = campaign.matrix.case(case_id).unwrap();
        let artifact = campaign.artifact(&case.artifact_key).unwrap().clone();
        let variant = campaign.matrix.variant(variant_id).unwrap().clone();
        let diagnostics: Vec<DiagnosticRecord> = variant
            .toggles
            .iter()
            .map(|toggle| DiagnosticRecord {
                domain: "toggle".to_owned(),
                site: toggle.as_str().to_owned(),
                outcome: "applied".to_owned(),
                count: 7,
                reason: None,
            })
            .collect();
        let encode_ns = 1_000_000_000;
        let denoise_ns = 2_000_000_000;
        let decode_ns = 1_000_000_000;
        let total_ns = encode_ns + denoise_ns + decode_ns;
        let digest = "d".repeat(64);
        let measurements = (0..campaign.matrix.measured_runs)
            .map(|repetition| MeasurementRecord {
                repetition,
                total_elapsed_nanos: total_ns,
                total_seconds: total_ns as f64 / 1e9,
                denoise_steps_per_second: case.steps as f64 / (denoise_ns as f64 / 1e9),
                progress: ProgressReceipt {
                    steps: (1..=case.steps)
                        .map(|current| StepReceipt {
                            current,
                            total: case.steps,
                            elapsed_nanos: encode_ns
                                + (u64::from(current) * denoise_ns / u64::from(case.steps + 1)),
                        })
                        .collect(),
                    decoding_elapsed_nanos: encode_ns + denoise_ns,
                },
                phase_boundaries: vec![
                    PhaseBoundaryReceipt {
                        boundary: PhaseBoundary::DenoiseStart,
                        elapsed_nanos: encode_ns,
                    },
                    PhaseBoundaryReceipt {
                        boundary: PhaseBoundary::DecodeStart,
                        elapsed_nanos: encode_ns + denoise_ns,
                    },
                ],
                phases: PhaseSet {
                    encode: phase(1.0, 100),
                    denoise: phase(2.0, 200),
                    decode: phase(1.0, 300),
                },
                output: OutputFingerprint {
                    kind: if case.family == BenchmarkFamily::WanVideo {
                        "video".to_owned()
                    } else {
                        "images".to_owned()
                    },
                    items: if case.family == BenchmarkFamily::WanVideo {
                        case.frames
                    } else {
                        1
                    },
                    width: case.width,
                    height: case.height,
                    payload_bytes: 1,
                    sha256: digest.clone(),
                },
                diagnostics: diagnostics.clone(),
            })
            .collect();
        RunRecord {
            schema_version: RUN_SCHEMA_VERSION.to_owned(),
            campaign_id: campaign.campaign_id.clone(),
            benchmark_id: campaign.matrix.benchmark_id.clone(),
            case_id: case.id.clone(),
            family: case.family,
            provider: case.provider.clone(),
            artifact: artifact.clone(),
            variant,
            request: request_receipt(campaign, case, &artifact).unwrap(),
            build: campaign.build.clone(),
            host: campaign.host.clone(),
            available_toggles: campaign.capabilities(&case.provider).unwrap().to_vec(),
            started_at_unix_millis: 1,
            load_seconds: 1.0,
            load_active_peak_bytes: 0,
            load_cache_bytes_after_load: 0,
            warmup_runs_completed: campaign.matrix.warmup_runs,
            measurements,
        }
    }

    #[test]
    fn committed_matrix_is_complete_and_content_pinned() {
        let matrix = committed_matrix();
        matrix.validate().unwrap();
        assert_eq!(matrix.cases.len(), 9);
        assert_eq!(matrix.variants.len(), 7);
        assert_eq!(matrix.artifacts.len(), 3);
        assert!(matrix
            .artifacts
            .iter()
            .all(|artifact| is_sha256(&artifact.inventory_sha256)));
    }

    #[test]
    fn old_fixture_sentinels_cannot_satisfy_committed_artifact_contracts() {
        let matrix = committed_matrix();
        let root = tempfile::tempdir().unwrap();
        let artifacts = matrix
            .artifacts
            .iter()
            .map(|contract| {
                let path = root
                    .path()
                    .join(&contract.resolved_revision)
                    .join(contract.tier.as_str());
                fs::create_dir_all(&path).unwrap();
                fs::write(path.join("config.json"), b"fixture").unwrap();
                ArtifactBinding {
                    key: contract.key.clone(),
                    repository: contract.repository.clone(),
                    resolved_revision: contract.resolved_revision.clone(),
                    tier: contract.tier,
                    path,
                }
            })
            .collect();
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
            benchmark_id: matrix.benchmark_id.clone(),
            artifacts,
        };
        assert!(freeze_artifacts(&matrix, &manifest)
            .unwrap_err()
            .to_string()
            .contains("inventory does not match revision"));
    }

    #[test]
    fn required_all_preflight_rejects_unavailable_provider_toggle() {
        let (mut campaign, _root) = fixture_campaign();
        campaign.provider_capabilities[0]
            .available_toggles
            .retain(|toggle| *toggle != OptimizationToggle::ExactEpilogues);
        campaign.campaign_id = sha256_json(&CampaignIdentityMaterial {
            created_at_unix_millis: campaign.created_at_unix_millis,
            mode: campaign.mode,
            selected_variants: &campaign.selected_variants,
            matrix_sha256: &campaign.matrix_sha256,
            input_artifact_manifest_sha256: &campaign.input_artifact_manifest_sha256,
            artifact_set_sha256: &campaign.artifact_set_sha256,
            build: &campaign.build,
            host: &campaign.host,
            provider_capabilities: &campaign.provider_capabilities,
        })
        .unwrap();
        assert!(campaign
            .validate()
            .unwrap_err()
            .to_string()
            .contains("capability is unavailable"));
    }

    #[test]
    fn baseline_only_campaign_is_runnable_but_not_acceptance_complete() {
        let (campaign, _root) = fixture_campaign();
        let matrix = campaign.matrix.clone();
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
            benchmark_id: matrix.benchmark_id.clone(),
            artifacts: campaign
                .artifacts
                .iter()
                .map(|artifact| ArtifactBinding {
                    key: artifact.key.clone(),
                    repository: artifact.repository.clone(),
                    resolved_revision: artifact.resolved_revision.clone(),
                    tier: artifact.tier,
                    path: artifact.canonical_path.clone(),
                })
                .collect(),
        };
        let baseline = FrozenCampaign::freeze(
            matrix,
            &manifest,
            vec!["baseline".to_owned()],
            campaign.build.clone(),
            campaign.host.clone(),
            campaign
                .provider_capabilities
                .iter()
                .map(|receipt| ProviderCapabilityReceipt {
                    provider: receipt.provider.clone(),
                    available_toggles: Vec::new(),
                })
                .collect(),
            2,
        )
        .unwrap();
        assert_eq!(baseline.mode, CampaignMode::BaselineOnly);
        let records: Vec<_> = baseline
            .matrix
            .cases
            .iter()
            .map(|case| valid_run(&baseline, &case.id, "baseline"))
            .collect();
        let summary = build_summary(&baseline, &records).unwrap();
        assert!(!summary.acceptance_complete);
    }

    #[test]
    fn run_rejects_wrong_build_request_model_or_campaign_identity() {
        let (campaign, _root) = fixture_campaign();
        let mut run = valid_run(&campaign, "qwen-q4-512", "baseline");
        run.build.source_revision = "e".repeat(40);
        run.request.case.seed += 1;
        run.artifact.inventory.sha256 = "f".repeat(64);
        assert!(run
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("identity"));
    }

    #[test]
    fn progress_rejects_duplicate_out_of_order_and_post_decode_steps() {
        let (campaign, _root) = fixture_campaign();
        let mut duplicate = valid_run(&campaign, "qwen-q4-512", "baseline");
        duplicate.measurements[0].progress.steps[1].current = 1;
        assert!(duplicate
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("exact monotone 1..N"));

        let mut post_decode = valid_run(&campaign, "qwen-q4-512", "baseline");
        let decode = post_decode.measurements[0].progress.decoding_elapsed_nanos;
        post_decode.measurements[0].progress.steps[7].elapsed_nanos = decode;
        assert!(post_decode
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("exact monotone 1..N"));

        let mut before_denoise = valid_run(&campaign, "qwen-q4-512", "baseline");
        before_denoise.measurements[0].progress.steps[0].elapsed_nanos = 1;
        assert!(before_denoise
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("explicit denoise phase"));
    }

    #[test]
    fn explicit_phase_boundaries_drive_all_step_rate_and_stage_durations() {
        let (campaign, _root) = fixture_campaign();
        let run = valid_run(&campaign, "qwen-q4-512", "baseline");
        run.validate_against(&campaign).unwrap();
        assert_eq!(run.measurements[0].phases.encode.seconds, 1.0);
        assert_eq!(run.measurements[0].phases.denoise.seconds, 2.0);
        assert_eq!(run.measurements[0].phases.decode.seconds, 1.0);
        assert_eq!(
            run.measurements[0].denoise_steps_per_second,
            run.request.case.steps as f64 / 2.0
        );
    }

    #[test]
    fn exact_set_toggle_receipts_reject_baseline_leak_fallback_and_unrequested_applied() {
        let (campaign, _root) = fixture_campaign();
        let mut baseline = valid_run(&campaign, "qwen-q4-512", "baseline");
        baseline.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "toggle".to_owned(),
            site: "retained_compilation".to_owned(),
            outcome: "applied".to_owned(),
            count: 1,
            reason: None,
        });
        assert!(baseline
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("baseline forbids"));

        let mut retained = valid_run(&campaign, "qwen-q4-512", "retained_compilation");
        retained.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "toggle".to_owned(),
            site: "retained_compilation".to_owned(),
            outcome: "fallback".to_owned(),
            count: 1,
            reason: Some("controlled".to_owned()),
        });
        retained.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "toggle".to_owned(),
            site: "exact_epilogues".to_owned(),
            outcome: "applied".to_owned(),
            count: 1,
            reason: None,
        });
        let error = retained
            .validate_against(&campaign)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exactly one terminal Applied"));
        assert!(error.contains("unrequested toggle"));
    }

    #[test]
    fn memory_receipt_rejects_fake_sum_and_inadequate_coverage() {
        let (campaign, _root) = fixture_campaign();
        let mut run = valid_run(&campaign, "qwen-q4-512", "baseline");
        let phase = &mut run.measurements[0].phases.denoise;
        phase.sampled_active_peak_bytes = 100;
        phase.native_active_peak_bytes = 100;
        phase.sampled_cache_peak_bytes = 90;
        phase.sampled_footprint_peak_bytes = 190;
        phase.footprint_peak_active_bytes = 100;
        phase.footprint_peak_cache_bytes = 0;
        phase.boundary_active_bytes = 0;
        phase.boundary_cache_bytes = 0;
        let error = run.validate_against(&campaign).unwrap_err().to_string();
        assert!(error.contains("paired-footprint witness"));

        let phase = &mut run.measurements[0].phases.denoise;
        phase.footprint_peak_cache_bytes = 90;
        phase.coverage.sampling_span_micros = 2_000_000;
        phase.coverage.periodic_sample_count = 0;
        phase.coverage.sample_count = 2;
        phase.coverage.max_gap_micros = 2_000_000;
        let error = run.validate_against(&campaign).unwrap_err().to_string();
        assert!(error.contains("fixed-cadence sampling coverage"));
    }

    #[test]
    fn summary_rejects_cross_variant_digest_changes_and_binds_same_sample_footprint() {
        let (campaign, _root) = fixture_campaign();
        let mut records = Vec::new();
        for case in &campaign.matrix.cases {
            for variant in &campaign.selected_variants {
                records.push(valid_run(&campaign, &case.id, variant));
            }
        }
        {
            let changed = records
                .iter_mut()
                .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "exact_epilogues")
                .unwrap();
            for measurement in &mut changed.measurements {
                measurement.output.sha256 = "e".repeat(64);
            }
        }
        assert!(build_summary(&campaign, &records)
            .unwrap_err()
            .to_string()
            .contains("differs from baseline"));

        {
            let changed = records
                .iter_mut()
                .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "exact_epilogues")
                .unwrap();
            for measurement in &mut changed.measurements {
                measurement.output.sha256 = "d".repeat(64);
            }
        }
        let summary = build_summary(&campaign, &records).unwrap();
        assert_eq!(summary.binding_rule, MEMORY_BINDING_RULE);
        assert!(summary.acceptance_complete);
        assert!(summary
            .rows
            .iter()
            .all(|row| row.binding_phase == BindingPhase::Decode));
    }

    #[test]
    fn mixed_or_stale_records_are_rejected() {
        let (campaign, _root) = fixture_campaign();
        let mut records = Vec::new();
        for case in &campaign.matrix.cases {
            for variant in &campaign.selected_variants {
                records.push(valid_run(&campaign, &case.id, variant));
            }
        }
        records[0].host.hardware_model = "different host".to_owned();
        records[1].campaign_id = "f".repeat(64);
        assert!(build_summary(&campaign, &records)
            .unwrap_err()
            .to_string()
            .contains("campaign/build/host identity"));
    }

    #[test]
    fn legacy_v1_run_cannot_deserialize_as_v2() {
        let value = serde_json::json!({
            "schemaVersion": "sceneworks.mlx-perf-run.v1",
            "benchmarkId": "legacy"
        });
        assert!(serde_json::from_value::<RunRecord>(value).is_err());
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let raw = include_str!("../benchmarks/mlx-perf-matrix-v1.json");
        let mut value: serde_json::Value = serde_json::from_str(raw).unwrap();
        value["invented"] = serde_json::json!(true);
        assert!(serde_json::from_value::<BenchmarkMatrix>(value)
            .unwrap_err()
            .to_string()
            .contains("unknown field `invented`"));
    }
}

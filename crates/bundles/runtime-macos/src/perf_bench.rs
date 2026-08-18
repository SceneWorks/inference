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

pub const MATRIX_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-matrix.v4";
pub const ARTIFACT_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-artifacts.v2";
pub const CAMPAIGN_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-campaign.v2";
pub const RUN_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-run.v4";
pub const ARTIFACT_SNAPSHOT_FORMAT: &str = "sceneworks.private-artifact-snapshot.v1";
pub const SUMMARY_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-summary.v3";
pub const INVENTORY_ALGORITHM: &str = "sha256-tree-content-v1";
/// P5's decode-allocation transients are measured in tens of milliseconds. Sample substantially
/// faster than that event and reject a sampler that was descheduled for more than three ticks.
pub const MEMORY_SAMPLE_INTERVAL_MICROS: u64 = 10_000;
pub const MEMORY_MAX_GAP_MULTIPLIER: u64 = 3;
pub const MEMORY_BINDING_RULE: &str = "median_peak_same_sample_active_plus_cache";

const CANONICAL_MATRIX_JSON: &str = include_str!("../benchmarks/mlx-perf-matrix-v1.json");

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

/// Request shape held constant around one optimization comparison.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecodeControlMode {
    /// Production/default decode selection. P9 may change this policy when its toggle is requested.
    Default,
    /// Force the matrix's fixed tile geometry through the benchmark-only request scope.
    FixedTiled,
}

/// Correctness relation between a variant and its declared control.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputComparison {
    /// Repetitions must be stable; no cross-variant byte relation is asserted for this control row.
    Deterministic,
    /// Every output byte must equal the declared control variant.
    ExactControl,
    /// P9 may differ from its control only when its explicit production-evidence-backed decision
    /// selected geometry tiling; an unchanged decision remains byte-exact.
    GeometryAwareControl,
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
    pub decode_control: DecodeControlMode,
    pub control_variant: String,
    pub output_comparison: OutputComparison,
}

/// Fixed tile geometry used only to isolate P5's accumulator mechanics from P9's policy.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TiledDecodeControl {
    pub spatial_tile_px: u32,
    pub spatial_overlap_px: u32,
    pub temporal_tile_frames: Option<u32>,
    pub temporal_overlap_frames: Option<u32>,
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
    /// Exact production retained-compile sites that P1 must exercise for this provider.
    pub expected_p1_compile_operations: Vec<String>,
    /// Exact production operation identities that P3 must apply at least once for this provider.
    /// The request may additionally carry truthful per-shape fallback receipts.
    pub expected_p3_exact_epilogue_operations: Vec<String>,
    pub artifact_key: String,
    pub repository: String,
    pub tier: ModelTier,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub steps: u32,
    pub seed: u64,
    pub prompt: String,
    pub tiled_decode_control: TiledDecodeControl,
}

const WAN_P1_COMPILE_OPERATIONS: &[&str] = &[
    "wan::rope::rope_rotate",
    "wan::transformer::gated",
    "wan::transformer::gelu_ffn",
    "wan::transformer::modulate",
];
const QWEN_P1_COMPILE_OPERATIONS: &[&str] = &[
    "qwen_image::attention::rope_rotate",
    "qwen_image::block::gated",
    "qwen_image::block::modulate",
    "qwen_image::feed_forward::gelu_ffn",
];
const SDXL_P1_COMPILE_OPERATIONS: &[&str] = &[
    "mlx_gen::nn::gelu_exact",
    "mlx_gen::nn::gelu_quick",
    "sdxl::silu_glue",
];
// These inventories cover the full request, not only denoise. Wan's staged UMT5/embed-text path
// reaches the shared eager tanh-GELU helper, while SDXL's VAE reaches shared eager SiLU even though
// its UNet SiLU glue is already compiled by P1.
const WAN_P3_EXACT_EPILOGUE_OPERATIONS: &[&str] = &[
    "conv2d_bias",
    "conv3d_bias",
    "gelu_tanh",
    "quantized_matmul_bias",
    "silu",
];
const QWEN_P3_EXACT_EPILOGUE_OPERATIONS: &[&str] = &[
    "conv2d_bias",
    "conv3d_bias",
    "quantized_matmul_bias",
    "silu",
];
const SDXL_P3_EXACT_EPILOGUE_OPERATIONS: &[&str] = &["conv2d_bias", "group_norm_affine", "silu"];

fn expected_p1_compile_operations(provider: &str) -> Option<&'static [&'static str]> {
    match provider {
        "wan2_2_ti2v_5b" => Some(WAN_P1_COMPILE_OPERATIONS),
        "qwen_image" => Some(QWEN_P1_COMPILE_OPERATIONS),
        "sdxl" => Some(SDXL_P1_COMPILE_OPERATIONS),
        _ => None,
    }
}

fn expected_p3_exact_epilogue_operations(provider: &str) -> Option<&'static [&'static str]> {
    match provider {
        "wan2_2_ti2v_5b" => Some(WAN_P3_EXACT_EPILOGUE_OPERATIONS),
        "qwen_image" => Some(QWEN_P3_EXACT_EPILOGUE_OPERATIONS),
        "sdxl" => Some(SDXL_P3_EXACT_EPILOGUE_OPERATIONS),
        _ => None,
    }
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
    pub fn canonical() -> Self {
        serde_json::from_str(CANONICAL_MATRIX_JSON)
            .expect("the embedded P6 canonical matrix must deserialize")
    }

    /// Acceptance is tied to the exact committed matrix, not merely to a structurally valid custom
    /// matrix. Equality binds prompts, models, tiers, artifacts, steps, seeds, geometries, controls,
    /// and variant semantics.
    pub fn is_canonical_acceptance_matrix(&self) -> bool {
        self == &Self::canonical()
    }

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
        let mut tiled_control_count = 0;
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
                if !variant.toggles.is_empty()
                    || variant.decode_control != DecodeControlMode::Default
                    || variant.control_variant != "baseline"
                    || variant.output_comparison != OutputComparison::Deterministic
                {
                    errors.push(
                        "baseline must be deterministic production/default decode with no toggles"
                            .to_owned(),
                    );
                }
            } else if variant.id == "tiled_decode_control" {
                tiled_control_count += 1;
                if !variant.toggles.is_empty()
                    || variant.decode_control != DecodeControlMode::FixedTiled
                    || variant.control_variant != "tiled_decode_control"
                    || variant.output_comparison != OutputComparison::Deterministic
                {
                    errors.push(
                        "tiled_decode_control must be a deterministic, toggle-free fixed-tile control"
                            .to_owned(),
                    );
                }
            } else if variant.id == "all_on" {
                all_on_count += 1;
                if unique != all_toggles {
                    errors
                        .push("all_on must contain every benchmark toggle exactly once".to_owned());
                }
                if variant.decode_control != DecodeControlMode::Default
                    || variant.control_variant != "geometry_aware_decode"
                    || variant.output_comparison != OutputComparison::ExactControl
                {
                    errors.push(
                        "all_on must preserve the geometry-aware control bytes while adding every exact toggle"
                            .to_owned(),
                    );
                }
            } else if variant.toggles.len() == 1 {
                let toggle = variant.toggles[0];
                if !singleton_toggles.insert(toggle) {
                    errors.push(format!(
                        "matrix repeats the independent {:?} toggle variant",
                        toggle
                    ));
                }
                if variant.id != toggle.as_str() {
                    errors.push(format!(
                        "singleton variant {:?} must use its toggle's stable id {:?}",
                        variant.id,
                        toggle.as_str()
                    ));
                }
                let expected = match toggle {
                    OptimizationToggle::IndexedDecodeAccumulator => (
                        DecodeControlMode::FixedTiled,
                        "tiled_decode_control",
                        OutputComparison::ExactControl,
                    ),
                    OptimizationToggle::GeometryAwareDecode => (
                        DecodeControlMode::Default,
                        "baseline",
                        OutputComparison::GeometryAwareControl,
                    ),
                    OptimizationToggle::RetainedCompilation
                    | OptimizationToggle::ExactEpilogues
                    | OptimizationToggle::FusedAttentionPrimitives => (
                        DecodeControlMode::Default,
                        "baseline",
                        OutputComparison::ExactControl,
                    ),
                };
                if (
                    variant.decode_control,
                    variant.control_variant.as_str(),
                    variant.output_comparison,
                ) != expected
                {
                    errors.push(format!(
                        "variant {:?} has the wrong decode/control/correctness contract",
                        variant.id
                    ));
                }
            } else {
                errors.push(format!(
                    "variant {:?} must be baseline, tiled_decode_control, one independent toggle, or all_on",
                    variant.id
                ));
            }
        }
        if self.variants.len() != 8
            || baseline_count != 1
            || tiled_control_count != 1
            || all_on_count != 1
            || singleton_toggles != all_toggles
        {
            errors.push(
                "matrix must contain baseline, tiled_decode_control, one independent variant per toggle, and all_on"
                    .to_owned(),
            );
        }
        for variant in &self.variants {
            let Some(control) = self.variant(&variant.control_variant) else {
                errors.push(format!(
                    "variant {:?} references missing control {:?}",
                    variant.id, variant.control_variant
                ));
                continue;
            };
            if variant.output_comparison == OutputComparison::ExactControl
                && variant.decode_control != control.decode_control
            {
                errors.push(format!(
                    "exact variant {:?} must use the same decode control as {:?}",
                    variant.id, control.id
                ));
            }
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
            let tiled = case.tiled_decode_control;
            if tiled.spatial_tile_px == 0
                || tiled.spatial_overlap_px == 0
                || tiled.spatial_overlap_px >= tiled.spatial_tile_px
                || tiled.spatial_tile_px > i32::MAX as u32
                || tiled.spatial_overlap_px > i32::MAX as u32
                || tiled.temporal_tile_frames.is_some() != tiled.temporal_overlap_frames.is_some()
                || tiled
                    .temporal_tile_frames
                    .is_some_and(|tile| tile == 0 || tile > i32::MAX as u32)
                || tiled.temporal_overlap_frames.is_some_and(|overlap| {
                    overlap == 0
                        || overlap > i32::MAX as u32
                        || tiled
                            .temporal_tile_frames
                            .is_some_and(|tile| overlap >= tile)
                })
            {
                errors.push(format!(
                    "case {:?} has an invalid fixed tiled-decode control",
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
            let expected_p1 = expected_p1_compile_operations(&case.provider);
            let declared_p1: Vec<_> = case
                .expected_p1_compile_operations
                .iter()
                .map(String::as_str)
                .collect();
            if expected_p1 != Some(declared_p1.as_slice()) {
                errors.push(format!(
                    "case {:?} must freeze the exact P1 compile-operation inventory for provider {:?}",
                    case.id, case.provider
                ));
            }
            let expected_p3 = expected_p3_exact_epilogue_operations(&case.provider);
            let declared_p3: Vec<_> = case
                .expected_p3_exact_epilogue_operations
                .iter()
                .map(String::as_str)
                .collect();
            if expected_p3 != Some(declared_p3.as_slice()) {
                errors.push(format!(
                    "case {:?} must freeze the exact P3 epilogue-operation inventory for provider {:?}",
                    case.id, case.provider
                ));
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
                    if tiled.temporal_tile_frames.is_none() {
                        errors.push(format!(
                            "Wan case {:?} requires a temporal fixed-tile control",
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
                    if tiled.temporal_tile_frames.is_some() {
                        errors.push(format!(
                            "image-DiT case {:?} must not carry temporal tiling",
                            case.id
                        ));
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
                    if tiled.temporal_tile_frames.is_some() {
                        errors.push(format!(
                            "SDXL case {:?} must not carry temporal tiling",
                            case.id
                        ));
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactSnapshotReceipt {
    pub format: String,
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
    pub cargo_profile: String,
    pub opt_level: String,
    pub debug_assertions: bool,
    pub target_triple: String,
    pub cargo_features: Vec<String>,
    pub target_features: Vec<String>,
    pub rustflags: Vec<String>,
    pub rustc_version: String,
    pub executable_sha256: String,
}

impl BuildProvenance {
    fn validate(&self, errors: &mut Vec<String>) {
        if !is_lower_hex_revision(&self.source_revision)
            || !is_lower_hex_revision(&self.mlx_revision)
            || self.source_dirty
            || self.cargo_profile.trim().is_empty()
            || self.opt_level.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.rustc_version.trim().is_empty()
            || !is_sha256(&self.executable_sha256)
            || self
                .cargo_features
                .windows(2)
                .any(|window| window[0] >= window[1])
            || self
                .target_features
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            errors.push(
                "campaign requires a clean, exact source/dependency/executable build receipt"
                    .to_owned(),
            );
        }
    }

    /// The documented acceptance build. Diagnostic campaigns may use other fully recorded builds,
    /// but a debug/custom-codegen executable must never set `acceptanceComplete`.
    pub fn is_acceptance_build(&self) -> bool {
        self.cargo_profile == "release"
            && self.opt_level == "3"
            && !self.debug_assertions
            && self.target_triple == "aarch64-apple-darwin"
            && self
                .cargo_features
                .iter()
                .map(String::as_str)
                .eq(["media", "perf-bench"])
            && self.rustflags.is_empty()
    }
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
    for id in selected {
        let Some(variant) = matrix.variant(id) else {
            continue;
        };
        if !unique.contains(variant.control_variant.as_str()) {
            errors.push(format!(
                "selected variant {:?} requires its control {:?}",
                variant.id, variant.control_variant
            ));
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
        if self.created_at_unix_millis == 0 {
            errors.push("campaign requires a nonzero creation timestamp".to_owned());
        }
        self.build.validate(&mut errors);
        if self.mode == CampaignMode::RequiredAll
            && self.matrix.is_canonical_acceptance_matrix()
            && !self.build.is_acceptance_build()
        {
            errors.push(
                "the canonical required-all campaign requires the documented release build"
                    .to_owned(),
            );
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

    pub fn acceptance_complete(&self) -> bool {
        self.mode == CampaignMode::RequiredAll
            && self.matrix.is_canonical_acceptance_matrix()
            && self.build.is_acceptance_build()
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
    pub decode_path: Option<String>,
    pub production_evidence_sha256: Option<String>,
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
    pub artifact_snapshot: ArtifactSnapshotReceipt,
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
    let observed_intervals = coverage.sample_count.saturating_sub(1);
    let minimum_possible_max_gap = if observed_intervals == 0 {
        u64::MAX
    } else {
        coverage.sampling_span_micros.div_ceil(observed_intervals)
    };
    if coverage.max_gap_micros > MEMORY_MAX_GAP_MULTIPLIER * MEMORY_SAMPLE_INTERVAL_MICROS
        || coverage.max_gap_micros > coverage.sampling_span_micros
        || coverage.max_gap_micros < minimum_possible_max_gap
    {
        errors.push(format!(
            "measurement {name} phase lacks fixed-cadence sampling coverage"
        ));
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum GeometryDecodeDecision {
    Unchanged,
    GeometryTiled,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PhysicalDecodePath {
    Dense,
    Tiled,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GeometryDecodeReceipt {
    decision: GeometryDecodeDecision,
    decode_path: PhysicalDecodePath,
    production_evidence_sha256: Option<String>,
}

fn geometry_decode_receipt(diagnostics: &[DiagnosticRecord]) -> Option<GeometryDecodeReceipt> {
    let record = diagnostics
        .iter()
        .find(|record| record.domain == "decode_policy")?;
    let decision = match record.outcome.as_str() {
        "unchanged" => GeometryDecodeDecision::Unchanged,
        "geometry_tiled" => GeometryDecodeDecision::GeometryTiled,
        _ => return None,
    };
    let decode_path = match record.decode_path.as_deref() {
        Some("dense") => PhysicalDecodePath::Dense,
        Some("tiled") => PhysicalDecodePath::Tiled,
        _ => return None,
    };
    Some(GeometryDecodeReceipt {
        decision,
        decode_path,
        production_evidence_sha256: record.production_evidence_sha256.clone(),
    })
}

fn validate_geometry_decode_receipt(
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
    errors: &mut Vec<String>,
) {
    let requested = variant
        .toggles
        .contains(&OptimizationToggle::GeometryAwareDecode);
    let records: Vec<_> = diagnostics
        .iter()
        .filter(|record| record.domain == "decode_policy")
        .collect();
    if !requested {
        if !records.is_empty() {
            errors.push(format!(
                "non-P9 variant {:?} forbids decode_policy receipts",
                variant.id
            ));
        }
        return;
    }
    if records.len() != 1 {
        errors.push(format!(
            "P9 variant {:?} requires exactly one decode_policy receipt",
            variant.id
        ));
        return;
    }
    let record = records[0];
    if record.site != OptimizationToggle::GeometryAwareDecode.as_str()
        || record.count != 1
        || record.reason.is_some()
    {
        errors.push(format!(
            "P9 variant {:?} has an invalid decode_policy identity/count",
            variant.id
        ));
        return;
    }
    match record.outcome.as_str() {
        "unchanged"
            if matches!(record.decode_path.as_deref(), Some("dense" | "tiled"))
                && record.production_evidence_sha256.is_none() => {}
        "geometry_tiled"
            if record.decode_path.as_deref() == Some("tiled")
                && record
                .production_evidence_sha256
                .as_deref()
                .is_some_and(is_sha256) => {}
        "geometry_tiled" => errors.push(format!(
            "P9 variant {:?} requires physical tiled decode and a production-evidence SHA-256 for geometry_tiled",
            variant.id
        )),
        "unchanged" => errors.push(format!(
            "P9 variant {:?} requires a physical decode path and forbids production evidence for unchanged",
            variant.id
        )),
        outcome => errors.push(format!(
            "P9 variant {:?} has unknown decode_policy outcome {outcome:?}",
            variant.id
        )),
    }
}

fn validate_exact_epilogue_receipts(
    case: &WorkloadCase,
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
    errors: &mut Vec<String>,
) {
    let requested = variant
        .toggles
        .contains(&OptimizationToggle::ExactEpilogues);
    let records: Vec<_> = diagnostics
        .iter()
        .filter(|record| record.domain == "exact_epilogue")
        .collect();
    if !requested {
        if !records.is_empty() {
            errors.push(format!(
                "variant {:?} emitted exact_epilogue receipts without requesting P3",
                variant.id
            ));
        }
        return;
    }

    let expected: BTreeSet<_> = case
        .expected_p3_exact_epilogue_operations
        .iter()
        .map(String::as_str)
        .collect();
    for record in &records {
        if !expected.contains(record.site.as_str()) {
            errors.push(format!(
                "P3 case {:?} emitted an operation outside its exact inventory: {:?}",
                case.id, record.site
            ));
        }
        match record.outcome.as_str() {
            "applied" if record.reason.is_none() => {}
            "fallback" | "unavailable" if record.reason.is_some() => {}
            outcome => errors.push(format!(
                "P3 operation {:?} has an invalid {outcome:?} receipt/reason pairing",
                record.site
            )),
        }
    }

    let actual_applied: BTreeSet<_> = records
        .iter()
        .filter(|record| record.outcome == "applied")
        .map(|record| record.site.as_str())
        .collect();
    if actual_applied != expected {
        errors.push(format!(
            "P3 Applied receipts for case {:?} must exactly cover {:?}, got {:?}",
            case.id, expected, actual_applied
        ));
    }
    for operation in expected {
        let applied: Vec<_> = records
            .iter()
            .filter(|record| record.site == operation && record.outcome == "applied")
            .collect();
        if applied.len() != 1 || applied[0].count == 0 || applied[0].reason.is_some() {
            errors.push(format!(
                "P3 operation {operation:?} requires one positive Applied receipt without a reason"
            ));
        }
    }
}

fn validate_toggle_receipts(
    case: &WorkloadCase,
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
    errors: &mut Vec<String>,
) {
    let declared: BTreeSet<_> = variant
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
            diagnostic.production_evidence_sha256.as_deref(),
        )) {
            errors
                .push("diagnostic records must be aggregated into an exact unique set".to_owned());
        }
        if diagnostic.domain != "decode_policy" && diagnostic.production_evidence_sha256.is_some() {
            errors.push("only decode_policy may carry a production-evidence SHA-256".to_owned());
        }
        if diagnostic.domain != "decode_policy" && diagnostic.decode_path.is_some() {
            errors.push("only decode_policy may carry a physical decode path".to_owned());
        }
    }
    validate_geometry_decode_receipt(variant, diagnostics, errors);
    validate_exact_epilogue_receipts(case, variant, diagnostics, errors);
    let all_on_p5_inactive = variant.id == "all_on"
        && geometry_decode_receipt(diagnostics)
            .is_some_and(|receipt| receipt.decode_path == PhysicalDecodePath::Dense);
    let mut expected = declared.clone();
    if all_on_p5_inactive {
        expected.remove(OptimizationToggle::IndexedDecodeAccumulator.as_str());
    }
    let toggle_records: Vec<_> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.domain == "toggle")
        .collect();
    if declared.is_empty() {
        if !toggle_records.is_empty() {
            errors.push("toggle-free variants forbid every toggle terminal receipt".to_owned());
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
        if !declared.contains(record.site.as_str()) {
            errors.push(format!(
                "variant {:?} emitted an unrequested toggle receipt for {}",
                variant.id, record.site
            ));
        }
    }
    if all_on_p5_inactive
        && toggle_records
            .iter()
            .any(|record| record.site == OptimizationToggle::IndexedDecodeAccumulator.as_str())
    {
        errors.push(
            "all_on forbids an indexed_decode_accumulator terminal receipt for a physical dense decode path"
                .to_owned(),
        );
    }
    for toggle in expected {
        let records: Vec<_> = toggle_records
            .iter()
            .filter(|record| record.site == toggle)
            .collect();
        if records.len() != 1
            || records[0].outcome != "applied"
            || records[0].count == 0
            || records[0].reason.is_some()
            || records[0].production_evidence_sha256.is_some()
        {
            errors.push(format!(
                "variant {:?} requires exactly one terminal Applied record and no fallback/unavailable for {toggle}",
                variant.id
            ));
        }
    }

    if variant
        .toggles
        .contains(&OptimizationToggle::RetainedCompilation)
    {
        let expected: BTreeSet<_> = case
            .expected_p1_compile_operations
            .iter()
            .map(String::as_str)
            .collect();
        let compile_records: Vec<_> = diagnostics
            .iter()
            .filter(|record| record.domain == "compile")
            .collect();
        let actual: BTreeSet<_> = compile_records
            .iter()
            .map(|record| record.site.as_str())
            .collect();
        if actual != expected {
            errors.push(format!(
                "P1 compile receipts for case {:?} must exactly cover {:?}, got {:?}",
                case.id, expected, actual
            ));
        }
        if compile_records
            .iter()
            .any(|record| record.outcome == "one_shot" || record.reason.is_some())
        {
            errors.push(
                "P1 forbids one-shot/fallback compile receipts when retention is requested"
                    .to_owned(),
            );
        }
        for site in expected {
            for outcome in ["retained_miss", "retained_hit"] {
                let records: Vec<_> = compile_records
                    .iter()
                    .filter(|record| record.site == site && record.outcome == outcome)
                    .collect();
                if records.len() != 1 || records[0].count == 0 || records[0].reason.is_some() {
                    errors.push(format!(
                        "P1 operation {site:?} requires one positive {outcome} receipt in every request"
                    ));
                }
            }
            if compile_records.iter().any(|record| {
                record.site == site
                    && !matches!(record.outcome.as_str(), "retained_miss" | "retained_hit")
            }) {
                errors.push(format!(
                    "P1 operation {site:?} emitted a non-retained compile outcome"
                ));
            }
        }
    }
}

/// Validate one request's complete, aggregated diagnostic set against its selected variant.
///
/// The runner applies this to warmups as well as measured repetitions so an unavailable or
/// silently-fallback toggle aborts the campaign before it can consume the remaining matrix.
pub fn validate_toggle_diagnostics(
    case: &WorkloadCase,
    variant: &VariantPlan,
    diagnostics: &[DiagnosticRecord],
) -> Result<(), ContractError> {
    let mut errors = Vec::new();
    validate_toggle_receipts(case, variant, diagnostics, &mut errors);
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
        if self.artifact_snapshot.format != ARTIFACT_SNAPSHOT_FORMAT
            || self.artifact_snapshot.inventory != expected_artifact.inventory
        {
            errors.push(
                "run artifact snapshot receipt does not match the exact frozen artifact content"
                    .to_owned(),
            );
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
            validate_toggle_receipts(case, &self.variant, &measurement.diagnostics, &mut errors);
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
    pub control_variant_id: String,
    pub speedup_vs_control: f64,
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

fn stable_geometry_decode_receipt(
    record: &RunRecord,
) -> Result<GeometryDecodeReceipt, ContractError> {
    let receipts: BTreeSet<_> = record
        .measurements
        .iter()
        .filter_map(|measurement| geometry_decode_receipt(&measurement.diagnostics))
        .collect();
    if receipts.len() != 1 || record.measurements.is_empty() {
        return Err(ContractError::new(vec![format!(
            "case {} variant {} has an unstable or missing decode_policy receipt",
            record.case_id, record.variant.id
        )]));
    }
    Ok(receipts
        .into_iter()
        .next()
        .expect("one stable decode-policy receipt"))
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
        for variant_id in &campaign.selected_variants {
            let Some(plan) = campaign.matrix.variant(variant_id) else {
                continue;
            };
            let Some(control) = by_key.get(&(case.id.as_str(), plan.control_variant.as_str()))
            else {
                continue;
            };
            let Some(record) = by_key.get(&(case.id.as_str(), variant_id.as_str())) else {
                continue;
            };
            let control_digest = &control.measurements[0].output.sha256;
            match plan.output_comparison {
                OutputComparison::Deterministic => {}
                OutputComparison::ExactControl => {
                    if record
                        .measurements
                        .iter()
                        .any(|measurement| measurement.output.sha256 != *control_digest)
                    {
                        errors.push(format!(
                            "case {} variant {} output digest differs from exact control {}",
                            case.id, variant_id, plan.control_variant
                        ));
                    }
                    if plan
                        .toggles
                        .contains(&OptimizationToggle::GeometryAwareDecode)
                    {
                        match (
                            stable_geometry_decode_receipt(record),
                            stable_geometry_decode_receipt(control),
                        ) {
                            (Ok(actual), Ok(expected)) if actual == expected => {}
                            (Ok(_), Ok(_)) => errors.push(format!(
                                "case {} variant {} decode_policy differs from control {}",
                                case.id, variant_id, plan.control_variant
                            )),
                            (Err(error), _) | (_, Err(error)) => errors.extend(error.messages),
                        }
                    }
                }
                OutputComparison::GeometryAwareControl => {
                    match stable_geometry_decode_receipt(record) {
                        Ok(receipt)
                            if receipt.decision == GeometryDecodeDecision::Unchanged
                                && record.measurements.iter().any(|measurement| {
                                    measurement.output.sha256 != *control_digest
                                }) =>
                        {
                            errors.push(format!(
                                "case {} variant {} claimed unchanged decode_policy but differs from control {}",
                                case.id, variant_id, plan.control_variant
                            ));
                        }
                        Ok(_) => {}
                        Err(error) => errors.extend(error.messages),
                    }
                }
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
            let plan = campaign
                .matrix
                .variant(variant_id)
                .expect("selected variant belongs to the validated matrix");
            let control = by_key[&(case.id.as_str(), plan.control_variant.as_str())];
            let control_total = median_f64(
                control
                    .measurements
                    .iter()
                    .map(|measurement| measurement.total_seconds)
                    .collect(),
            );
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
                control_variant_id: plan.control_variant.clone(),
                speedup_vs_control: control_total / total,
                speedup_vs_baseline: baseline_total / total,
            });
        }
    }
    Ok(BenchmarkSummary {
        schema_version: SUMMARY_SCHEMA_VERSION.to_owned(),
        campaign_id: campaign.campaign_id.clone(),
        benchmark_id: campaign.matrix.benchmark_id.clone(),
        mode: campaign.mode,
        acceptance_complete: campaign.acceptance_complete(),
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

    fn fixture_build() -> BuildProvenance {
        BuildProvenance {
            source_revision: "a".repeat(40),
            mlx_revision: "b".repeat(40),
            source_dirty: false,
            cargo_profile: "release".to_owned(),
            opt_level: "3".to_owned(),
            debug_assertions: false,
            target_triple: "aarch64-apple-darwin".to_owned(),
            cargo_features: vec!["media".to_owned(), "perf-bench".to_owned()],
            target_features: vec!["neon".to_owned()],
            rustflags: Vec::new(),
            rustc_version: "rustc test".to_owned(),
            executable_sha256: "c".repeat(64),
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
            fixture_build(),
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

    /// Synthetic diagnostic fixture used only to exercise the frozen P6 contract. It must never
    /// be serialized as benchmark evidence; real run records are produced only from provider
    /// diagnostics in `mlx_perf_bench::measure_request`.
    fn diagnostic_contract_fixture(
        case: &WorkloadCase,
        variant: &VariantPlan,
    ) -> Vec<DiagnosticRecord> {
        let mut diagnostics: Vec<_> = variant
            .toggles
            .iter()
            .filter(|toggle| {
                !(variant.id == "all_on"
                    && **toggle == OptimizationToggle::IndexedDecodeAccumulator)
            })
            .map(|toggle| DiagnosticRecord {
                domain: "toggle".to_owned(),
                site: toggle.as_str().to_owned(),
                outcome: "applied".to_owned(),
                count: 7,
                reason: None,
                decode_path: None,
                production_evidence_sha256: None,
            })
            .collect();
        if variant
            .toggles
            .contains(&OptimizationToggle::GeometryAwareDecode)
        {
            diagnostics.push(DiagnosticRecord {
                domain: "decode_policy".to_owned(),
                site: OptimizationToggle::GeometryAwareDecode.as_str().to_owned(),
                outcome: "unchanged".to_owned(),
                count: 1,
                reason: None,
                decode_path: Some("dense".to_owned()),
                production_evidence_sha256: None,
            });
        }
        if variant
            .toggles
            .contains(&OptimizationToggle::RetainedCompilation)
        {
            for site in &case.expected_p1_compile_operations {
                for (outcome, count) in [("retained_miss", 1), ("retained_hit", 7)] {
                    diagnostics.push(DiagnosticRecord {
                        domain: "compile".to_owned(),
                        site: site.clone(),
                        outcome: outcome.to_owned(),
                        count,
                        reason: None,
                        decode_path: None,
                        production_evidence_sha256: None,
                    });
                }
            }
        }
        if variant
            .toggles
            .contains(&OptimizationToggle::ExactEpilogues)
        {
            for operation in &case.expected_p3_exact_epilogue_operations {
                diagnostics.push(DiagnosticRecord {
                    domain: "exact_epilogue".to_owned(),
                    site: operation.clone(),
                    outcome: "applied".to_owned(),
                    count: 7,
                    reason: None,
                    decode_path: None,
                    production_evidence_sha256: None,
                });
            }
        }
        diagnostics
    }

    fn valid_run(campaign: &FrozenCampaign, case_id: &str, variant_id: &str) -> RunRecord {
        let case = campaign.matrix.case(case_id).unwrap();
        let artifact = campaign.artifact(&case.artifact_key).unwrap().clone();
        let variant = campaign.matrix.variant(variant_id).unwrap().clone();
        let diagnostics = diagnostic_contract_fixture(case, &variant);
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
            artifact_snapshot: ArtifactSnapshotReceipt {
                format: ARTIFACT_SNAPSHOT_FORMAT.to_owned(),
                inventory: artifact.inventory.clone(),
            },
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

    fn set_geometry_decode_policy(
        run: &mut RunRecord,
        outcome: &str,
        decode_path: &str,
        production_evidence_sha256: Option<&str>,
    ) {
        let indexed = OptimizationToggle::IndexedDecodeAccumulator.as_str();
        let all_on = run.variant.id == "all_on";
        for measurement in &mut run.measurements {
            let receipt = measurement
                .diagnostics
                .iter_mut()
                .find(|record| record.domain == "decode_policy")
                .expect("P9 fixture has a decode-policy receipt");
            receipt.outcome = outcome.to_owned();
            receipt.decode_path = Some(decode_path.to_owned());
            receipt.production_evidence_sha256 = production_evidence_sha256.map(str::to_owned);
            if all_on && decode_path == "tiled" {
                if !measurement
                    .diagnostics
                    .iter()
                    .any(|record| record.domain == "toggle" && record.site == indexed)
                {
                    measurement.diagnostics.push(DiagnosticRecord {
                        domain: "toggle".to_owned(),
                        site: indexed.to_owned(),
                        outcome: "applied".to_owned(),
                        count: 7,
                        reason: None,
                        decode_path: None,
                        production_evidence_sha256: None,
                    });
                }
            } else if all_on {
                measurement
                    .diagnostics
                    .retain(|record| !(record.domain == "toggle" && record.site == indexed));
            }
        }
    }

    #[test]
    fn committed_matrix_is_complete_and_content_pinned() {
        let matrix = committed_matrix();
        matrix.validate().unwrap();
        // The `cases.len() == 9` / `variants.len() == 8` / `artifacts.len() == 3` pins that used to
        // sit here were frozen populations of what `validate()` above already states as a contract,
        // and they were the stricter, wrong end of it: `validate` derives the variant total from
        // `OptimizationToggle::ALL` plus the baseline/tiled-control/all_on trio, requires
        // `referenced_artifacts == artifact_keys` so the artifact table is exactly what the cases
        // cite, and allows 2-3 cases per family. Pinning 9 therefore forbade a family legitimately
        // carrying 2. (`is_canonical_acceptance_matrix` cannot stand in for them either: `canonical`
        // and `committed_matrix` both parse the same embedded JSON, so here it is a tautology — its
        // real work is in `structurally_valid_custom_matrix_is_never_acceptance_canonical`.)
        assert!(matrix.is_canonical_acceptance_matrix());
        assert!(matrix
            .artifacts
            .iter()
            .all(|artifact| is_sha256(&artifact.inventory_sha256)));
    }

    #[test]
    fn structurally_valid_custom_matrix_is_never_acceptance_canonical() {
        let mut matrix = committed_matrix();
        matrix.cases[0].prompt.push_str(" diagnostic override");

        matrix.validate().unwrap();
        assert!(!matrix.is_canonical_acceptance_matrix());
    }

    #[test]
    fn matrix_requires_fixed_tiled_control_for_the_p5_comparison() {
        let matrix = committed_matrix();
        assert!(validate_selection(
            &matrix,
            &[
                "baseline".to_owned(),
                "indexed_decode_accumulator".to_owned(),
            ]
        )
        .unwrap_err()
        .to_string()
        .contains("requires its control \"tiled_decode_control\""));

        let mut invalid = matrix;
        invalid
            .variants
            .iter_mut()
            .find(|variant| variant.id == "indexed_decode_accumulator")
            .unwrap()
            .decode_control = DecodeControlMode::Default;
        assert!(invalid
            .validate()
            .unwrap_err()
            .to_string()
            .contains("wrong decode/control/correctness contract"));
    }

    #[test]
    fn acceptance_build_is_exact_and_fully_receipted() {
        let build = fixture_build();
        assert!(build.is_acceptance_build());

        let mutations: [fn(&mut BuildProvenance); 6] = [
            |build: &mut BuildProvenance| build.cargo_profile = "dev".to_owned(),
            |build: &mut BuildProvenance| build.opt_level = "2".to_owned(),
            |build: &mut BuildProvenance| build.debug_assertions = true,
            |build: &mut BuildProvenance| build.target_triple = "x86_64-apple-darwin".to_owned(),
            |build: &mut BuildProvenance| build.cargo_features.push("audio".to_owned()),
            |build: &mut BuildProvenance| build.rustflags.push("-Ctarget-cpu=native".to_owned()),
        ];
        for mutate in mutations {
            let mut changed = build.clone();
            mutate(&mut changed);
            assert!(!changed.is_acceptance_build());
        }

        let mut invalid = build;
        invalid.executable_sha256 = "not-a-digest".to_owned();
        let mut errors = Vec::new();
        invalid.validate(&mut errors);
        assert!(errors
            .iter()
            .any(|error| error.contains("exact source/dependency/executable build receipt")));
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

        let mut wrong_snapshot = valid_run(&campaign, "qwen-q4-512", "baseline");
        wrong_snapshot.artifact_snapshot.inventory.sha256 = "f".repeat(64);
        assert!(wrong_snapshot
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("snapshot receipt"));
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
            decode_path: None,
            production_evidence_sha256: None,
        });
        assert!(baseline
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("toggle-free variants forbid"));

        let mut retained = valid_run(&campaign, "qwen-q4-512", "retained_compilation");
        retained.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "toggle".to_owned(),
            site: "retained_compilation".to_owned(),
            outcome: "fallback".to_owned(),
            count: 1,
            reason: Some("controlled".to_owned()),
            decode_path: None,
            production_evidence_sha256: None,
        });
        retained.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "toggle".to_owned(),
            site: "exact_epilogues".to_owned(),
            outcome: "applied".to_owned(),
            count: 1,
            reason: None,
            decode_path: None,
            production_evidence_sha256: None,
        });
        let error = retained
            .validate_against(&campaign)
            .unwrap_err()
            .to_string();
        assert!(error.contains("exactly one terminal Applied"));
        assert!(error.contains("unrequested toggle"));

        let mut missing_operation = valid_run(&campaign, "qwen-q4-512", "retained_compilation");
        missing_operation.measurements[0]
            .diagnostics
            .retain(|record| {
                !(record.domain == "compile"
                    && record.site == "qwen_image::attention::rope_rotate"
                    && record.outcome == "retained_hit")
            });
        assert!(missing_operation
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("requires one positive retained_hit"));

        let mut one_shot = valid_run(&campaign, "qwen-q4-512", "retained_compilation");
        one_shot.measurements[0].diagnostics.push(DiagnosticRecord {
            domain: "compile".to_owned(),
            site: "qwen_image::block::gated".to_owned(),
            outcome: "one_shot".to_owned(),
            count: 1,
            reason: None,
            decode_path: None,
            production_evidence_sha256: None,
        });
        let error = one_shot
            .validate_against(&campaign)
            .unwrap_err()
            .to_string();
        assert!(error.contains("forbids one-shot"));
        assert!(error.contains("non-retained compile outcome"));
    }

    #[test]
    fn p3_requires_the_exact_applied_inventory_and_allows_truthful_per_shape_fallback() {
        let (campaign, _root) = fixture_campaign();
        let mut with_fallback = valid_run(&campaign, "qwen-q4-512", "exact_epilogues");
        with_fallback.measurements[0]
            .diagnostics
            .push(DiagnosticRecord {
                domain: "exact_epilogue".to_owned(),
                site: "conv2d_bias".to_owned(),
                outcome: "fallback".to_owned(),
                count: 2,
                reason: Some("unsupported_dtype_shape_or_dispatch".to_owned()),
                decode_path: None,
                production_evidence_sha256: None,
            });
        with_fallback.validate_against(&campaign).unwrap();

        let mut incomplete = valid_run(&campaign, "qwen-q4-512", "exact_epilogues");
        incomplete.measurements[0].diagnostics.retain(|record| {
            !(record.domain == "exact_epilogue"
                && record.site == "conv3d_bias"
                && record.outcome == "applied")
        });
        assert!(incomplete
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("must exactly cover"));

        let mut reasonless = valid_run(&campaign, "qwen-q4-512", "exact_epilogues");
        reasonless.measurements[0]
            .diagnostics
            .push(DiagnosticRecord {
                domain: "exact_epilogue".to_owned(),
                site: "conv2d_bias".to_owned(),
                outcome: "fallback".to_owned(),
                count: 1,
                reason: None,
                decode_path: None,
                production_evidence_sha256: None,
            });
        assert!(reasonless
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("invalid \"fallback\" receipt/reason pairing"));
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

        let mut run = valid_run(&campaign, "qwen-q4-512", "baseline");
        run.measurements[0].phases.denoise.coverage.max_gap_micros =
            (MEMORY_MAX_GAP_MULTIPLIER + 1) * MEMORY_SAMPLE_INTERVAL_MICROS;
        let error = run.validate_against(&campaign).unwrap_err().to_string();
        assert!(error.contains("fixed-cadence sampling coverage"));
    }

    #[test]
    fn sampling_gap_gate_applies_to_short_measured_phases() {
        let mut acceptable = phase(0.025, 100);
        acceptable.coverage.sample_count = 2;
        acceptable.coverage.periodic_sample_count = 0;
        acceptable.coverage.sampling_span_micros = 25_000;
        acceptable.coverage.max_gap_micros = 25_000;
        let mut errors = Vec::new();
        validate_phase("short", &acceptable, &mut errors);
        assert!(errors.is_empty(), "{errors:?}");

        let mut starved = phase(0.035, 100);
        starved.coverage.sample_count = 2;
        starved.coverage.periodic_sample_count = 0;
        starved.coverage.sampling_span_micros = 35_000;
        starved.coverage.max_gap_micros = 35_000;
        let mut errors = Vec::new();
        validate_phase("short", &starved, &mut errors);
        assert!(errors
            .iter()
            .any(|error| error.contains("fixed-cadence sampling coverage")));

        // A serialized receipt cannot conceal that same starvation by claiming a max gap smaller
        // than the average gap implied by its span and sample count.
        starved.coverage.max_gap_micros = 1;
        let mut errors = Vec::new();
        validate_phase("short", &starved, &mut errors);
        assert!(errors
            .iter()
            .any(|error| error.contains("fixed-cadence sampling coverage")));
    }

    #[test]
    fn all_on_p5_receipt_tracks_the_actual_physical_decode_path() {
        let (campaign, _root) = fixture_campaign();
        let mut unchanged = valid_run(&campaign, "qwen-q4-512", "all_on");
        unchanged.validate_against(&campaign).unwrap();
        assert!(!unchanged.measurements.iter().any(|measurement| {
            measurement.diagnostics.iter().any(|record| {
                record.domain == "toggle"
                    && record.site == OptimizationToggle::IndexedDecodeAccumulator.as_str()
            })
        }));

        for measurement in &mut unchanged.measurements {
            measurement.diagnostics.push(DiagnosticRecord {
                domain: "toggle".to_owned(),
                site: OptimizationToggle::IndexedDecodeAccumulator
                    .as_str()
                    .to_owned(),
                outcome: "applied".to_owned(),
                count: 1,
                reason: None,
                decode_path: None,
                production_evidence_sha256: None,
            });
        }
        assert!(unchanged
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("forbids an indexed_decode_accumulator terminal receipt"));

        let mut tiled = valid_run(&campaign, "qwen-q4-512", "all_on");
        set_geometry_decode_policy(&mut tiled, "geometry_tiled", "tiled", Some(&"9".repeat(64)));
        tiled.validate_against(&campaign).unwrap();
        for measurement in &mut tiled.measurements {
            measurement.diagnostics.retain(|record| {
                !(record.domain == "toggle"
                    && record.site == OptimizationToggle::IndexedDecodeAccumulator.as_str())
            });
        }
        assert!(tiled
            .validate_against(&campaign)
            .unwrap_err()
            .to_string()
            .contains("requires exactly one terminal Applied record"));

        // Wan may preserve its production policy while that pre-existing policy auto-tiles. The
        // physical path, not P9's semantic disposition, is therefore authoritative for P5.
        let mut unchanged_but_tiled = valid_run(&campaign, "wan-q4-512x480-f17", "all_on");
        set_geometry_decode_policy(&mut unchanged_but_tiled, "unchanged", "tiled", None);
        unchanged_but_tiled.validate_against(&campaign).unwrap();
    }

    #[test]
    fn summary_enforces_declared_controls_and_accepts_evidence_backed_p9_drift() {
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
            .contains("differs from exact control baseline"));

        {
            let changed = records
                .iter_mut()
                .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "exact_epilogues")
                .unwrap();
            for measurement in &mut changed.measurements {
                measurement.output.sha256 = "d".repeat(64);
            }
        }
        for variant_id in ["geometry_aware_decode", "all_on"] {
            let changed = records
                .iter_mut()
                .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == variant_id)
                .unwrap();
            set_geometry_decode_policy(changed, "geometry_tiled", "tiled", Some(&"9".repeat(64)));
            for measurement in &mut changed.measurements {
                measurement.output.sha256 = "e".repeat(64);
            }
        }
        for variant_id in ["tiled_decode_control", "indexed_decode_accumulator"] {
            let changed = records
                .iter_mut()
                .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == variant_id)
                .unwrap();
            for measurement in &mut changed.measurements {
                measurement.output.sha256 = "f".repeat(64);
            }
        }
        let summary = build_summary(&campaign, &records).unwrap();
        assert_eq!(summary.binding_rule, MEMORY_BINDING_RULE);
        assert!(!summary.acceptance_complete);
        assert_eq!(
            summary
                .rows
                .iter()
                .find(|row| {
                    row.case_id == "qwen-q4-512" && row.variant_id == "indexed_decode_accumulator"
                })
                .unwrap()
                .control_variant_id,
            "tiled_decode_control"
        );
        assert_eq!(
            summary
                .rows
                .iter()
                .find(|row| row.case_id == "qwen-q4-512" && row.variant_id == "all_on")
                .unwrap()
                .control_variant_id,
            "geometry_aware_decode"
        );
        assert!(summary
            .rows
            .iter()
            .all(|row| row.binding_phase == BindingPhase::Decode));
    }

    #[test]
    fn summary_rejects_missing_mismatched_and_false_p9_policy_receipts() {
        let (campaign, _root) = fixture_campaign();
        let records = || {
            let mut records = Vec::new();
            for case in &campaign.matrix.cases {
                for variant in &campaign.selected_variants {
                    records.push(valid_run(&campaign, &case.id, variant));
                }
            }
            records
        };

        let mut missing = records();
        missing
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap()
            .measurements[0]
            .diagnostics
            .retain(|record| record.domain != "decode_policy");
        assert!(build_summary(&campaign, &missing)
            .unwrap_err()
            .to_string()
            .contains("requires exactly one decode_policy receipt"));

        let mut false_unchanged = records();
        let p9 = false_unchanged
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        for measurement in &mut p9.measurements {
            measurement.output.sha256 = "e".repeat(64);
        }
        assert!(build_summary(&campaign, &false_unchanged)
            .unwrap_err()
            .to_string()
            .contains("claimed unchanged decode_policy but differs from control baseline"));

        let mut no_evidence = records();
        let p9 = no_evidence
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        set_geometry_decode_policy(p9, "geometry_tiled", "tiled", None);
        assert!(build_summary(&campaign, &no_evidence)
            .unwrap_err()
            .to_string()
            .contains("production-evidence SHA-256 for geometry_tiled"));

        let mut impossible_dense_geometry = records();
        let p9 = impossible_dense_geometry
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        set_geometry_decode_policy(p9, "geometry_tiled", "dense", Some(&"8".repeat(64)));
        assert!(build_summary(&campaign, &impossible_dense_geometry)
            .unwrap_err()
            .to_string()
            .contains("requires physical tiled decode"));

        let mut missing_path = records();
        let p9 = missing_path
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        for measurement in &mut p9.measurements {
            measurement
                .diagnostics
                .iter_mut()
                .find(|record| record.domain == "decode_policy")
                .unwrap()
                .decode_path = None;
        }
        assert!(build_summary(&campaign, &missing_path)
            .unwrap_err()
            .to_string()
            .contains("requires a physical decode path"));

        let mut mismatched = records();
        let p9 = mismatched
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        set_geometry_decode_policy(p9, "geometry_tiled", "tiled", Some(&"8".repeat(64)));
        let all_on = mismatched
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "all_on")
            .unwrap();
        set_geometry_decode_policy(all_on, "geometry_tiled", "tiled", Some(&"9".repeat(64)));
        assert!(build_summary(&campaign, &mismatched)
            .unwrap_err()
            .to_string()
            .contains("decode_policy differs from control geometry_aware_decode"));

        let mut unstable = records();
        let p9 = unstable
            .iter_mut()
            .find(|run| run.case_id == "qwen-q4-512" && run.variant.id == "geometry_aware_decode")
            .unwrap();
        let receipt = p9.measurements[0]
            .diagnostics
            .iter_mut()
            .find(|record| record.domain == "decode_policy")
            .unwrap();
        receipt.outcome = "geometry_tiled".to_owned();
        receipt.decode_path = Some("tiled".to_owned());
        receipt.production_evidence_sha256 = Some("7".repeat(64));
        assert!(build_summary(&campaign, &unstable)
            .unwrap_err()
            .to_string()
            .contains("unstable or missing decode_policy receipt"));
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
    fn legacy_v1_run_cannot_deserialize_as_current_schema() {
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

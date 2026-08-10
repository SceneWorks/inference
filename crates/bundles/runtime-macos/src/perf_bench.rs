//! Stable workload and result contracts for the cross-family MLX performance harness (sc-18321).
//!
//! The real-weight runner lives in the opt-in `mlx-perf-bench` binary.  These types stay in the
//! platform bundle so the committed matrix, child-process records, and final comparison table all
//! pass through one strict schema instead of ad-hoc log parsing.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;

pub const MATRIX_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-matrix.v1";
pub const ARTIFACT_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-artifacts.v1";
pub const RUN_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-run.v1";
pub const SUMMARY_SCHEMA_VERSION: &str = "sceneworks.mlx-perf-summary.v1";

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
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct VariantPlan {
    pub id: String,
    pub toggles: Vec<OptimizationToggle>,
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
    messages: Vec<String>,
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

fn is_lower_hex_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
        if self.warmup_runs == 0 {
            errors.push("matrix.warmupRuns must be at least 1".to_owned());
        }
        if self.measured_runs < 2 {
            errors.push("matrix.measuredRuns must be at least 2".to_owned());
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
                singleton_toggles.insert(variant.toggles[0]);
            } else {
                errors.push(format!(
                    "variant {:?} must be baseline, one independent toggle, or all_on",
                    variant.id
                ));
            }
        }
        if baseline_count != 1 {
            errors.push("matrix must contain exactly one baseline variant".to_owned());
        }
        if all_on_count != 1 {
            errors.push("matrix must contain exactly one all_on variant".to_owned());
        }
        if singleton_toggles != all_toggles {
            errors.push("matrix must contain one independent variant for every toggle".to_owned());
        }

        let mut case_ids = BTreeSet::new();
        let mut counts = BTreeMap::new();
        let mut decode_bound = BTreeSet::new();
        for case in &self.cases {
            if case.id.trim().is_empty() || !case_ids.insert(case.id.as_str()) {
                errors.push(format!(
                    "matrix case ids must be non-empty and unique (got {:?})",
                    case.id
                ));
            }
            *counts.entry(case.family).or_insert(0usize) += 1;
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
            match case.family {
                BenchmarkFamily::WanVideo => {
                    if !case.provider.starts_with("wan") || case.frames <= 1 {
                        errors.push(format!(
                            "Wan case {:?} must use a Wan provider and a multi-frame clip",
                            case.id
                        ));
                    }
                    if !(480..=1280).contains(&case.width)
                        || !(480..=1280).contains(&case.height)
                        || !case.width.is_multiple_of(32)
                        || !case.height.is_multiple_of(32)
                        || u64::from(case.width) * u64::from(case.height) > 1280 * 704
                        || case.frames % 4 != 1
                        || case.frames > 1025
                    {
                        errors.push(format!(
                            "Wan case {:?} must satisfy the TI2V-5B contract: each dimension \
                             480..=1280 on the 32-pixel grid, area <= 1280x704, and frames=1+4k \
                             through 1025",
                            case.id
                        ));
                    }
                }
                BenchmarkFamily::ImageDit => {
                    if !(case.provider.starts_with("qwen_image")
                        || case.provider.starts_with("krea_2"))
                    {
                        errors.push(format!(
                            "image-DiT case {:?} must use Qwen-Image or Krea-2",
                            case.id
                        ));
                    }
                    if case.frames != 1 {
                        errors.push(format!("image case {:?} must have frames=1", case.id));
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
        let mut expected = BTreeMap::new();
        for case in &matrix.cases {
            let identity = (case.repository.as_str(), case.tier);
            if let Some(previous) = expected.insert(case.artifact_key.as_str(), identity) {
                if previous != identity {
                    errors.push(format!(
                        "artifact key {:?} maps to conflicting repository/tier identities",
                        case.artifact_key
                    ));
                }
            }
        }
        let mut actual = BTreeSet::new();
        for artifact in &self.artifacts {
            if !actual.insert(artifact.key.as_str()) {
                errors.push(format!("artifact key {:?} is duplicated", artifact.key));
            }
            match expected.get(artifact.key.as_str()) {
                Some((repository, tier))
                    if *repository != artifact.repository || *tier != artifact.tier =>
                {
                    errors.push(format!(
                        "artifact {:?} repository/tier {:?}:{} does not match matrix {:?}:{}",
                        artifact.key,
                        artifact.repository,
                        artifact.tier.as_str(),
                        repository,
                        tier.as_str()
                    ))
                }
                None => errors.push(format!(
                    "artifact {:?} is not referenced by the matrix",
                    artifact.key
                )),
                _ => {}
            }
            if !is_lower_hex_revision(&artifact.resolved_revision) {
                errors.push(format!(
                    "artifact {:?} resolvedRevision must be a 40-character lowercase git SHA",
                    artifact.key
                ));
            }
            if !artifact.path.is_absolute() || !artifact.path.is_dir() {
                errors.push(format!(
                    "artifact {:?} path must be an existing absolute directory: {}",
                    artifact.key,
                    artifact.path.display()
                ));
            } else if artifact.path.file_name().and_then(|name| name.to_str())
                != Some(artifact.tier.as_str())
            {
                errors.push(format!(
                    "artifact {:?} path must end in its exact tier {:?}: {}",
                    artifact.key,
                    artifact.tier.as_str(),
                    artifact.path.display()
                ));
            } else if artifact
                .path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                != Some(artifact.resolved_revision.as_str())
            {
                errors.push(format!(
                    "artifact {:?} path must be the tier under resolved revision {:?}: {}",
                    artifact.key,
                    artifact.resolved_revision,
                    artifact.path.display()
                ));
            }
            if let Some(case) = matrix
                .cases
                .iter()
                .find(|case| case.artifact_key == artifact.key)
            {
                for relative in required_artifact_files(case) {
                    if !artifact.path.join(relative).is_file() {
                        errors.push(format!(
                            "artifact {:?} is incomplete: missing {relative}",
                            artifact.key
                        ));
                    }
                }
            }
        }
        for key in expected.keys() {
            if !actual.contains(key) {
                errors.push(format!("artifact manifest is missing key {key:?}"));
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(ContractError::new(errors))
        }
    }

    pub fn artifact(&self, key: &str) -> Option<&ArtifactBinding> {
        self.artifacts.iter().find(|artifact| artifact.key == key)
    }
}

fn required_artifact_files(case: &WorkloadCase) -> &'static [&'static str] {
    match case.provider.as_str() {
        "wan2_2_ti2v_5b" => &[
            "config.json",
            "model.safetensors",
            "t5_encoder.safetensors",
            "vae.safetensors",
            "tokenizer.json",
        ],
        "qwen_image" => &[
            "model_index.json",
            "transformer/model.safetensors",
            "text_encoder/model.safetensors.index.json",
            "vae/diffusion_pytorch_model.safetensors",
            "tokenizer/tokenizer.json",
        ],
        "sdxl" if case.tier == ModelTier::Q4 || case.tier == ModelTier::Q8 => &[
            "model_index.json",
            "unet/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "tokenizer/vocab.json",
            "tokenizer_2/vocab.json",
        ],
        "sdxl" => &[
            "model_index.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "tokenizer/vocab.json",
            "tokenizer_2/vocab.json",
        ],
        _ => &[],
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct EnvironmentRecord {
    pub inference_revision: String,
    pub mlx_revision: String,
    pub rustc_version: String,
    pub os_version: String,
    pub hardware_model: String,
    pub metal_device: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactReceipt {
    pub key: String,
    pub repository: String,
    pub resolved_revision: String,
    pub tier: ModelTier,
    pub canonical_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PhaseMetrics {
    pub seconds: f64,
    pub active_peak_bytes: u64,
    pub cache_peak_bytes: u64,
    pub cache_bytes_at_boundary: u64,
    pub samples: u32,
}

impl PhaseMetrics {
    pub fn allocator_peak_bytes(&self) -> u64 {
        self.active_peak_bytes.saturating_add(self.cache_peak_bytes)
    }
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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
    pub total_seconds: f64,
    pub denoise_steps_per_second: f64,
    pub step_events: u32,
    pub saw_decode: bool,
    pub phases: PhaseSet,
    pub output: OutputFingerprint,
    pub diagnostics: Vec<DiagnosticRecord>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunRecord {
    pub schema_version: String,
    pub benchmark_id: String,
    pub case_id: String,
    pub family: BenchmarkFamily,
    pub provider: String,
    pub artifact: ArtifactReceipt,
    pub variant: VariantPlan,
    pub environment: EnvironmentRecord,
    pub started_at_unix_millis: u128,
    pub load_seconds: f64,
    pub load_active_peak_bytes: u64,
    pub load_cache_bytes_after_load: u64,
    pub warmup_runs_completed: u32,
    pub measurements: Vec<MeasurementRecord>,
}

impl RunRecord {
    pub fn validate_against(&self, matrix: &BenchmarkMatrix) -> Result<(), ContractError> {
        let mut errors = Vec::new();
        if self.schema_version != RUN_SCHEMA_VERSION {
            errors.push(format!("run.schemaVersion must be {RUN_SCHEMA_VERSION:?}"));
        }
        if self.benchmark_id != matrix.benchmark_id {
            errors.push("run benchmarkId does not match the matrix".to_owned());
        }
        let Some(case) = matrix.case(&self.case_id) else {
            return Err(ContractError::new(vec![format!(
                "run references unknown case {:?}",
                self.case_id
            )]));
        };
        if self.family != case.family || self.provider != case.provider {
            errors.push("run family/provider do not match the workload case".to_owned());
        }
        if self.artifact.key != case.artifact_key
            || self.artifact.repository != case.repository
            || self.artifact.tier != case.tier
            || !is_lower_hex_revision(&self.artifact.resolved_revision)
            || !self.artifact.canonical_path.is_absolute()
            || !self.artifact.canonical_path.is_dir()
            || self
                .artifact
                .canonical_path
                .file_name()
                .and_then(|name| name.to_str())
                != Some(case.tier.as_str())
            || self
                .artifact
                .canonical_path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                != Some(self.artifact.resolved_revision.as_str())
        {
            errors.push("run artifact receipt does not match the workload case".to_owned());
        }
        if matrix.variant(&self.variant.id) != Some(&self.variant) {
            errors.push("run variant does not match the matrix".to_owned());
        }
        if self.started_at_unix_millis == 0
            || !(self.load_seconds.is_finite() && self.load_seconds > 0.0)
        {
            errors.push("run must record a timestamp and positive cold-load timing".to_owned());
        }
        if !is_lower_hex_revision(&self.environment.inference_revision)
            || !is_lower_hex_revision(&self.environment.mlx_revision)
            || self.environment.rustc_version.trim().is_empty()
            || self.environment.os_version.trim().is_empty()
            || self.environment.hardware_model.trim().is_empty()
            || self.environment.metal_device.trim().is_empty()
        {
            errors.push(
                "run environment must contain exact revisions and nonempty host identity"
                    .to_owned(),
            );
        }
        if self.warmup_runs_completed != matrix.warmup_runs {
            errors.push(format!(
                "run completed {} warmups, expected {}",
                self.warmup_runs_completed, matrix.warmup_runs
            ));
        }
        if self.measurements.len() != matrix.measured_runs as usize {
            errors.push(format!(
                "run has {} measured repetitions, expected {}",
                self.measurements.len(),
                matrix.measured_runs
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
            if !(measurement.total_seconds.is_finite()
                && measurement.total_seconds > 0.0
                && measurement.denoise_steps_per_second.is_finite()
                && measurement.denoise_steps_per_second > 0.0)
            {
                errors.push("measurement timing must be finite and positive".to_owned());
            }
            if measurement.step_events != case.steps || !measurement.saw_decode {
                errors.push(format!(
                    "measurement must observe exactly {} denoise steps and a decode event",
                    case.steps
                ));
            }
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
            for (name, phase) in [
                ("encode", &measurement.phases.encode),
                ("denoise", &measurement.phases.denoise),
                ("decode", &measurement.phases.decode),
            ] {
                if !(phase.seconds.is_finite() && phase.seconds > 0.0)
                    || phase.active_peak_bytes == 0
                    || phase.samples == 0
                    || phase.cache_bytes_at_boundary > phase.cache_peak_bytes
                {
                    errors.push(format!(
                        "measurement {name} phase must have positive timing, active peak, valid cache samples, and samples"
                    ));
                }
            }
            if measurement
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.count == 0)
            {
                errors.push("diagnostic receipts must have positive counts".to_owned());
            }
            for toggle in &self.variant.toggles {
                let applied = measurement.diagnostics.iter().any(|diagnostic| {
                    diagnostic.domain == "toggle"
                        && diagnostic.site == toggle.as_str()
                        && diagnostic.outcome == "applied"
                        && diagnostic.count > 0
                });
                if !applied {
                    errors.push(format!(
                        "variant {:?} lacks an applied receipt for toggle {}",
                        self.variant.id,
                        toggle.as_str()
                    ));
                }
            }
        }
        if digests.len() > 1 {
            errors.push("measured outputs are not byte-stable for this case/config".to_owned());
        }
        let expected_repetitions: BTreeSet<_> = (0..matrix.measured_runs).collect();
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

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    pub median_denoise_steps_per_second: f64,
    pub median_encode_active_peak_bytes: u64,
    pub median_denoise_active_peak_bytes: u64,
    pub median_decode_active_peak_bytes: u64,
    pub median_encode_cache_peak_bytes: u64,
    pub median_denoise_cache_peak_bytes: u64,
    pub median_decode_cache_peak_bytes: u64,
    pub binding_phase: BindingPhase,
    pub speedup_vs_baseline: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BenchmarkSummary {
    pub schema_version: String,
    pub benchmark_id: String,
    pub rows: Vec<ComparisonRow>,
}

pub fn build_summary(
    matrix: &BenchmarkMatrix,
    records: &[RunRecord],
    selected_variants: &[String],
) -> Result<BenchmarkSummary, ContractError> {
    let mut errors = Vec::new();
    let selected: BTreeSet<_> = selected_variants.iter().map(String::as_str).collect();
    if selected.len() != selected_variants.len() || selected.is_empty() {
        errors.push("selected variants must be non-empty and unique".to_owned());
    }
    if !selected.contains("baseline") {
        errors.push("selected variants must include baseline".to_owned());
    }
    for variant in &selected {
        if matrix.variant(variant).is_none() {
            errors.push(format!("selected unknown variant {variant:?}"));
        }
    }

    let mut by_key = BTreeMap::new();
    for record in records {
        if let Err(error) = record.validate_against(matrix) {
            errors.extend(error.messages);
            continue;
        }
        let key = (record.case_id.as_str(), record.variant.id.as_str());
        if by_key.insert(key, record).is_some() {
            errors.push(format!("duplicate run record for {} / {}", key.0, key.1));
        }
    }
    for case in &matrix.cases {
        for variant in &selected {
            if !by_key.contains_key(&(case.id.as_str(), *variant)) {
                errors.push(format!("missing run record for {} / {}", case.id, variant));
            }
        }
    }
    if !errors.is_empty() {
        return Err(ContractError::new(errors));
    }

    let mut rows = Vec::new();
    for case in &matrix.cases {
        let baseline = by_key[&(case.id.as_str(), "baseline")];
        let baseline_total = median_f64(
            baseline
                .measurements
                .iter()
                .map(|measurement| measurement.total_seconds)
                .collect(),
        );
        for variant in &matrix.variants {
            if !selected.contains(variant.id.as_str()) {
                continue;
            }
            let record = by_key[&(case.id.as_str(), variant.id.as_str())];
            let total = median_f64(
                record
                    .measurements
                    .iter()
                    .map(|measurement| measurement.total_seconds)
                    .collect(),
            );
            let steps_per_second = median_f64(
                record
                    .measurements
                    .iter()
                    .map(|measurement| measurement.denoise_steps_per_second)
                    .collect(),
            );
            let active = |pick: fn(&PhaseSet) -> &PhaseMetrics| {
                median_u64(
                    record
                        .measurements
                        .iter()
                        .map(|measurement| pick(&measurement.phases).active_peak_bytes)
                        .collect(),
                )
            };
            let cache = |pick: fn(&PhaseSet) -> &PhaseMetrics| {
                median_u64(
                    record
                        .measurements
                        .iter()
                        .map(|measurement| pick(&measurement.phases).cache_peak_bytes)
                        .collect(),
                )
            };
            let encode_active = active(|phases| &phases.encode);
            let denoise_active = active(|phases| &phases.denoise);
            let decode_active = active(|phases| &phases.decode);
            let encode_cache = cache(|phases| &phases.encode);
            let denoise_cache = cache(|phases| &phases.denoise);
            let decode_cache = cache(|phases| &phases.decode);
            let binding_phase = [
                (
                    BindingPhase::Encode,
                    encode_active.saturating_add(encode_cache),
                ),
                (
                    BindingPhase::Denoise,
                    denoise_active.saturating_add(denoise_cache),
                ),
                (
                    BindingPhase::Decode,
                    decode_active.saturating_add(decode_cache),
                ),
            ]
            .into_iter()
            .max_by_key(|(_, bytes)| *bytes)
            .expect("three phases")
            .0;
            rows.push(ComparisonRow {
                case_id: case.id.clone(),
                family: case.family,
                tier: case.tier,
                variant_id: variant.id.clone(),
                median_total_seconds: total,
                median_denoise_steps_per_second: steps_per_second,
                median_encode_active_peak_bytes: encode_active,
                median_denoise_active_peak_bytes: denoise_active,
                median_decode_active_peak_bytes: decode_active,
                median_encode_cache_peak_bytes: encode_cache,
                median_denoise_cache_peak_bytes: denoise_cache,
                median_decode_cache_peak_bytes: decode_cache,
                binding_phase,
                speedup_vs_baseline: baseline_total / total,
            });
        }
    }
    Ok(BenchmarkSummary {
        schema_version: SUMMARY_SCHEMA_VERSION.to_owned(),
        benchmark_id: matrix.benchmark_id.clone(),
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

    fn matrix() -> BenchmarkMatrix {
        serde_json::from_str(include_str!("../benchmarks/mlx-perf-matrix-v1.json")).unwrap()
    }

    #[test]
    fn committed_matrix_is_the_complete_cross_family_contract() {
        let matrix = matrix();
        matrix.validate().unwrap();
        assert_eq!(matrix.cases.len(), 9);
        assert_eq!(matrix.variants.len(), 7);
    }

    #[test]
    fn matrix_rejects_a_missing_decode_bound_family() {
        let mut matrix = matrix();
        matrix.cases.retain(|case| {
            !(case.family == BenchmarkFamily::SdxlUnet && case.width >= 2048 && case.height >= 2048)
        });
        let error = matrix.validate().unwrap_err().to_string();
        assert!(error.contains("sdxl_unet must contain a decode-bound geometry"));
    }

    #[test]
    fn matrix_rejects_a_collapsed_toggle_cross_product() {
        let mut matrix = matrix();
        matrix
            .variants
            .retain(|variant| variant.id != "retained_compilation");
        assert!(matrix
            .validate()
            .unwrap_err()
            .to_string()
            .contains("one independent variant for every toggle"));
    }

    #[test]
    fn matrix_rejects_wan_geometry_outside_the_provider_contract() {
        let mut matrix = matrix();
        let wan = matrix
            .cases
            .iter_mut()
            .find(|case| case.family == BenchmarkFamily::WanVideo)
            .unwrap();
        wan.height = 320;
        let error = matrix.validate().unwrap_err().to_string();
        assert!(error.contains("must satisfy the TI2V-5B contract"));
    }

    #[test]
    fn artifact_manifest_is_exact_and_requires_real_paths_and_revisions() {
        let matrix = matrix();
        let root = tempfile::tempdir().unwrap();
        let artifacts = [
            ("wan_q4", "SceneWorks/wan2.2-ti2v-5b-mlx"),
            ("qwen_image_q4", "SceneWorks/qwen-image-mlx"),
            ("sdxl_bf16", "SceneWorks/sdxl-base-mlx"),
        ]
        .into_iter()
        .map(|(key, repository)| {
            let tier = if key == "sdxl_bf16" {
                ModelTier::Bf16
            } else {
                ModelTier::Q4
            };
            let path = root
                .path()
                .join(key)
                .join("a".repeat(40))
                .join(tier.as_str());
            fs::create_dir_all(&path).unwrap();
            ArtifactBinding {
                key: key.to_owned(),
                repository: repository.to_owned(),
                resolved_revision: "a".repeat(40),
                tier,
                path,
            }
        })
        .collect();
        let manifest = ArtifactManifest {
            schema_version: ARTIFACT_SCHEMA_VERSION.to_owned(),
            benchmark_id: matrix.benchmark_id.clone(),
            artifacts,
        };
        for case in &matrix.cases {
            let artifact = manifest.artifact(&case.artifact_key).unwrap();
            for relative in required_artifact_files(case) {
                let path = artifact.path.join(relative);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(path, b"fixture").unwrap();
            }
        }
        manifest.validate_against(&matrix).unwrap();

        let mut stale = manifest;
        stale.artifacts[0].resolved_revision = "main".to_owned();
        assert!(stale
            .validate_against(&matrix)
            .unwrap_err()
            .to_string()
            .contains("40-character lowercase git SHA"));
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let raw = include_str!("../benchmarks/mlx-perf-matrix-v1.json");
        let mut value: serde_json::Value = serde_json::from_str(raw).unwrap();
        value["invented"] = serde_json::json!(true);
        let error = serde_json::from_value::<BenchmarkMatrix>(value).unwrap_err();
        assert!(error.to_string().contains("unknown field `invented`"));
    }

    fn phase() -> PhaseMetrics {
        PhaseMetrics {
            seconds: 1.0,
            active_peak_bytes: 1024,
            cache_peak_bytes: 512,
            cache_bytes_at_boundary: 256,
            samples: 2,
        }
    }

    fn valid_run(matrix: &BenchmarkMatrix, variant_id: &str, artifact_path: PathBuf) -> RunRecord {
        let case = matrix.case("qwen-q4-512").unwrap();
        let variant = matrix.variant(variant_id).unwrap().clone();
        let diagnostics = variant
            .toggles
            .iter()
            .map(|toggle| DiagnosticRecord {
                domain: "toggle".to_owned(),
                site: toggle.as_str().to_owned(),
                outcome: "applied".to_owned(),
                count: 1,
                reason: None,
            })
            .collect::<Vec<_>>();
        let measurements = (0..matrix.measured_runs)
            .map(|repetition| MeasurementRecord {
                repetition,
                total_seconds: 3.0,
                denoise_steps_per_second: 7.0,
                step_events: case.steps,
                saw_decode: true,
                phases: PhaseSet {
                    encode: phase(),
                    denoise: phase(),
                    decode: phase(),
                },
                output: OutputFingerprint {
                    kind: "images".to_owned(),
                    items: 1,
                    width: case.width,
                    height: case.height,
                    payload_bytes: u64::from(case.width) * u64::from(case.height) * 3,
                    sha256: "b".repeat(64),
                },
                diagnostics: diagnostics.clone(),
            })
            .collect();
        RunRecord {
            schema_version: RUN_SCHEMA_VERSION.to_owned(),
            benchmark_id: matrix.benchmark_id.clone(),
            case_id: case.id.clone(),
            family: case.family,
            provider: case.provider.clone(),
            artifact: ArtifactReceipt {
                key: case.artifact_key.clone(),
                repository: case.repository.clone(),
                resolved_revision: "a".repeat(40),
                tier: case.tier,
                canonical_path: artifact_path,
            },
            variant,
            environment: EnvironmentRecord {
                inference_revision: "c".repeat(40),
                mlx_revision: "d".repeat(40),
                rustc_version: "rustc test".to_owned(),
                os_version: "macOS test".to_owned(),
                hardware_model: "Mac-test".to_owned(),
                metal_device: "Apple test".to_owned(),
            },
            started_at_unix_millis: 1,
            load_seconds: 1.0,
            load_active_peak_bytes: 1024,
            load_cache_bytes_after_load: 0,
            warmup_runs_completed: matrix.warmup_runs,
            measurements,
        }
    }

    #[test]
    fn run_record_rejects_partial_unstable_and_unacknowledged_runs() {
        let matrix = matrix();
        let root = tempfile::tempdir().unwrap();
        let artifact_path = root.path().join("a".repeat(40)).join("q4");
        fs::create_dir_all(&artifact_path).unwrap();

        let valid = valid_run(&matrix, "baseline", artifact_path.clone());
        valid.validate_against(&matrix).unwrap();

        let mut partial = valid.clone();
        partial.measurements[0].step_events -= 1;
        assert!(partial
            .validate_against(&matrix)
            .unwrap_err()
            .to_string()
            .contains("observe exactly 8 denoise steps"));

        let mut unstable = valid;
        unstable.measurements[1].output.sha256 = "e".repeat(64);
        assert!(unstable
            .validate_against(&matrix)
            .unwrap_err()
            .to_string()
            .contains("not byte-stable"));

        let mut missing_receipt = valid_run(&matrix, "retained_compilation", artifact_path);
        missing_receipt.measurements[0].diagnostics.clear();
        assert!(missing_receipt
            .validate_against(&matrix)
            .unwrap_err()
            .to_string()
            .contains("lacks an applied receipt"));
    }

    #[test]
    fn run_record_accepts_lazy_load_without_eager_active_memory() {
        let matrix = matrix();
        let root = tempfile::tempdir().unwrap();
        let artifact_path = root.path().join("a".repeat(40)).join("q4");
        fs::create_dir_all(&artifact_path).unwrap();

        let mut lazy = valid_run(&matrix, "baseline", artifact_path);
        lazy.load_active_peak_bytes = 0;
        lazy.load_cache_bytes_after_load = 0;
        lazy.validate_against(&matrix).unwrap();
    }
}

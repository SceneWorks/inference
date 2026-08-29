//! Source-owned evidence primitives for the SC-20671 dense baseline campaign.
//!
//! This module deliberately keeps the receipt producer beside the product decoder.  The JSON
//! harness can validate evidence, but it must not be the component inventing model identity,
//! geometry, or lifecycle observations.  Device collection is supplied by the campaign runner;
//! these helpers make its inputs deterministic and testable without weights or Metal.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use core_llm::{Message, Role, Sampling, StreamEvent, TextLlm, TextLlmOutput, TextLlmRequest};

pub const REQUIRED_PHASES: [&str; 8] = [
    "process-start",
    "weights-loaded",
    "prefill-peak",
    "first-token",
    "decode-steady",
    "prompt-cache-reuse",
    "cancellation-cleanup",
    "post-run-release",
];
pub const QUALITY_CONTRACT_HASH: &str =
    "03c44b0f12caf79c1560e29fcfe536e2d7fd57153add4f3958697057b10116d";

pub const CONTEXT_BANDS: [&str; 4] = ["short", "medium", "memory-material", "fit-boundary"];

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptProvenance {
    pub scene_works_revision: String,
    pub inference_revision: String,
    pub mlx_revision: String,
    pub dependency_lock_sha256: String,
    pub os: String,
    pub xcode: String,
    pub hardware: String,
    pub model_id: String,
    pub model_file_sha256: String,
    pub model_file_bytes: u64,
    pub power_mode: String,
    pub thermal_state: String,
    pub command_template: String,
    pub command: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptMatrix {
    pub family: String,
    pub context_band: String,
    pub request_mode: String,
    pub prefill_mode: String,
    pub process_temperature: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptGeometry {
    pub batch: u64,
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dimension: u64,
    pub query_length: u64,
    pub kv_length: u64,
    pub layers: u64,
    pub element_bytes: u64,
    pub capacity: u64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptMlx {
    pub source: String,
    pub active_bytes: u64,
    pub cache_bytes: u64,
    pub peak_bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptPhase {
    pub phase: String,
    pub pid: u32,
    pub source: String,
    pub timestamp: String,
    pub phys_footprint_bytes: u64,
    pub phys_footprint_peak_bytes: u64,
    pub mlx: ReceiptMlx,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptAllocation {
    pub kind: String,
    pub role: String,
    pub lifetime: String,
    pub phase: String,
    pub timestamp: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptReconciliation {
    pub expected_dense_kv_bytes: u64,
    pub observed_persistent_kv_bytes: u64,
    pub tolerance_bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptRelease {
    pub verified: bool,
    pub phys_footprint_tolerance_bytes: u64,
    pub mlx_active_tolerance_bytes: u64,
    pub mlx_cache_tolerance_bytes: u64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptMemory {
    pub model_weights_bytes: u64,
    pub persistent_kv_bytes: u64,
    pub transient_workspace_bytes: u64,
    pub dense_theoretical_kv_bytes: u64,
    pub phase_samples: Vec<ReceiptPhase>,
    pub allocation_events: Vec<ReceiptAllocation>,
    pub reconciliation: ReceiptReconciliation,
    pub release: ReceiptRelease,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptTimingSample {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    pub cold_compile_ms: f64,
    pub warm_compile_ms: f64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptTimingSummary {
    pub decode_tokens_per_second_mean: f64,
    pub decode_tokens_per_second_p95: f64,
    pub decode_tokens_per_second_variance: f64,
    pub decode_tokens_per_second_coefficient_of_variation: f64,
    pub confidence_interval_low: f64,
    pub confidence_interval_high: f64,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptTimings {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    pub cold_compile_ms: f64,
    pub warm_compile_ms: f64,
    pub samples: Vec<ReceiptTimingSample>,
    pub summary: ReceiptTimingSummary,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptFixture {
    pub passed: bool,
    pub artifact_sha256: String,
    pub independent_reference: String,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptQualityStatistics {
    pub repeats: u64,
    pub warmups: u64,
    pub confidence_interval: String,
    pub outlier_policy: String,
    pub variance_policy: String,
    pub max_coefficient_of_variation: f64,
}
#[derive(Clone, Debug, Serialize)]
pub struct ReceiptQuality {
    #[serde(rename = "parityMaxError")]
    pub parity_max_error: f64,
    #[serde(rename = "perplexityDelta")]
    pub perplexity_delta: f64,
    #[serde(rename = "greedyTokenAgreement")]
    pub greedy_token_agreement: f64,
    #[serde(rename = "structuredToolAgreement")]
    pub structured_tool_agreement: f64,
    #[serde(rename = "needleRetrieval")]
    pub needle_retrieval: f64,
    #[serde(rename = "multiTurnPromptCache")]
    pub multi_turn_prompt_cache: f64,
    pub statistics: ReceiptQualityStatistics,
    #[serde(rename = "fixtureEvidence")]
    pub fixture_evidence: std::collections::BTreeMap<String, ReceiptFixture>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptLifecycle {
    pub append: bool,
    pub chunked_prefill: bool,
    pub single_shot_prefill: bool,
    pub prompt_cache_reuse: bool,
    pub trim: bool,
    pub rollback: bool,
    pub clear: bool,
    pub cancel: bool,
    pub clone: bool,
    pub batch_split: bool,
    pub batch_merge: bool,
    pub prefix_copy_on_write: bool,
    pub page_import: bool,
    pub page_export: bool,
    pub serialization: bool,
    pub restore: bool,
    pub dense_fallback: bool,
    pub post_run_release: bool,
    /// Schema-approved `<capability>FallbackReason` fields for unsupported capabilities.
    #[serde(flatten)]
    pub fallback_reasons: std::collections::BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReceiptCancellation {
    pub cleanup_verified: bool,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub schema_version: u32,
    pub harness_version: String,
    pub run_id: String,
    pub captured_at: String,
    pub mode: String,
    pub status: String,
    pub contract_hash: String,
    pub receipt_sha256: String,
    pub provenance: ReceiptProvenance,
    pub matrix: ReceiptMatrix,
    pub geometry: ReceiptGeometry,
    pub memory: ReceiptMemory,
    pub timings: ReceiptTimings,
    pub quality: ReceiptQuality,
    pub lifecycle: ReceiptLifecycle,
    pub cancellation: ReceiptCancellation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawTiming {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    pub cold_compile_ms: f64,
    pub warm_compile_ms: f64,
}

pub struct ReceiptBuilder {
    pub template: Receipt,
    pub phases: Vec<ReceiptPhase>,
    pub allocations: Vec<ReceiptAllocation>,
    pub timings: Vec<RawTiming>,
    pub quality: QualityObservation,
}

impl ReceiptBuilder {
    pub fn finish(mut self) -> Result<Receipt, String> {
        self.template.schema_version = 3;
        self.template.harness_version = "sc-20671-kv-baseline-v3".into();
        let digest = |v: &str| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit());
        let revision = |v: &str| v.len() == 40 && v.bytes().all(|b| b.is_ascii_hexdigit());
        if !digest(&self.template.contract_hash)
            || !digest(&self.template.provenance.dependency_lock_sha256)
            || !digest(&self.template.provenance.model_file_sha256)
            || !revision(&self.template.provenance.scene_works_revision)
            || !revision(&self.template.provenance.inference_revision)
        {
            return Err("malformed receipt identity".into());
        }
        if self.template.provenance.thermal_state != "nominal"
            || !self.template.provenance.command_template.contains("{mode}")
            || self.template.provenance.command
                != self
                    .template
                    .provenance
                    .command_template
                    .replace("{mode}", &self.template.mode)
        {
            return Err("invalid command or thermal provenance".into());
        }
        if !["llama", "qwen"].contains(&self.template.matrix.family.as_str())
            || !CONTEXT_BANDS.contains(&self.template.matrix.context_band.as_str())
            || !["single", "supported-batch"].contains(&self.template.matrix.request_mode.as_str())
            || !["chunked", "single-shot"].contains(&self.template.matrix.prefill_mode.as_str())
            || !["cold", "warm"].contains(&self.template.matrix.process_temperature.as_str())
        {
            return Err("invalid matrix coordinate".into());
        }
        if self.template.geometry.query_heads == 0
            || self.template.geometry.kv_heads == 0
            || self.template.geometry.query_heads % self.template.geometry.kv_heads != 0
            || self.template.geometry.capacity < self.template.geometry.kv_length
        {
            return Err("invalid geometry".into());
        }
        if (self.template.matrix.request_mode == "single" && self.template.geometry.batch != 1)
            || (self.template.matrix.request_mode == "supported-batch"
                && self.template.geometry.batch <= 1)
        {
            return Err("batch disagrees with matrix".into());
        }
        if self.phases.len() != 8 || self.allocations.is_empty() || self.timings.len() != 5 {
            return Err("receipt evidence is incomplete".into());
        }
        for (phase, expected) in self.phases.iter().zip(REQUIRED_PHASES) {
            if phase.phase != expected
                || phase.pid == 0
                || phase.source != "footprint -p"
                || phase.mlx.source != "mlx_rs::memory"
            {
                return Err("invalid phase evidence".into());
            }
        }
        if self
            .phases
            .windows(2)
            .any(|w| w[0].timestamp >= w[1].timestamp)
        {
            return Err("phase timestamps are not strictly increasing".into());
        }
        let geometry = &self.template.geometry;
        let dense = dense_kv_bytes(
            geometry.batch,
            geometry.layers,
            geometry.kv_heads,
            geometry.capacity,
            geometry.head_dimension,
            geometry.element_bytes,
        )?;
        let metrics = compute_quality(&self.quality)?;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if self.timings.iter().any(|t| {
            ![
                t.load_ms,
                t.prefill_ms,
                t.ttft_ms,
                t.first_token_ms,
                t.decode_tokens_per_second,
                t.cold_compile_ms,
                t.warm_compile_ms,
            ]
            .into_iter()
            .all(positive)
        }) {
            return Err("timing samples must be finite and positive".into());
        }
        let mean = |f: fn(&RawTiming) -> f64| {
            self.timings.iter().map(f).sum::<f64>() / self.timings.len() as f64
        };
        let decode_mean = mean(|t| t.decode_tokens_per_second);
        let variance = self
            .timings
            .iter()
            .map(|t| (t.decode_tokens_per_second - decode_mean).powi(2))
            .sum::<f64>()
            / self.timings.len() as f64;
        let mut sorted: Vec<f64> = self
            .timings
            .iter()
            .map(|t| t.decode_tokens_per_second)
            .collect();
        sorted.sort_by(f64::total_cmp);
        let summary = ReceiptTimingSummary {
            decode_tokens_per_second_mean: decode_mean,
            decode_tokens_per_second_p95: sorted[4],
            decode_tokens_per_second_variance: variance,
            decode_tokens_per_second_coefficient_of_variation: variance.sqrt() / decode_mean,
            confidence_interval_low: sorted[0],
            confidence_interval_high: sorted[4],
        };
        self.template.memory.phase_samples = self.phases;
        self.template.memory.allocation_events = self.allocations;
        self.template.memory.dense_theoretical_kv_bytes = dense;
        self.template.memory.reconciliation.expected_dense_kv_bytes = dense;
        self.template
            .memory
            .reconciliation
            .observed_persistent_kv_bytes = self.template.memory.persistent_kv_bytes;
        self.template.timings = ReceiptTimings {
            load_ms: mean(|t| t.load_ms),
            prefill_ms: mean(|t| t.prefill_ms),
            ttft_ms: mean(|t| t.ttft_ms),
            first_token_ms: mean(|t| t.first_token_ms),
            decode_tokens_per_second: decode_mean,
            cold_compile_ms: mean(|t| t.cold_compile_ms),
            warm_compile_ms: mean(|t| t.warm_compile_ms),
            samples: self
                .timings
                .into_iter()
                .map(|t| ReceiptTimingSample {
                    load_ms: t.load_ms,
                    prefill_ms: t.prefill_ms,
                    ttft_ms: t.ttft_ms,
                    first_token_ms: t.first_token_ms,
                    decode_tokens_per_second: t.decode_tokens_per_second,
                    cold_compile_ms: t.cold_compile_ms,
                    warm_compile_ms: t.warm_compile_ms,
                })
                .collect(),
            summary,
        };
        self.template.quality.parity_max_error = metrics.parity_max_error;
        self.template.quality.perplexity_delta = metrics.perplexity_delta;
        self.template.quality.greedy_token_agreement = metrics.greedy_token_agreement;
        self.template.quality.structured_tool_agreement = metrics.structured_tool_agreement;
        self.template.quality.needle_retrieval = metrics.needle_retrieval;
        self.template.quality.multi_turn_prompt_cache = metrics.multi_turn_prompt_cache;
        if self.template.memory.persistent_kv_bytes == 0
            || self.template.memory.model_weights_bytes == 0
        {
            return Err("memory attribution totals must be nonzero".into());
        }
        validate_receipt_semantics(&self.template)?;
        Ok(self.template)
    }
}

impl Receipt {
    /// Serialize the exact receipt bytes; `receipt_sha256` is filled by the caller after clearing
    /// that field according to the SceneWorks semantic-core convention.
    pub fn bytes(&self) -> Result<Vec<u8>, serde_json::Error> {
        let mut bytes = serde_json::to_vec(self)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

pub fn validate_receipt_semantics(receipt: &Receipt) -> Result<(), String> {
    if receipt.schema_version != 3
        || receipt.harness_version != "sc-20671-kv-baseline-v3"
        || receipt.status != "complete"
        || receipt.contract_hash != QUALITY_CONTRACT_HASH
    {
        return Err("receipt constants mismatch".into());
    }
    if receipt.run_id.is_empty()
        || !receipt.captured_at.contains('T')
        || !receipt.captured_at.ends_with('Z')
    {
        return Err("receipt timestamp/run id is malformed".into());
    }
    let pids: Vec<u32> = receipt.memory.phase_samples.iter().map(|p| p.pid).collect();
    if pids.len() != 8 || pids.iter().any(|p| *p == 0 || *p != pids[0]) {
        return Err("phase PID evidence is inconsistent".into());
    }
    if receipt.memory.phase_samples.windows(2).any(|w| {
        w[0].timestamp >= w[1].timestamp
            || w[1].phys_footprint_peak_bytes < w[0].phys_footprint_peak_bytes
            || w[1].mlx.peak_bytes < w[0].mlx.peak_bytes
    }) {
        return Err("phase sequence or peak monotonicity failed".into());
    }
    if receipt.memory.phase_samples.iter().any(|p| {
        p.phys_footprint_bytes < p.mlx.active_bytes || p.mlx.peak_bytes < p.mlx.active_bytes
    }) {
        return Err("memory containment failed".into());
    }
    let dense = dense_kv_bytes(
        receipt.geometry.batch,
        receipt.geometry.layers,
        receipt.geometry.kv_heads,
        receipt.geometry.capacity,
        receipt.geometry.head_dimension,
        receipt.geometry.element_bytes,
    )?;
    if receipt.memory.dense_theoretical_kv_bytes != dense
        || receipt.memory.reconciliation.expected_dense_kv_bytes != dense
        || receipt.memory.reconciliation.observed_persistent_kv_bytes
            != receipt.memory.persistent_kv_bytes
    {
        return Err("dense KV reconciliation failed".into());
    }
    if receipt.timings.samples.len() != 5
        || receipt
            .timings
            .summary
            .decode_tokens_per_second_coefficient_of_variation
            > 0.05
    {
        return Err("timing policy failed".into());
    }
    if receipt.quality.fixture_evidence.len() != 4
        || receipt.quality.statistics.repeats != 5
        || receipt.quality.statistics.warmups != 2
    {
        return Err("quality contract evidence incomplete".into());
    }
    if !receipt.memory.release.verified || !receipt.cancellation.cleanup_verified {
        return Err("release/cancellation evidence failed".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundle {
    pub receipt: Vec<u8>,
    pub receipt_sidecar: String,
    pub human: Vec<u8>,
    pub human_sidecar: String,
}

/// Assemble all receipt artifacts in memory before any caller writes them. This prevents a
/// partially-written receipt directory from being mistaken for a campaign result.
pub fn assemble_artifacts(mut receipt: Receipt) -> Result<ArtifactBundle, String> {
    let mut semantic = serde_json::to_value(&receipt).map_err(|e| e.to_string())?;
    semantic
        .as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    let semantic = serde_json::to_vec(&semantic).map_err(|e| e.to_string())?;
    receipt.receipt_sha256 = seal_bytes(&semantic);
    let bytes = receipt.bytes().map_err(|e| e.to_string())?;
    let human = serde_json::to_string_pretty(&receipt)
        .map_err(|e| e.to_string())?
        .into_bytes();
    Ok(ArtifactBundle {
        receipt_sidecar: format!("{}  receipt.json", seal_bytes(&bytes)),
        human_sidecar: format!("{}  receipt.txt", seal_bytes(&human)),
        receipt: bytes,
        human,
    })
}

pub fn validate_artifact_bundle(bundle: &ArtifactBundle) -> Result<(), String> {
    if bundle.receipt_sidecar != format!("{}  receipt.json", seal_bytes(&bundle.receipt))
        || bundle.human_sidecar != format!("{}  receipt.txt", seal_bytes(&bundle.human))
    {
        return Err("artifact sidecar does not match exact bytes".into());
    }
    if bundle.receipt.is_empty() || bundle.human.is_empty() {
        return Err("partial artifact bundle".into());
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bundle.receipt).map_err(|e| e.to_string())?;
    let object = value.as_object().ok_or("receipt is not an object")?;
    for key in ["memory", "timings", "quality", "lifecycle", "cancellation"] {
        if !object.contains_key(key) {
            return Err(format!("receipt missing {key}"));
        }
    }
    let memory = object["memory"]
        .as_object()
        .ok_or("memory is not an object")?;
    if memory["phaseSamples"]
        .as_array()
        .map_or(true, |v| v.len() != 8)
        || memory["allocationEvents"]
            .as_array()
            .map_or(true, |v| v.is_empty())
    {
        return Err("receipt memory evidence is incomplete".into());
    }
    let timings = object["timings"]
        .as_object()
        .ok_or("timings is not an object")?;
    if timings["samples"].as_array().map_or(true, |v| v.len() != 5) {
        return Err("receipt timing samples are incomplete".into());
    }
    let quality = object["quality"]
        .as_object()
        .ok_or("quality is not an object")?;
    if quality["fixtureEvidence"]
        .as_object()
        .map_or(true, |v| v.len() != 4)
    {
        return Err("receipt fixture evidence is incomplete".into());
    }
    let mut core = value.clone();
    core.as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    let expected = seal_bytes(&serde_json::to_vec(&core).map_err(|e| e.to_string())?);
    if object.get("receiptSha256").and_then(|v| v.as_str()) != Some(expected.as_str()) {
        return Err("receipt semantic hash mismatch".into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Coordinate {
    pub family: &'static str,
    pub context_band: &'static str,
    pub request_mode: &'static str,
    pub prefill_mode: &'static str,
    pub process_temperature: &'static str,
}

/// The exact required dense campaign frontier (2 × 4 × 2 × 2 × 2).
pub fn required_coordinates() -> Vec<Coordinate> {
    ["llama", "qwen"]
        .into_iter()
        .flat_map(|family| {
            CONTEXT_BANDS
                .into_iter()
                .map(move |context_band| (family, context_band))
        })
        .flat_map(|(family, context_band)| {
            ["single", "supported-batch"]
                .into_iter()
                .map(move |request_mode| (family, context_band, request_mode))
        })
        .flat_map(|(family, context_band, request_mode)| {
            ["chunked", "single-shot"]
                .into_iter()
                .map(move |prefill_mode| (family, context_band, request_mode, prefill_mode))
        })
        .flat_map(|(family, context_band, request_mode, prefill_mode)| {
            ["cold", "warm"]
                .into_iter()
                .map(move |process_temperature| Coordinate {
                    family,
                    context_band,
                    request_mode,
                    prefill_mode,
                    process_temperature,
                })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InventoryFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotInventory {
    pub root: PathBuf,
    pub files: Vec<InventoryFile>,
    pub bytes: u64,
    pub sha256: String,
}

/// Hash every resolved file in a HF snapshot.  Symlink names and blob names are never trusted.
pub fn inventory_snapshot(root: impl AsRef<Path>) -> std::io::Result<SnapshotInventory> {
    fn visit(root: &Path, dir: &Path, out: &mut Vec<InventoryFile>) -> std::io::Result<()> {
        let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)?;
            if metadata.is_dir() {
                visit(root, &path, out)?;
                continue;
            }
            if !metadata.is_file() && !metadata.file_type().is_symlink() {
                continue;
            }
            let resolved = fs::canonicalize(&path)?;
            let bytes = fs::metadata(&resolved)?.len();
            if bytes == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("empty snapshot file {}", path.display()),
                ));
            }
            let mut input = File::open(&resolved)?;
            let mut digest = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let read = input.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                digest.update(&buffer[..read]);
            }
            let sha = hex(&digest.finalize());
            let relative = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            out.push(InventoryFile {
                path: relative,
                bytes,
                sha256: sha,
            });
        }
        Ok(())
    }
    let root = fs::canonicalize(root)?;
    let mut files = Vec::new();
    visit(&root, &root, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    if files.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty snapshot",
        ));
    }
    let mut digest = Sha256::new();
    for file in &files {
        digest.update(file.path.as_bytes());
        digest.update([0]);
        digest.update(file.bytes.to_string().as_bytes());
        digest.update([0]);
        digest.update(file.sha256.as_bytes());
        digest.update([b'\n']);
    }
    Ok(SnapshotInventory {
        root,
        bytes: files.iter().map(|f| f.bytes).sum(),
        sha256: hex(&digest.finalize()),
        files,
    })
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Exact dense KV theoretical size, including batch and K+V.
pub fn dense_kv_bytes(
    batch: u64,
    layers: u64,
    kv_heads: u64,
    capacity: u64,
    head_dimension: u64,
    element_bytes: u64,
) -> Result<u64, String> {
    [
        batch,
        layers,
        kv_heads,
        capacity,
        head_dimension,
        element_bytes,
        2,
    ]
    .into_iter()
    .try_fold(1_u64, |value, factor| value.checked_mul(factor))
    .ok_or_else(|| "dense KV geometry overflows u64".into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseSample {
    pub phase: String,
    pub pid: u32,
    pub source: String,
    pub captured_at: String,
    pub footprint_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}

/// Validate the non-forgeable structural part of phase evidence before serialization.
pub fn validate_phases(samples: &[PhaseSample]) -> Result<(), String> {
    if samples.len() != REQUIRED_PHASES.len() {
        return Err("phase set is incomplete or duplicated".into());
    }
    let pid = samples
        .first()
        .map(|s| s.pid)
        .filter(|p| *p > 0)
        .ok_or("worker PID is missing")?;
    for (sample, expected) in samples.iter().zip(REQUIRED_PHASES) {
        if sample.phase != expected {
            return Err(format!("expected phase {expected}, got {}", sample.phase));
        }
        if sample.pid != pid || sample.source.is_empty() || sample.captured_at.is_empty() {
            return Err(format!("invalid evidence for {expected}"));
        }
    }
    Ok(())
}

/// The producer's cancellation invariant: cleanup evidence is required even if cancellation was
/// already set before inference began.
pub fn cancellation_cleanup_required(
    aborted_before_start: bool,
    cleanup_observed: bool,
    released: bool,
) -> Result<(), String> {
    if (aborted_before_start || cleanup_observed) && !released {
        return Err("cancellation did not release resources".into());
    }
    Ok(())
}

/// Stable seal over the exact bytes written to a receipt or fixture artifact.
pub fn seal_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Serialize producer-owned artifacts once; the returned bytes are exactly what must be written
/// and hashed in the adjacent `.sha256` sidecar.  Callers must not reseal parsed JSON.
pub fn sealed_json(value: &serde_json::Value) -> (Vec<u8>, String) {
    let mut bytes = serde_json::to_vec(value).expect("receipt JSON is serializable");
    bytes.push(b'\n');
    let digest = seal_bytes(&bytes);
    (bytes, digest)
}

pub const REQUIRED_FIXTURES: [&str; 4] = [
    "kernel-fp32-reference",
    "structured-tool-call",
    "long-context-needle",
    "multi-turn-prompt-cache",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixtureEvidence {
    pub name: String,
    pub artifact_sha256: String,
    pub independent_reference: String,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualityObservation {
    pub parity_errors: Vec<f64>,
    pub reference_perplexity: f64,
    pub candidate_perplexity: f64,
    pub greedy_matches: u64,
    pub greedy_total: u64,
    pub tool_matches: u64,
    pub tool_total: u64,
    pub needle_matches: u64,
    pub needle_total: u64,
    pub cache_matches: u64,
    pub cache_total: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityMetrics {
    pub parity_max_error: f64,
    pub perplexity_delta: f64,
    pub greedy_token_agreement: f64,
    pub structured_tool_agreement: f64,
    pub needle_retrieval: f64,
    pub multi_turn_prompt_cache: f64,
}

/// Compute quality from raw reference/candidate observations; no caller-supplied pass flag exists.
pub fn compute_quality(raw: &QualityObservation) -> Result<QualityMetrics, String> {
    let ratio = |matched: u64, total: u64| {
        if total == 0 {
            Err("quality observation has zero denominator".into())
        } else {
            Ok(matched as f64 / total as f64)
        }
    };
    if !raw.reference_perplexity.is_finite()
        || !raw.candidate_perplexity.is_finite()
        || raw.parity_errors.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("quality observation contains non-finite values".into());
    }
    Ok(QualityMetrics {
        parity_max_error: raw.parity_errors.iter().copied().fold(0.0, f64::max),
        perplexity_delta: raw.candidate_perplexity - raw.reference_perplexity,
        greedy_token_agreement: ratio(raw.greedy_matches, raw.greedy_total)?,
        structured_tool_agreement: ratio(raw.tool_matches, raw.tool_total)?,
        needle_retrieval: ratio(raw.needle_matches, raw.needle_total)?,
        multi_turn_prompt_cache: ratio(raw.cache_matches, raw.cache_total)?,
    })
}

/// Build and seal a producer-owned fixture artifact from raw observations.
pub fn fixture_artifact(name: &str, raw: &QualityObservation) -> Result<(Vec<u8>, String), String> {
    if !REQUIRED_FIXTURES.contains(&name) {
        return Err(format!("unknown fixture {name}"));
    }
    let metrics = compute_quality(raw)?;
    let value = serde_json::json!({ "fixture": name, "metrics": { "parityMaxError": metrics.parity_max_error, "perplexityDelta": metrics.perplexity_delta, "greedyTokenAgreement": metrics.greedy_token_agreement, "structuredToolAgreement": metrics.structured_tool_agreement, "needleRetrieval": metrics.needle_retrieval, "multiTurnPromptCache": metrics.multi_turn_prompt_cache } });
    Ok(sealed_json(&value))
}

pub fn validate_fixture_evidence(evidence: &[FixtureEvidence]) -> Result<(), String> {
    if evidence.len() != REQUIRED_FIXTURES.len() || !evidence.iter().all(|e| e.passed) {
        return Err("all four independent quality fixtures must pass".into());
    }
    for (item, expected) in evidence.iter().zip(REQUIRED_FIXTURES) {
        if item.name != expected
            || item.artifact_sha256.len() != 64
            || !item.artifact_sha256.bytes().all(|b| b.is_ascii_hexdigit())
            || item.independent_reference.is_empty()
        {
            return Err(format!("invalid sealed evidence for {expected}"));
        }
    }
    Ok(())
}

/// A monotonic timestamp suitable for the producer's internal sequencing tests.
pub fn sequence_marker() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    SEQUENCE.fetch_add(1, Ordering::Relaxed).saturating_add(1)
}

/// The narrow observation seam used by a real campaign runner.  The runner owns platform probes
/// (`footprint` and `mlx_rs::memory`); the product path owns the phase boundaries and generation.
pub trait Observer {
    fn phase(&mut self, name: &'static str);
    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySample {
    pub captured_at: String,
    pub pid: u32,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}

fn timestamp_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("{seconds}")
}

pub trait CampaignSampler {
    fn sample(&mut self, phase: &'static str) -> Result<MemorySample, String>;
}

/// Injectable phase recorder used by both the device runner and weightless tests.
pub struct PhaseRecorder<S> {
    sampler: S,
    pub samples: Vec<ReceiptPhase>,
    pid: Option<u32>,
}

impl<S: CampaignSampler> PhaseRecorder<S> {
    pub fn new(sampler: S) -> Self {
        Self {
            sampler,
            samples: Vec::new(),
            pid: None,
        }
    }
    pub fn capture(&mut self, phase: &'static str) -> Result<(), String> {
        let sample = self.sampler.sample(phase)?;
        if let Some(pid) = self.pid {
            if pid != sample.pid {
                return Err("campaign worker PID changed".into());
            }
        } else {
            if sample.pid == 0 {
                return Err("campaign worker PID is zero".into());
            }
            self.pid = Some(sample.pid);
        }
        self.samples.push(ReceiptPhase {
            phase: phase.into(),
            pid: sample.pid,
            source: "footprint -p".into(),
            timestamp: sample.captured_at,
            phys_footprint_bytes: sample.current_bytes,
            phys_footprint_peak_bytes: sample.peak_bytes,
            mlx: ReceiptMlx {
                source: "mlx_rs::memory".into(),
                active_bytes: sample.mlx_active_bytes,
                cache_bytes: sample.mlx_cache_bytes,
                peak_bytes: sample.mlx_peak_bytes,
            },
        });
        Ok(())
    }
    pub fn finish(self) -> Result<Vec<ReceiptPhase>, String> {
        if self.samples.len() != REQUIRED_PHASES.len() {
            return Err("campaign did not capture all required phases".into());
        }
        for (sample, expected) in self.samples.iter().zip(REQUIRED_PHASES) {
            if sample.phase != expected {
                return Err(format!("phase order mismatch: expected {expected}"));
            }
            if sample.phys_footprint_bytes < sample.mlx.active_bytes
                || sample.phys_footprint_peak_bytes < sample.phys_footprint_bytes
            {
                return Err(format!("invalid memory attribution at {expected}"));
            }
        }
        if self
            .samples
            .windows(2)
            .any(|w| w[0].timestamp >= w[1].timestamp)
        {
            return Err("phase timestamps are not strictly increasing".into());
        }
        Ok(self.samples)
    }
}

/// Run the evidence state machine around a product operation. The operation receives the recorder
/// so it can report cache/prefix/cancellation events at their ownership sites.
pub fn run_lifecycle<S: CampaignSampler, F>(
    sampler: S,
    mut operation: F,
) -> Result<Vec<ReceiptPhase>, String>
where
    F: FnMut(&mut PhaseRecorder<S>) -> Result<(), String>,
{
    let mut recorder = PhaseRecorder::new(sampler);
    operation(&mut recorder)?;
    recorder.finish()
}

/// Parse Darwin footprint fields while accepting the units emitted by different macOS releases.
pub fn parse_footprint_value(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let number: f64 = parts.next()?.parse().ok()?;
    let unit = parts.next().unwrap_or("B").to_ascii_uppercase();
    let multiplier = match unit.as_str() {
        "B" => 1.0,
        "KB" => 1024.0,
        "MB" => 1024.0 * 1024.0,
        "GB" => 1024.0 * 1024.0 * 1024.0,
        _ => return None,
    };
    if !number.is_finite() || number < 0.0 {
        return None;
    }
    (number * multiplier).round().try_into().ok()
}

#[cfg(target_os = "macos")]
pub fn sample_memory(pid: u32) -> std::io::Result<MemorySample> {
    use std::process::Command;
    let output = Command::new("/usr/bin/footprint")
        .args(["--pid", &pid.to_string(), "--noCategories", "--wired"])
        .output()?;
    if !output.status.success() {
        return Err(std::io::Error::other("footprint failed"));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let field = |name: &str| {
        text.lines()
            .find_map(|line| line.trim().strip_prefix(name))
            .and_then(parse_footprint_value)
    };
    Ok(MemorySample {
        captured_at: timestamp_now(),
        pid,
        current_bytes: field("phys_footprint:")
            .ok_or_else(|| std::io::Error::other("missing phys_footprint"))?,
        peak_bytes: field("phys_footprint_peak:")
            .ok_or_else(|| std::io::Error::other("missing phys_footprint_peak"))?,
        mlx_active_bytes: mlx_rs::memory::get_active_memory() as u64,
        mlx_cache_bytes: mlx_rs::memory::get_cache_memory() as u64,
        mlx_peak_bytes: mlx_rs::memory::get_peak_memory() as u64,
    })
}

#[cfg(not(target_os = "macos"))]
pub fn sample_memory(_pid: u32) -> std::io::Result<MemorySample> {
    Err(std::io::Error::other(
        "SC-20671 memory sampling requires macOS",
    ))
}

/// Run one dense product-path coordinate.  This intentionally loads through `LlamaProvider` and
/// `core-llm::TextLlm`, rather than a test double or HTTP route.  The lower-level cache observation
/// callbacks remain a separate seam because MLX arrays are backend-owned and cannot cross the
/// backend-neutral contract.  A campaign runner must call this in a fresh process per coordinate.
pub fn run_dense_coordinate(
    snapshot: impl AsRef<Path>,
    prompt: &str,
    max_new_tokens: u32,
    observer: &mut dyn Observer,
) -> core_llm::Result<TextLlmOutput> {
    observer.phase("process-start");
    let _inventory = inventory_snapshot(snapshot.as_ref())
        .map_err(|e| core_llm::Error::Load(format!("snapshot inventory: {e}")))?;
    let provider = crate::provider::LlamaProvider::load(&core_llm::LoadSpec::dense(
        snapshot.as_ref().to_string_lossy().to_string(),
    ))?;
    observer.phase("weights-loaded");
    let request = TextLlmRequest {
        messages: vec![Message::text(Role::User, prompt)],
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        },
        max_new_tokens,
        seed: Some(0),
        ..Default::default()
    };
    let mut saw_token = false;
    let output = provider.generate_observed(
        &request,
        &mut |event| {
            if matches!(event, StreamEvent::Token { .. }) {
                saw_token = true;
            }
        },
        observer,
    )?;
    if !saw_token {
        return Err(core_llm::Error::InvalidRequest(
            "dense campaign produced no first-token observation".into(),
        ));
    }
    observer.phase("decode-steady");
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn formula_includes_batch_and_key_value_pair() {
        assert_eq!(dense_kv_bytes(2, 3, 4, 5, 6, 2), Ok(1440));
        assert!(dense_kv_bytes(u64::MAX, 2, 1, 1, 1, 1).is_err());
    }

    #[test]
    fn inventory_hashes_resolved_shards_deterministically() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("b.safetensors"), b"b").unwrap();
        let mut file = fs::File::create(dir.path().join("a.safetensors")).unwrap();
        file.write_all(b"a").unwrap();
        let first = inventory_snapshot(dir.path()).unwrap();
        let second = inventory_snapshot(dir.path()).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.files[0].path, "a.safetensors");
    }

    #[test]
    fn phases_are_exact_and_single_pid() {
        let samples = REQUIRED_PHASES
            .iter()
            .map(|phase| PhaseSample {
                phase: (*phase).into(),
                pid: 42,
                source: "test".into(),
                captured_at: "2026-01-01T00:00:00Z".into(),
                footprint_bytes: 1,
                mlx_active_bytes: 1,
                mlx_cache_bytes: 1,
                mlx_peak_bytes: 1,
            })
            .collect::<Vec<_>>();
        assert!(validate_phases(&samples).is_ok());
    }

    #[test]
    fn cancellation_before_start_still_requires_release() {
        assert!(cancellation_cleanup_required(true, false, false).is_err());
        assert!(cancellation_cleanup_required(true, false, true).is_ok());
    }

    #[test]
    fn seal_includes_exact_newline_bytes() {
        assert_ne!(seal_bytes(b"{}"), seal_bytes(b"{}\n"));
    }

    #[test]
    fn sealed_json_hashes_written_bytes() {
        let (bytes, digest) = sealed_json(&serde_json::json!({"b": 2, "a": 1}));
        assert_eq!(digest, seal_bytes(&bytes));
        assert_ne!(digest, seal_bytes(&bytes[..bytes.len() - 1]));
    }

    #[test]
    fn footprint_units_are_deterministic() {
        assert_eq!(parse_footprint_value("1664 KB"), Some(1_703_936));
        assert_eq!(parse_footprint_value("2 MB"), Some(2 * 1024 * 1024));
        assert!(parse_footprint_value("nan B").is_none());
        assert!(parse_footprint_value("4 TB").is_none());
    }

    #[test]
    fn required_matrix_is_exactly_64_coordinates() {
        let coordinates = required_coordinates();
        assert_eq!(coordinates.len(), 64);
        for (index, coordinate) in coordinates.iter().enumerate() {
            assert!(!coordinates[index + 1..].contains(coordinate));
        }
    }

    #[test]
    fn quality_metrics_are_derived_from_raw_observations() {
        let raw = QualityObservation {
            parity_errors: vec![0.0, 0.0002],
            reference_perplexity: 10.0,
            candidate_perplexity: 9.5,
            greedy_matches: 9,
            greedy_total: 10,
            tool_matches: 1,
            tool_total: 1,
            needle_matches: 1,
            needle_total: 1,
            cache_matches: 1,
            cache_total: 1,
        };
        let metrics = compute_quality(&raw).unwrap();
        assert_eq!(metrics.parity_max_error, 0.0002);
        assert_eq!(metrics.perplexity_delta, -0.5);
        assert_eq!(metrics.greedy_token_agreement, 0.9);
        assert!(compute_quality(&QualityObservation {
            greedy_total: 0,
            ..raw
        })
        .is_err());
    }

    struct FakeSampler;
    impl CampaignSampler for FakeSampler {
        fn sample(&mut self, _phase: &'static str) -> Result<MemorySample, String> {
            Ok(MemorySample {
                captured_at: "2026-01-01T00:00:00Z".into(),
                pid: 7,
                current_bytes: 100,
                peak_bytes: 100,
                mlx_active_bytes: 10,
                mlx_cache_bytes: 1,
                mlx_peak_bytes: 10,
            })
        }
    }

    #[test]
    fn fake_runner_requires_and_orders_all_phases() {
        let phases = run_lifecycle(FakeSampler, |recorder| {
            recorder.capture("process-start")?;
            recorder.capture("weights-loaded")?;
            recorder.capture("prefill-peak")?;
            recorder.capture("first-token")?;
            recorder.capture("decode-steady")?;
            recorder.capture("prompt-cache-reuse")?;
            recorder.capture("cancellation-cleanup")?;
            recorder.capture("post-run-release")?;
            Ok(())
        })
        .unwrap();
        assert_eq!(phases.len(), 8);
        assert_eq!(phases[5].phase, "prompt-cache-reuse");
        assert!(run_lifecycle(FakeSampler, |_recorder| Ok(())).is_err());
    }

    #[test]
    fn fixture_evidence_is_fail_closed() {
        let evidence = REQUIRED_FIXTURES
            .iter()
            .map(|name| FixtureEvidence {
                name: (*name).into(),
                artifact_sha256: "a".repeat(64),
                independent_reference: "fp32-reference".into(),
                passed: true,
            })
            .collect::<Vec<_>>();
        assert!(validate_fixture_evidence(&evidence).is_ok());
        assert!(validate_fixture_evidence(&evidence[..3]).is_err());
    }

    #[test]
    fn artifact_bundle_rejects_tampering_and_partial_outputs() {
        let receipt = Receipt {
            schema_version: 3,
            harness_version: "sc-20671-kv-baseline-v3".into(),
            run_id: "run".into(),
            captured_at: "2026-01-01T00:00:00Z".into(),
            mode: "dense".into(),
            status: "complete".into(),
            contract_hash: "a".repeat(64),
            receipt_sha256: String::new(),
            provenance: ReceiptProvenance {
                scene_works_revision: "a".repeat(40),
                inference_revision: "b".repeat(40),
                mlx_revision: "mlx".into(),
                dependency_lock_sha256: "c".repeat(64),
                os: "macOS".into(),
                xcode: "xcode".into(),
                hardware: "hardware".into(),
                model_id: "model".into(),
                model_file_sha256: "d".repeat(64),
                model_file_bytes: 1,
                power_mode: "nominal".into(),
                thermal_state: "nominal".into(),
                command_template: "run --mode {mode}".into(),
                command: "run --mode dense".into(),
            },
            matrix: ReceiptMatrix {
                family: "llama".into(),
                context_band: "short".into(),
                request_mode: "single".into(),
                prefill_mode: "single-shot".into(),
                process_temperature: "cold".into(),
            },
            geometry: ReceiptGeometry {
                batch: 1,
                query_heads: 1,
                kv_heads: 1,
                head_dimension: 1,
                query_length: 1,
                kv_length: 1,
                layers: 1,
                element_bytes: 2,
                capacity: 1,
            },
            memory: ReceiptMemory {
                model_weights_bytes: 1,
                persistent_kv_bytes: 4,
                transient_workspace_bytes: 0,
                dense_theoretical_kv_bytes: 4,
                phase_samples: vec![],
                allocation_events: vec![],
                reconciliation: ReceiptReconciliation {
                    expected_dense_kv_bytes: 4,
                    observed_persistent_kv_bytes: 4,
                    tolerance_bytes: 0,
                },
                release: ReceiptRelease {
                    verified: true,
                    phys_footprint_tolerance_bytes: 0,
                    mlx_active_tolerance_bytes: 0,
                    mlx_cache_tolerance_bytes: 0,
                },
            },
            timings: ReceiptTimings {
                load_ms: 1.0,
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 1.0,
                decode_tokens_per_second: 1.0,
                cold_compile_ms: 1.0,
                warm_compile_ms: 1.0,
                samples: vec![],
                summary: ReceiptTimingSummary {
                    decode_tokens_per_second_mean: 1.0,
                    decode_tokens_per_second_p95: 1.0,
                    decode_tokens_per_second_variance: 0.0,
                    decode_tokens_per_second_coefficient_of_variation: 0.0,
                    confidence_interval_low: 1.0,
                    confidence_interval_high: 1.0,
                },
            },
            quality: ReceiptQuality {
                parity_max_error: 0.0,
                perplexity_delta: 0.0,
                greedy_token_agreement: 1.0,
                structured_tool_agreement: 1.0,
                needle_retrieval: 1.0,
                multi_turn_prompt_cache: 1.0,
                statistics: ReceiptQualityStatistics {
                    repeats: 5,
                    warmups: 2,
                    confidence_interval: "95% bootstrap".into(),
                    outlier_policy: "report all samples; no silent deletion".into(),
                    variance_policy: "all raw repeats retained".into(),
                    max_coefficient_of_variation: 0.05,
                },
                fixture_evidence: std::collections::BTreeMap::new(),
            },
            lifecycle: ReceiptLifecycle {
                append: true,
                chunked_prefill: true,
                single_shot_prefill: true,
                prompt_cache_reuse: true,
                trim: true,
                rollback: true,
                clear: true,
                cancel: true,
                clone: true,
                batch_split: true,
                batch_merge: true,
                prefix_copy_on_write: true,
                page_import: true,
                page_export: true,
                serialization: true,
                restore: true,
                dense_fallback: true,
                post_run_release: true,
                fallback_reasons: std::collections::BTreeMap::new(),
            },
            cancellation: ReceiptCancellation {
                cleanup_verified: true,
            },
        };
        let bundle = assemble_artifacts(receipt).unwrap();
        assert!(validate_artifact_bundle(&bundle).is_ok());
        let mut tampered = bundle.clone();
        tampered.receipt.push(b'x');
        assert!(validate_artifact_bundle(&tampered).is_err());
    }
}

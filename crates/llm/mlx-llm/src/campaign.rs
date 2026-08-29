//! Source-owned evidence primitives for the SC-20671 dense baseline campaign.
//!
//! This module deliberately keeps the receipt producer beside the product decoder.  The JSON
//! harness can validate evidence, but it must not be the component inventing model identity,
//! geometry, or lifecycle observations.  Device collection is supplied by the campaign runner;
//! these helpers make its inputs deterministic and testable without weights or Metal.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

use core_llm::{Message, Role, Sampling, StreamEvent, TextLlmOutput, TextLlmRequest, ToolSpec};

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
    "03c44b0f12caf79c1560e29fcfe536e2d7fd57153add4f3958697057b10116d2";

pub const CONTEXT_BANDS: [&str; 4] = ["short", "medium", "memory-material", "fit-boundary"];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptMatrix {
    pub family: String,
    pub context_band: String,
    pub request_mode: String,
    pub prefill_mode: String,
    pub process_temperature: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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

/// Geometry copied from the loaded decoder configuration, not an input JSON field.  Query/KV
/// lengths and batch are supplied by the actual request immediately before dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductGeometry {
    pub query_heads: u64,
    pub kv_heads: u64,
    pub head_dimension: u64,
    pub layers: u64,
    pub element_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptMlx {
    pub source: String,
    pub active_bytes: u64,
    pub cache_bytes: u64,
    pub peak_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptPhase {
    pub phase: String,
    pub pid: u32,
    pub source: String,
    pub timestamp: String,
    pub phys_footprint_bytes: u64,
    pub phys_footprint_peak_bytes: u64,
    pub mlx: ReceiptMlx,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptAllocation {
    pub kind: String,
    pub role: String,
    pub lifetime: String,
    pub phase: String,
    pub timestamp: String,
    pub bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptReconciliation {
    pub expected_dense_kv_bytes: u64,
    pub observed_persistent_kv_bytes: u64,
    pub tolerance_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptRelease {
    pub verified: bool,
    pub phys_footprint_tolerance_bytes: u64,
    pub mlx_active_tolerance_bytes: u64,
    pub mlx_cache_tolerance_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptTimingSample {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    pub cold_compile_ms: f64,
    pub warm_compile_ms: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptTimingSummary {
    pub decode_tokens_per_second_mean: f64,
    pub decode_tokens_per_second_p95: f64,
    pub decode_tokens_per_second_variance: f64,
    pub decode_tokens_per_second_coefficient_of_variation: f64,
    pub confidence_interval_low: f64,
    pub confidence_interval_high: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptFixture {
    pub passed: bool,
    pub artifact_name: String,
    pub artifact_sha256: String,
    pub artifact_sidecar_sha256: String,
    pub independent_reference: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptQualityStatistics {
    pub repeats: u64,
    pub warmups: u64,
    pub confidence_interval: String,
    pub outlier_policy: String,
    pub variance_policy: String,
    pub max_coefficient_of_variation: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
// `fallback_reasons` is flattened intentionally; semantic validation below enforces its allowlist.
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCancellation {
    pub cleanup_verified: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptWarmup {
    pub required: bool,
    pub completed: bool,
    pub worker_pid: u32,
    pub suite_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
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
    pub warmup: ReceiptWarmup,
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
        let digest = |v: &str| {
            v.len() == 64
                && v.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        let revision = |v: &str| {
            v.len() == 40
                && v.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
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
        let mut bytes = canonical_json_bytes(&serde_json::to_value(self)?)?;
        bytes.push(b'\n');
        Ok(bytes)
    }
}

/// The portable receipt identity is the same canonical JSON used by SceneWorks: recursively
/// sorted object keys, two-space indentation, and no trailing newline in the semantic core.
/// File bytes add exactly one newline in [`Receipt::bytes`].  Do not replace this with serde's
/// struct-order serialization: the JS validator intentionally does not trust insertion order.
pub fn canonical_json_bytes(value: &serde_json::Value) -> Result<Vec<u8>, serde_json::Error> {
    fn stable(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Array(values) => {
                serde_json::Value::Array(values.iter().map(stable).collect())
            }
            serde_json::Value::Object(values) => {
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                let mut object = serde_json::Map::new();
                for key in keys {
                    object.insert(key.clone(), stable(&values[key]));
                }
                serde_json::Value::Object(object)
            }
            _ => value.clone(),
        }
    }
    serde_json::to_vec_pretty(&stable(value))
}

pub fn validate_receipt_semantics(receipt: &Receipt) -> Result<(), String> {
    let rfc3339 = |v: &str| {
        let b = v.as_bytes();
        let fixed = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
        let shape = (b.len() == 20 || (b.len() >= 22 && b[19] == b'.' && b[b.len() - 1] == b'Z'))
            && b.get(4) == Some(&b'-')
            && b.get(7) == Some(&b'-')
            && b.get(10) == Some(&b'T')
            && b.get(13) == Some(&b':')
            && b.get(16) == Some(&b':')
            && b.last() == Some(&b'Z')
            && (0..4).all(fixed)
            && (5..7).all(fixed)
            && (8..10).all(fixed)
            && (11..13).all(fixed)
            && (14..16).all(fixed)
            && (17..19).all(fixed)
            && (b.len() == 20 || (20..b.len() - 1).all(fixed));
        if !shape {
            return false;
        }
        let parse =
            |range: std::ops::Range<usize>| v.get(range).and_then(|part| part.parse::<u32>().ok());
        let year = parse(0..4).unwrap_or(0);
        let month = parse(5..7).unwrap_or(0);
        let day = parse(8..10).unwrap_or(0);
        let hour = parse(11..13).unwrap_or(99);
        let minute = parse(14..16).unwrap_or(99);
        let second = parse(17..19).unwrap_or(99);
        let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
        let days = [
            0,
            31,
            if leap { 29 } else { 28 },
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ];
        (1..=12).contains(&month)
            && day >= 1
            && day <= days[month as usize]
            && hour < 24
            && minute < 60
            && second < 60
    };
    let lowercase_hex = |v: &str, len: usize| {
        v.len() == len
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let revision = |v: &str| {
        v.len() == 40
            && v.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    if receipt.schema_version != 3
        || receipt.harness_version != "sc-20671-kv-baseline-v3"
        || receipt.status != "complete"
        || receipt.contract_hash != QUALITY_CONTRACT_HASH
    {
        return Err("receipt constants mismatch".into());
    }
    if !["dense", "compressed"].contains(&receipt.mode.as_str())
        || receipt.run_id.is_empty()
        || !rfc3339(&receipt.captured_at)
    {
        return Err("receipt timestamp/run id is malformed".into());
    }
    let p = &receipt.provenance;
    if !revision(&p.scene_works_revision)
        || !revision(&p.inference_revision)
        || !lowercase_hex(&p.dependency_lock_sha256, 64)
        || !lowercase_hex(&p.model_file_sha256, 64)
        || [
            p.mlx_revision.as_str(),
            p.os.as_str(),
            p.xcode.as_str(),
            p.hardware.as_str(),
            p.model_id.as_str(),
            p.power_mode.as_str(),
            p.thermal_state.as_str(),
            p.command_template.as_str(),
            p.command.as_str(),
        ]
        .iter()
        .any(|v| v.is_empty())
        || p.model_file_bytes == 0
        || p.thermal_state != "nominal"
        || !p.command_template.contains("{mode}")
        || p.command_template.replace("{mode}", &receipt.mode) != p.command
    {
        return Err("provenance is incomplete".into());
    }
    let g = &receipt.geometry;
    if [
        g.batch,
        g.query_heads,
        g.kv_heads,
        g.head_dimension,
        g.query_length,
        g.kv_length,
        g.layers,
        g.element_bytes,
        g.capacity,
    ]
    .into_iter()
    .any(|v| v == 0)
    {
        return Err("geometry contains zero".into());
    }
    if !["llama", "qwen"].contains(&receipt.matrix.family.as_str())
        || !CONTEXT_BANDS.contains(&receipt.matrix.context_band.as_str())
        || !["single", "supported-batch"].contains(&receipt.matrix.request_mode.as_str())
        || !["chunked", "single-shot"].contains(&receipt.matrix.prefill_mode.as_str())
        || !["cold", "warm"].contains(&receipt.matrix.process_temperature.as_str())
        || g.query_heads % g.kv_heads != 0
        || g.capacity < g.kv_length
        || (receipt.matrix.request_mode == "single" && g.batch != 1)
        || (receipt.matrix.request_mode == "supported-batch" && g.batch <= 1)
    {
        return Err("matrix coordinate or geometry relationship is invalid".into());
    }
    let pids: Vec<u32> = receipt.memory.phase_samples.iter().map(|p| p.pid).collect();
    if pids.len() != 8
        || pids.iter().any(|p| *p == 0 || *p != pids[0])
        || receipt
            .memory
            .phase_samples
            .iter()
            .zip(REQUIRED_PHASES)
            .any(|(p, expected)| {
                p.phase != expected
                    || p.source != "footprint -p"
                    || p.mlx.source != "mlx_rs::memory"
                    || p.phys_footprint_peak_bytes < p.phys_footprint_bytes
                    || !rfc3339(&p.timestamp)
            })
    {
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
    for event in &receipt.memory.allocation_events {
        if event.bytes == 0
            || !["cache", "attention-workspace", "weights", "output"].contains(&event.role.as_str())
            || !["persistent", "transient"].contains(&event.lifetime.as_str())
            || event.phase.is_empty()
            || event.kind.is_empty()
            || !REQUIRED_PHASES.contains(&event.phase.as_str())
            || !rfc3339(&event.timestamp)
        {
            return Err("allocation event is malformed".into());
        }
    }
    let max_role = |role: &str, lifetime: &str| {
        let mut phases = std::collections::BTreeMap::<&str, u64>::new();
        for event in receipt
            .memory
            .allocation_events
            .iter()
            .filter(|e| e.role == role && e.lifetime == lifetime)
        {
            *phases.entry(event.phase.as_str()).or_default() += event.bytes;
        }
        phases.values().copied().max().unwrap_or(0)
    };
    if max_role("weights", "persistent") != receipt.memory.model_weights_bytes
        || max_role("cache", "persistent") != receipt.memory.persistent_kv_bytes
        || max_role("attention-workspace", "transient") != receipt.memory.transient_workspace_bytes
    {
        return Err("allocation totals do not reconcile".into());
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
    let mut transient_by_phase = std::collections::BTreeMap::<&str, u128>::new();
    for event in receipt.memory.allocation_events.iter().filter(|e| {
        e.lifetime == "transient" && (e.role == "cache" || e.role == "attention-workspace")
    }) {
        let total = transient_by_phase.entry(event.phase.as_str()).or_default();
        *total = total
            .checked_add(u128::from(event.bytes))
            .ok_or("transient allocation total overflow")?;
    }
    if transient_by_phase
        .values()
        .copied()
        .any(|bytes| bytes.saturating_mul(10) >= u128::from(dense).saturating_mul(9))
    {
        return Err("aggregate full-cache temporary detected".into());
    }
    let weights = receipt.memory.model_weights_bytes;
    let kv = receipt.memory.persistent_kv_bytes;
    let workspace = receipt.memory.transient_workspace_bytes;
    let sample_for = |phase: &str| {
        receipt
            .memory
            .phase_samples
            .iter()
            .find(|sample| sample.phase == phase)
            .ok_or_else(|| format!("missing containment phase {phase}"))
    };
    let weights_loaded = sample_for("weights-loaded")?;
    if weights_loaded.mlx.active_bytes < weights {
        return Err("weights-loaded MLX active bytes do not contain weights".into());
    }
    let prefill = sample_for("prefill-peak")?;
    let prefill_total = weights.saturating_add(kv).saturating_add(workspace);
    if prefill.mlx.active_bytes < prefill_total || prefill.mlx.peak_bytes < prefill_total {
        return Err("prefill MLX samples do not contain attributed allocations".into());
    }
    let decode = sample_for("decode-steady")?;
    if decode.mlx.active_bytes < weights.saturating_add(kv) {
        return Err("decode MLX active bytes do not contain weights and KV".into());
    }
    if receipt.mode == "dense"
        && receipt.memory.persistent_kv_bytes.abs_diff(dense)
            > receipt.memory.reconciliation.tolerance_bytes
    {
        return Err("dense KV exceeds tolerance".into());
    }
    let start = &receipt.memory.phase_samples[0];
    let end = receipt.memory.phase_samples.last().unwrap();
    if end.phys_footprint_bytes
        > start.phys_footprint_bytes + receipt.memory.release.phys_footprint_tolerance_bytes
        || end.mlx.active_bytes
            > start.mlx.active_bytes + receipt.memory.release.mlx_active_tolerance_bytes
        || end.mlx.cache_bytes
            > start.mlx.cache_bytes + receipt.memory.release.mlx_cache_tolerance_bytes
    {
        return Err("release did not return within tolerance".into());
    }
    if receipt.timings.samples.len() != 5
        || !receipt
            .timings
            .samples
            .iter()
            .flat_map(|s| {
                [
                    s.load_ms,
                    s.prefill_ms,
                    s.ttft_ms,
                    s.first_token_ms,
                    s.decode_tokens_per_second,
                    s.cold_compile_ms,
                    s.warm_compile_ms,
                ]
            })
            .all(|v| v.is_finite() && v > 0.0)
        || ![
            receipt.timings.load_ms,
            receipt.timings.prefill_ms,
            receipt.timings.ttft_ms,
            receipt.timings.first_token_ms,
            receipt.timings.decode_tokens_per_second,
            receipt.timings.cold_compile_ms,
            receipt.timings.warm_compile_ms,
            receipt.timings.summary.decode_tokens_per_second_mean,
            receipt.timings.summary.decode_tokens_per_second_p95,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v > 0.0)
        || ![
            receipt.timings.summary.decode_tokens_per_second_variance,
            receipt
                .timings
                .summary
                .decode_tokens_per_second_coefficient_of_variation,
            receipt.timings.summary.confidence_interval_low,
            receipt.timings.summary.confidence_interval_high,
        ]
        .into_iter()
        .all(|v| v.is_finite() && v >= 0.0)
        || receipt
            .timings
            .summary
            .decode_tokens_per_second_coefficient_of_variation
            > 0.05
    {
        return Err("timing policy failed".into());
    }
    let decode_mean = receipt
        .timings
        .samples
        .iter()
        .map(|s| s.decode_tokens_per_second)
        .sum::<f64>()
        / 5.0;
    if (receipt.timings.decode_tokens_per_second - decode_mean).abs() > 1e-9 {
        return Err("timing mean is not derived".into());
    }
    let average = |f: fn(&ReceiptTimingSample) -> f64| {
        receipt.timings.samples.iter().map(f).sum::<f64>() / 5.0
    };
    if (receipt.timings.load_ms - average(|s| s.load_ms)).abs() > 1e-9
        || (receipt.timings.prefill_ms - average(|s| s.prefill_ms)).abs() > 1e-9
        || (receipt.timings.ttft_ms - average(|s| s.ttft_ms)).abs() > 1e-9
        || (receipt.timings.first_token_ms - average(|s| s.first_token_ms)).abs() > 1e-9
        || (receipt.timings.cold_compile_ms - average(|s| s.cold_compile_ms)).abs() > 1e-9
        || (receipt.timings.warm_compile_ms - average(|s| s.warm_compile_ms)).abs() > 1e-9
    {
        return Err("timing field is not derived".into());
    }
    let variance = receipt
        .timings
        .samples
        .iter()
        .map(|s| (s.decode_tokens_per_second - decode_mean).powi(2))
        .sum::<f64>()
        / 5.0;
    let cv = variance.sqrt() / decode_mean;
    let mut p95 = receipt
        .timings
        .samples
        .iter()
        .map(|s| s.decode_tokens_per_second)
        .collect::<Vec<_>>();
    p95.sort_by(f64::total_cmp);
    if (receipt.timings.summary.decode_tokens_per_second_mean - decode_mean).abs() > 1e-9
        || (receipt.timings.summary.decode_tokens_per_second_p95 - p95[4]).abs() > 1e-9
    {
        return Err("timing mean/P95 is not derived".into());
    }
    if (receipt.timings.summary.decode_tokens_per_second_variance - variance).abs() > 1e-9
        || (receipt
            .timings
            .summary
            .decode_tokens_per_second_coefficient_of_variation
            - cv)
            .abs()
            > 1e-9
        || receipt.timings.summary.confidence_interval_low > decode_mean
        || receipt.timings.summary.confidence_interval_high < decode_mean
        || receipt.timings.summary.confidence_interval_low
            > receipt.timings.summary.confidence_interval_high
    {
        return Err("timing summary derivation failed".into());
    }
    if !receipt.quality.parity_max_error.is_finite()
        || !receipt.quality.perplexity_delta.is_finite()
        || receipt.quality.parity_max_error < 0.0
        || receipt.quality.parity_max_error > 0.0001
        || receipt.quality.perplexity_delta > 0.01
        || [
            receipt.quality.greedy_token_agreement,
            receipt.quality.structured_tool_agreement,
            receipt.quality.needle_retrieval,
            receipt.quality.multi_turn_prompt_cache,
        ]
        .into_iter()
        .any(|v| !(0.0..=1.0).contains(&v))
    {
        return Err("quality thresholds failed".into());
    }
    if receipt.quality.fixture_evidence.len() != 4
        || receipt.quality.statistics.repeats != 5
        || receipt.quality.statistics.warmups != 2
    {
        return Err("quality contract evidence incomplete".into());
    }
    if receipt.quality.greedy_token_agreement < 0.999
        || receipt.quality.structured_tool_agreement < 1.0
        || receipt.quality.needle_retrieval < 1.0
        || receipt.quality.multi_turn_prompt_cache < 1.0
        || REQUIRED_FIXTURES.iter().any(|name| {
            receipt.quality.fixture_evidence.get(*name).is_none_or(|f| {
                !f.passed
                    || f.artifact_name != format!("fixtures/{name}.json")
                    || f.artifact_sha256.len() != 64
                    || !f
                        .artifact_sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b >= b'a' && b <= b'f'))
                    || f.independent_reference.is_empty()
                    || f.artifact_sidecar_sha256.len() != 64
                    || !f
                        .artifact_sidecar_sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            })
        })
    {
        return Err("quality threshold or fixture evidence failed".into());
    }
    if !receipt.memory.release.verified || !receipt.cancellation.cleanup_verified {
        return Err("release/cancellation evidence failed".into());
    }
    let warm = &receipt.warmup;
    if warm.worker_pid == 0
        || (warm.required != (receipt.matrix.process_temperature == "warm"))
        || (warm.required && (!warm.completed || warm.suite_sha256.len() != 64))
        || (!warm.required && (warm.completed || !warm.suite_sha256.is_empty()))
    {
        return Err("warm worker discipline evidence failed".into());
    }
    let lifecycle = [
        ("append", receipt.lifecycle.append),
        ("chunkedPrefill", receipt.lifecycle.chunked_prefill),
        ("singleShotPrefill", receipt.lifecycle.single_shot_prefill),
        ("promptCacheReuse", receipt.lifecycle.prompt_cache_reuse),
        ("trim", receipt.lifecycle.trim),
        ("rollback", receipt.lifecycle.rollback),
        ("clear", receipt.lifecycle.clear),
        ("cancel", receipt.lifecycle.cancel),
        ("clone", receipt.lifecycle.clone),
        ("batchSplit", receipt.lifecycle.batch_split),
        ("batchMerge", receipt.lifecycle.batch_merge),
        ("prefixCopyOnWrite", receipt.lifecycle.prefix_copy_on_write),
        ("pageImport", receipt.lifecycle.page_import),
        ("pageExport", receipt.lifecycle.page_export),
        ("serialization", receipt.lifecycle.serialization),
        ("restore", receipt.lifecycle.restore),
        ("denseFallback", receipt.lifecycle.dense_fallback),
        ("postRunRelease", receipt.lifecycle.post_run_release),
    ];
    for (name, supported) in lifecycle {
        let reason = format!("{name}FallbackReason");
        if supported && receipt.lifecycle.fallback_reasons.contains_key(&reason) {
            return Err(format!(
                "supported lifecycle capability has fallback reason: {name}"
            ));
        }
        if !supported && !receipt.lifecycle.fallback_reasons.contains_key(&reason) {
            return Err(format!(
                "unsupported lifecycle capability lacks fallback reason: {name}"
            ));
        }
    }
    if receipt
        .lifecycle
        .fallback_reasons
        .iter()
        .any(|(key, value)| {
            !lifecycle
                .iter()
                .any(|(name, _)| format!("{name}FallbackReason") == *key)
                || value.trim().is_empty()
        })
    {
        return Err("unknown or empty lifecycle fallback reason".into());
    }
    if !receipt.lifecycle.post_run_release {
        return Err("postRunRelease is required".into());
    }
    if receipt.quality.statistics.variance_policy != "all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum" || receipt.quality.statistics.confidence_interval != "95% bootstrap" || receipt.quality.statistics.outlier_policy != "report all samples; no silent deletion" || receipt.quality.statistics.max_coefficient_of_variation != 0.05 { return Err("frozen quality statistics mismatch".into()); }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundle {
    pub receipt_name: String,
    pub human_name: String,
    pub receipt: Vec<u8>,
    pub receipt_sidecar: String,
    pub human: Vec<u8>,
    pub human_sidecar: String,
    pub fixtures: Vec<SealedFixtureArtifact>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SealedFixtureArtifact {
    pub name: String,
    pub bytes: Vec<u8>,
    pub sidecar: String,
}

/// Assemble all receipt artifacts in memory before any caller writes them. This prevents a
/// partially-written receipt directory from being mistaken for a campaign result.
pub fn assemble_artifacts(receipt: Receipt) -> Result<ArtifactBundle, String> {
    assemble_artifacts_with_fixtures_named(receipt, "receipt.json", "receipt.md", Vec::new())
}

pub fn assemble_artifacts_with_fixtures(
    receipt: Receipt,
    fixtures: Vec<SealedFixtureArtifact>,
) -> Result<ArtifactBundle, String> {
    assemble_artifacts_with_fixtures_named(receipt, "receipt.json", "receipt.md", fixtures)
}

pub fn assemble_artifacts_named(
    mut receipt: Receipt,
    receipt_name: &str,
    human_name: &str,
) -> Result<ArtifactBundle, String> {
    assemble_artifacts_with_fixtures_named(receipt, receipt_name, human_name, Vec::new())
}

fn assemble_artifacts_with_fixtures_named(
    mut receipt: Receipt,
    receipt_name: &str,
    human_name: &str,
    fixtures: Vec<SealedFixtureArtifact>,
) -> Result<ArtifactBundle, String> {
    let mut semantic = serde_json::to_value(&receipt).map_err(|e| e.to_string())?;
    semantic
        .as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    let semantic = canonical_json_bytes(&semantic).map_err(|e| e.to_string())?;
    receipt.receipt_sha256 = seal_bytes(&semantic);
    let bytes = receipt.bytes().map_err(|e| e.to_string())?;
    let human = format!(
        "# {} KV receipt\\n\\n- Run: {}\\n- Mode: {}\\n- Receipt hash: {}\\n",
        if receipt.mode == "dense" {
            "Dense"
        } else {
            "Compressed"
        },
        receipt.run_id,
        receipt.mode,
        receipt.receipt_sha256
    )
    .into_bytes();
    Ok(ArtifactBundle {
        receipt_name: receipt_name.into(),
        human_name: human_name.into(),
        receipt_sidecar: format!("{}  {receipt_name}\n", seal_bytes(&bytes)),
        human_sidecar: format!("{}  {human_name}\n", seal_bytes(&human)),
        receipt: bytes,
        human,
        fixtures,
    })
}

/// Atomically publish a complete artifact set. Temporary files are confined to `directory` and
/// renamed only after all bytes and sidecars have been prepared.
pub fn write_artifacts(
    directory: &Path,
    receipt_name: &str,
    human_name: &str,
    bundle: &ArtifactBundle,
) -> std::io::Result<()> {
    if directory.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "artifact destination must be absent for atomic publication",
        ));
    }
    for name in [receipt_name, human_name] {
        if Path::new(name).file_name().and_then(|n| n.to_str()) != Some(name) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "artifact names must be confined basenames",
            ));
        }
    }
    if bundle.receipt_name != receipt_name || bundle.human_name != human_name {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "artifact bundle names do not match publication names",
        ));
    }
    let parent = directory.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let staging = parent.join(format!(
        ".{}.staging-{}",
        directory
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("receipt"),
        std::process::id()
    ));
    if staging.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "artifact staging path already exists",
        ));
    }
    fs::create_dir(&staging)?;
    let mut files: Vec<(String, &[u8])> = vec![
        (receipt_name.into(), &bundle.receipt),
        (human_name.into(), &bundle.human),
        (
            format!("{receipt_name}.sha256"),
            bundle.receipt_sidecar.as_bytes(),
        ),
        (
            format!("{human_name}.sha256"),
            bundle.human_sidecar.as_bytes(),
        ),
    ];
    for fixture in &bundle.fixtures {
        files.push((fixture.name.clone(), &fixture.bytes));
        files.push((
            format!("{}.sha256", fixture.name),
            fixture.sidecar.as_bytes(),
        ));
    }
    let result = (|| {
        for (name, bytes) in files {
            if let Some(parent) = staging.join(&name).parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(staging.join(&name), bytes)?;
        }
        fs::rename(&staging, directory)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

pub fn validate_artifact_bundle(bundle: &ArtifactBundle) -> Result<(), String> {
    validate_artifact_bundle_named(bundle, &bundle.receipt_name, &bundle.human_name)
}

pub fn validate_artifact_bundle_named(
    bundle: &ArtifactBundle,
    receipt_name: &str,
    human_name: &str,
) -> Result<(), String> {
    if bundle.receipt_name != receipt_name
        || bundle.human_name != human_name
        || bundle.receipt_sidecar != format!("{}  {receipt_name}\n", seal_bytes(&bundle.receipt))
        || bundle.human_sidecar != format!("{}  {human_name}\n", seal_bytes(&bundle.human))
    {
        return Err("artifact sidecar does not match exact bytes".into());
    }
    if bundle.receipt.is_empty() || bundle.human.is_empty() {
        return Err("partial artifact bundle".into());
    }
    for fixture in &bundle.fixtures {
        if fixture.name.starts_with('/')
            || fixture.name.contains("..")
            || fixture.bytes.is_empty()
            || fixture.sidecar != format!("{}  {}\n", seal_bytes(&fixture.bytes), fixture.name)
        {
            return Err("fixture artifact or sidecar is malformed".into());
        }
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bundle.receipt).map_err(|e| e.to_string())?;
    let typed: Receipt =
        serde_json::from_value(value.clone()).map_err(|e| format!("receipt decode: {e}"))?;
    validate_receipt_semantics(&typed)?;
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
    for (name, evidence) in &typed.quality.fixture_evidence {
        let artifact = bundle
            .fixtures
            .iter()
            .find(|artifact| artifact.name == evidence.artifact_name)
            .ok_or_else(|| format!("receipt fixture {name} has no published artifact"))?;
        if evidence.artifact_sha256 != seal_bytes(&artifact.bytes)
            || evidence.artifact_sidecar_sha256 != seal_bytes(artifact.sidecar.as_bytes())
        {
            return Err(format!(
                "receipt fixture {name} is not bound to exact artifact bytes"
            ));
        }
    }
    let mut core = value.clone();
    core.as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    let expected = seal_bytes(&canonical_json_bytes(&core).map_err(|e| e.to_string())?);
    if object.get("receiptSha256").and_then(|v| v.as_str()) != Some(expected.as_str()) {
        return Err("receipt semantic hash mismatch".into());
    }
    let human = std::str::from_utf8(&bundle.human).map_err(|_| "human receipt is not UTF-8")?;
    if !human.contains(&format!("- Receipt hash: {expected}\\n")) {
        return Err("human receipt is not bound to the sealed JSON receipt".into());
    }
    Ok(())
}

/// A receipt prepared by a child worker, before the parent makes the campaign visible.  The
/// parent deliberately receives the sealed in-memory artifacts rather than a caller-authored
/// summary: a child crash therefore cannot leave a partially complete baseline at the requested
/// output path.
#[derive(Clone, Debug)]
pub struct PreparedCoordinateReceipt {
    pub coordinate: Coordinate,
    pub bundle: ArtifactBundle,
}

fn coordinate_slug(coordinate: &Coordinate) -> String {
    format!(
        "{}-{}-{}-{}-{}",
        coordinate.family,
        coordinate.context_band,
        coordinate.request_mode,
        coordinate.prefill_mode,
        coordinate.process_temperature
    )
}

/// Atomically publish the *whole* 64-coordinate receipt collection.  Individual worker output is
/// intentionally not a campaign result; only this function creates `destination`, and it does so
/// after every receipt, sidecar, coordinate, and product-owned worker PID has been validated.
pub fn publish_complete_campaign(
    destination: &Path,
    prepared: &[PreparedCoordinateReceipt],
) -> Result<(), String> {
    if destination.exists() {
        return Err("campaign destination must be absent for atomic publication".into());
    }
    let schedule = required_schedule();
    if prepared.len() != schedule.len() {
        return Err("complete campaign requires exactly 64 prepared receipts".into());
    }
    let mut outcomes = Vec::with_capacity(prepared.len());
    let mut seen = std::collections::BTreeSet::new();
    for item in prepared {
        validate_artifact_bundle(&item.bundle)?;
        let receipt: Receipt = serde_json::from_slice(&item.bundle.receipt)
            .map_err(|e| format!("prepared receipt is not decodable: {e}"))?;
        let coordinate = Coordinate {
            family: match receipt.matrix.family.as_str() {
                "llama" => "llama",
                "qwen" => "qwen",
                _ => return Err("prepared receipt has unsupported family".into()),
            },
            context_band: match receipt.matrix.context_band.as_str() {
                "short" => "short",
                "medium" => "medium",
                "memory-material" => "memory-material",
                "fit-boundary" => "fit-boundary",
                _ => return Err("prepared receipt has unsupported context band".into()),
            },
            request_mode: match receipt.matrix.request_mode.as_str() {
                "single" => "single",
                "supported-batch" => "supported-batch",
                _ => return Err("prepared receipt has unsupported request mode".into()),
            },
            prefill_mode: match receipt.matrix.prefill_mode.as_str() {
                "chunked" => "chunked",
                "single-shot" => "single-shot",
                _ => return Err("prepared receipt has unsupported prefill mode".into()),
            },
            process_temperature: match receipt.matrix.process_temperature.as_str() {
                "cold" => "cold",
                "warm" => "warm",
                _ => return Err("prepared receipt has unsupported process temperature".into()),
            },
        };
        if coordinate != item.coordinate || !seen.insert(coordinate_slug(&coordinate)) {
            return Err("prepared receipt coordinate disagrees with immutable schedule".into());
        }
        let discipline = if coordinate.process_temperature == "cold" {
            ProcessDiscipline::FreshChild
        } else {
            ProcessDiscipline::ReusedWarmWorker
        };
        let pid = receipt
            .memory
            .phase_samples
            .first()
            .ok_or("prepared receipt has no product-owned process phase")?
            .pid;
        outcomes.push((coordinate, discipline, pid));
    }
    validate_schedule_outcomes(&schedule, &outcomes)?;

    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let staging = parent.join(format!(
        ".{}.campaign-staging-{}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("sc20671"),
        std::process::id()
    ));
    if staging.exists() {
        return Err("campaign staging path already exists".into());
    }
    fs::create_dir(&staging).map_err(|e| e.to_string())?;
    let result = (|| -> Result<(), String> {
        let mut manifest_rows = Vec::with_capacity(prepared.len());
        for item in prepared {
            let slug = coordinate_slug(&item.coordinate);
            write_artifacts(
                &staging.join(&slug),
                &item.bundle.receipt_name,
                &item.bundle.human_name,
                &item.bundle,
            )
            .map_err(|e| e.to_string())?;
            let receipt: Receipt =
                serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
            let mut files = vec![
                serde_json::json!({"name": item.bundle.receipt_name, "sha256": seal_bytes(&item.bundle.receipt), "sidecarSha256": seal_bytes(item.bundle.receipt_sidecar.as_bytes())}),
                serde_json::json!({"name": item.bundle.human_name, "sha256": seal_bytes(&item.bundle.human), "sidecarSha256": seal_bytes(item.bundle.human_sidecar.as_bytes())}),
            ];
            for fixture in &item.bundle.fixtures {
                files.push(serde_json::json!({"name": fixture.name, "sha256": seal_bytes(&fixture.bytes), "sidecarSha256": seal_bytes(fixture.sidecar.as_bytes())}));
            }
            manifest_rows.push(serde_json::json!({
                "coordinate": slug,
                "receiptSha256": receipt.receipt_sha256,
                "workerPid": receipt.memory.phase_samples[0].pid,
                "files": files,
            }));
        }
        let manifest = serde_json::json!({
            "schemaVersion": 1,
            "kind": "sc-20671-complete-coordinate-set",
            "coordinates": manifest_rows,
        });
        let manifest_bytes = canonical_json_bytes(&manifest).map_err(|e| e.to_string())?;
        fs::write(staging.join("campaign.json"), &manifest_bytes).map_err(|e| e.to_string())?;
        fs::write(
            staging.join("campaign.json.sha256"),
            format!("{}  campaign.json\n", seal_bytes(&manifest_bytes)),
        )
        .map_err(|e| e.to_string())?;
        fs::rename(&staging, destination).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

/// Load a child-produced sealed set for the parent transaction.  This is intentionally strict:
/// workers communicate only through byte-sealed receipt artifacts, not through an unsealed stdout
/// summary or caller-provided identity fields.
pub fn load_prepared_coordinate_receipt(
    directory: &Path,
    coordinate: Coordinate,
) -> Result<PreparedCoordinateReceipt, String> {
    let receipt = fs::read(directory.join("receipt.json")).map_err(|e| e.to_string())?;
    let human = fs::read(directory.join("receipt.md")).map_err(|e| e.to_string())?;
    let receipt_sidecar =
        fs::read_to_string(directory.join("receipt.json.sha256")).map_err(|e| e.to_string())?;
    let human_sidecar =
        fs::read_to_string(directory.join("receipt.md.sha256")).map_err(|e| e.to_string())?;
    let typed: Receipt = serde_json::from_slice(&receipt).map_err(|e| e.to_string())?;
    let mut fixtures = Vec::with_capacity(typed.quality.fixture_evidence.len());
    for evidence in typed.quality.fixture_evidence.values() {
        let bytes = fs::read(directory.join(&evidence.artifact_name)).map_err(|e| e.to_string())?;
        let sidecar =
            fs::read_to_string(directory.join(format!("{}.sha256", evidence.artifact_name)))
                .map_err(|e| e.to_string())?;
        fixtures.push(SealedFixtureArtifact {
            name: evidence.artifact_name.clone(),
            bytes,
            sidecar,
        });
    }
    let bundle = ArtifactBundle {
        receipt_name: "receipt.json".into(),
        human_name: "receipt.md".into(),
        receipt,
        receipt_sidecar,
        human,
        human_sidecar,
        fixtures,
    };
    validate_artifact_bundle(&bundle)?;
    Ok(PreparedCoordinateReceipt { coordinate, bundle })
}

/// Immutable parent inputs.  The only model-related choices are snapshot paths; model identity,
/// geometry, memory, timing, and quality fields are collected by the child from the loaded product
/// and are never accepted by this command-line boundary.
#[derive(Clone, Debug)]
pub struct CampaignLaunch {
    pub executable: PathBuf,
    pub llama_snapshot: PathBuf,
    pub qwen_snapshot: PathBuf,
    pub fp32_reference_snapshot: PathBuf,
    pub prompt_file: PathBuf,
    pub destination: PathBuf,
}

/// Spawn every immutable matrix row.  A cold row receives a brand-new child process.  Warm rows
/// also run in a child, but that child is required to perform an in-process warm-up before its
/// measured lifecycle; its receipt PID is later checked by [`publish_complete_campaign`].  Child
/// results live in a hidden staging root and are deleted on *any* failure, so incomplete sets can
/// never appear at the requested destination.
pub fn launch_complete_campaign(launch: &CampaignLaunch) -> Result<(), String> {
    if launch.destination.exists() {
        return Err("campaign destination already exists".into());
    }
    for path in [
        &launch.executable,
        &launch.llama_snapshot,
        &launch.qwen_snapshot,
        &launch.fp32_reference_snapshot,
        &launch.prompt_file,
    ] {
        if !path.exists() {
            return Err(format!(
                "required campaign input is absent: {}",
                path.display()
            ));
        }
    }
    let parent = launch
        .destination
        .parent()
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let worker_root = parent.join(format!(".sc20671-workers-{}", std::process::id()));
    if worker_root.exists() {
        return Err("worker staging root already exists".into());
    }
    fs::create_dir(&worker_root).map_err(|e| e.to_string())?;
    let schedule = required_schedule();
    let result = (|| -> Result<(), String> {
        let mut prepared = Vec::with_capacity(schedule.len());
        let mut failures = Vec::new();
        for (index, row) in schedule.iter().enumerate() {
            let child_dir = worker_root.join(coordinate_slug(&row.coordinate));
            let snapshot = if row.coordinate.family == "llama" {
                &launch.llama_snapshot
            } else {
                &launch.qwen_snapshot
            };
            let output = Command::new(&launch.executable)
                .arg("worker")
                .arg("--coordinate-index")
                .arg(index.to_string())
                .arg("--snapshot")
                .arg(snapshot)
                .arg("--prompt-file")
                .arg(&launch.prompt_file)
                .arg("--fp32-reference-snapshot")
                .arg(&launch.fp32_reference_snapshot)
                .arg("--out")
                .arg(&child_dir)
                .output()
                .map_err(|e| format!("launch worker {index}: {e}"))?;
            if !output.status.success() {
                failures.push(format!(
                    "{}: {}",
                    coordinate_slug(&row.coordinate),
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
                continue;
            }
            prepared.push(load_prepared_coordinate_receipt(
                &child_dir,
                row.coordinate.clone(),
            )?);
        }
        if !failures.is_empty() {
            return Err(format!(
                "{} of 64 product workers failed; no campaign was published: {}",
                failures.len(),
                failures.join("; ")
            ));
        }
        publish_complete_campaign(&launch.destination, &prepared)
    })();
    let _ = fs::remove_dir_all(&worker_root);
    result
}

fn required_flag(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then(|| window[1].clone()))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {name}"))
}

/// CLI entrypoint used by the standalone `sc20671-kv-baseline` binary.  The child mode remains
/// fail-closed until the decoder exposes its real numeric quality-reference hooks; it still runs
/// the complete product lifecycle before refusing publication, rather than manufacturing a
/// caller-authored receipt.  This keeps the executable safe to invoke while making the missing
/// product measurement surface explicit.
pub fn sc20671_cli(args: &[String]) -> Result<(), String> {
    let Some(mode) = args.first().map(String::as_str) else {
        return Err("usage: sc20671-kv-baseline parent|worker [options]".into());
    };
    match mode {
        "parent" => launch_complete_campaign(&CampaignLaunch {
            executable: std::env::current_exe().map_err(|e| e.to_string())?,
            llama_snapshot: PathBuf::from(required_flag(args, "--llama-snapshot")?),
            qwen_snapshot: PathBuf::from(required_flag(args, "--qwen-snapshot")?),
            fp32_reference_snapshot: PathBuf::from(required_flag(
                args,
                "--fp32-reference-snapshot",
            )?),
            prompt_file: PathBuf::from(required_flag(args, "--prompt-file")?),
            destination: PathBuf::from(required_flag(args, "--out")?),
        }),
        "worker" => {
            let index = required_flag(args, "--coordinate-index")?
                .parse::<usize>()
                .map_err(|_| "--coordinate-index must be an integer".to_string())?;
            let row = required_schedule()
                .get(index)
                .cloned()
                .ok_or("--coordinate-index is outside the frozen 64-coordinate matrix")?;
            let snapshot = PathBuf::from(required_flag(args, "--snapshot")?);
            let reference_snapshot =
                PathBuf::from(required_flag(args, "--fp32-reference-snapshot")?);
            let prompt = fs::read_to_string(required_flag(args, "--prompt-file")?)
                .map_err(|e| format!("read prompt file: {e}"))?;
            if prompt.trim().is_empty() {
                return Err("campaign prompt must not be empty".into());
            }
            // Warm rows run the exact same product fixture suite once before collection in this
            // child.  The measured suite is therefore a genuine same-worker warm observation;
            // cold rows intentionally skip this path and are launched in fresh children.
            let warmup_suite_sha256 = if row.coordinate.process_temperature == "warm" {
                let suite = run_product_fixture_suite(
                    &snapshot,
                    &reference_snapshot,
                    &prompt,
                    &row.coordinate,
                )
                .map_err(|e| {
                    format!(
                        "product warmup for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?
                .quality()
                .map_err(|e| {
                    format!(
                        "product warmup quality for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
                Some(seal_bytes(&canonical_json_bytes(&serde_json::json!({"workerPid": std::process::id(), "kernel": suite.kernel_candidate.output.text})).map_err(|e| e.to_string())?))
            } else {
                None
            };
            let mut suites = Vec::with_capacity(5);
            for repeat in 0..5 {
                let suite = run_product_fixture_suite(
                    &snapshot,
                    &reference_snapshot,
                    &prompt,
                    &row.coordinate,
                )
                .map_err(|e| {
                    format!(
                        "product fixture repeat {repeat} for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
                // Every raw fixture is independently quality-validated before any timing or
                // aggregate publication decision.  There is no successful partial-repeat path.
                suite.quality()?;
                suites.push(suite);
            }
            let kernel_runs = suites
                .iter()
                .map(|suite| &suite.kernel_candidate)
                .collect::<Vec<_>>();
            let timings = timing_samples_from_product_repeats(
                &kernel_runs,
                row.coordinate.process_temperature,
            )?;
            let executable = std::env::current_exe().map_err(|e| e.to_string())?;
            let fixtures = sealed_product_fixture_artifacts(&suites[0])?;
            let receipt = product_receipt(
                &row.coordinate,
                &suites[0],
                timings,
                &executable,
                &fixtures,
                warmup_suite_sha256,
            )?;
            let bundle = assemble_artifacts_with_fixtures(receipt, fixtures)?;
            write_artifacts(
                &PathBuf::from(required_flag(args, "--out")?),
                "receipt.json",
                "receipt.md",
                &bundle,
            )
            .map_err(|e| e.to_string())
        }
        _ => Err("usage: sc20671-kv-baseline parent|worker [options]".into()),
    }
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

/// Process discipline for a matrix coordinate.  A cold result is invalid unless it came from a
/// brand-new worker; a warm result is invalid unless it follows an in-worker warmup using the same
/// loaded model/session.  This is intentionally part of the checked-in schedule rather than a
/// free-form receipt label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProcessDiscipline {
    FreshChild,
    ReusedWarmWorker,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScheduledCoordinate {
    pub coordinate: Coordinate,
    pub discipline: ProcessDiscipline,
}

/// The parent executable consumes exactly this schedule.  It never accepts a caller-supplied row
/// list, which prevents a selectively successful campaign from being published as the baseline.
pub fn required_schedule() -> Vec<ScheduledCoordinate> {
    required_coordinates()
        .into_iter()
        .map(|coordinate| ScheduledCoordinate {
            discipline: if coordinate.process_temperature == "cold" {
                ProcessDiscipline::FreshChild
            } else {
                ProcessDiscipline::ReusedWarmWorker
            },
            coordinate,
        })
        .collect()
}

/// Validate worker outcomes before aggregate publication.  `pid` is read from the product-owned
/// receipt phase samples, never supplied separately by the orchestrator.  Cold worker PID reuse is
/// rejected so a warmed model cannot be relabelled as cold.
pub fn validate_schedule_outcomes(
    scheduled: &[ScheduledCoordinate],
    outcomes: &[(Coordinate, ProcessDiscipline, u32)],
) -> Result<(), String> {
    if scheduled.len() != 64 || outcomes.len() != scheduled.len() {
        return Err("campaign schedule must contain exactly 64 outcomes".into());
    }
    let key = |coordinate: &Coordinate| {
        format!(
            "{}/{}/{}/{}/{}",
            coordinate.family,
            coordinate.context_band,
            coordinate.request_mode,
            coordinate.prefill_mode,
            coordinate.process_temperature
        )
    };
    let expected = scheduled
        .iter()
        .map(|row| (key(&row.coordinate), row.discipline))
        .collect::<std::collections::BTreeMap<_, _>>();
    if expected.len() != 64 {
        return Err("campaign schedule contains duplicate coordinates".into());
    }
    let mut actual = std::collections::BTreeMap::new();
    let mut cold_pids = std::collections::BTreeSet::new();
    for (coordinate, discipline, pid) in outcomes {
        if *pid == 0 || actual.insert(key(coordinate), *discipline).is_some() {
            return Err("campaign outcome has a zero PID or duplicate coordinate".into());
        }
        if *discipline == ProcessDiscipline::FreshChild && !cold_pids.insert(*pid) {
            return Err("cold coordinate reused a worker PID".into());
        }
    }
    if actual != expected {
        return Err("campaign outcomes do not equal the required matrix".into());
    }
    Ok(())
}

/// The only scheduling seam.  Production supplies a child-process launcher; tests may use a fake
/// worker, but it receives the immutable schedule rather than inventing rows or temperatures.
pub(crate) fn execute_required_schedule<F>(
    mut worker: F,
) -> Result<Vec<(Coordinate, ProcessDiscipline, u32)>, String>
where
    F: FnMut(&ScheduledCoordinate) -> Result<u32, String>,
{
    let schedule = required_schedule();
    let mut outcomes = Vec::with_capacity(schedule.len());
    for row in &schedule {
        let pid = worker(row)?;
        outcomes.push((row.coordinate.clone(), row.discipline, pid));
    }
    validate_schedule_outcomes(&schedule, &outcomes)?;
    Ok(outcomes)
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
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == ".git"
                || name == ".DS_Store"
                || name.ends_with(".lock")
                || name.ends_with(".tmp")
                || name.ends_with(".partial")
            {
                continue;
            }
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
    let names = files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    if !names.contains("config.json") || !names.contains("tokenizer.json") {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot is missing config.json or tokenizer.json",
        ));
    }
    if !files.iter().any(|file| file.path.ends_with(".safetensors")) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "snapshot has no resolved safetensors shard",
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
    let ratio = |matched: u64, total: u64| -> Result<f64, String> {
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

/// Seal the evidence belonging to one product fixture.  The aggregate quality row is not a
/// substitute for this: each artifact carries the source/reference facts that were actually
/// compared for its named fixture.
fn product_fixture_artifact(
    name: &str,
    suite: &ProductFixtureSuite,
) -> Result<(Vec<u8>, String), String> {
    let quality = suite.quality()?;
    let (evidence, reference) = match name {
        "kernel-fp32-reference" => (
            serde_json::json!({
                "candidatePerplexity": quality.candidate_perplexity,
                "referencePerplexity": quality.reference_perplexity,
                "parityErrors": quality.parity_errors,
                "greedyMatches": quality.greedy_matches,
                "greedyTotal": quality.greedy_total,
            }),
            suite.kernel_reference.observation.snapshot.sha256.clone(),
        ),
        "structured-tool-call" => (
            serde_json::json!({ "matches": quality.tool_matches, "total": quality.tool_total }),
            suite.tool_reference.observation.snapshot.sha256.clone(),
        ),
        "long-context-needle" => (
            serde_json::json!({ "matches": quality.needle_matches, "total": quality.needle_total }),
            suite.needle_reference.observation.snapshot.sha256.clone(),
        ),
        "multi-turn-prompt-cache" => (
            serde_json::json!({ "matches": quality.cache_matches, "total": quality.cache_total }),
            suite.cache_reference.observation.snapshot.sha256.clone(),
        ),
        _ => return Err(format!("unknown fixture {name}")),
    };
    let metrics = compute_quality(&quality)?;
    let value = serde_json::json!({
        "fixture": name,
        "independentReference": format!("fp32-snapshot:{reference}"),
        "evidence": evidence,
        "metrics": {
            "parityMaxError": metrics.parity_max_error,
            "perplexityDelta": metrics.perplexity_delta,
            "greedyTokenAgreement": metrics.greedy_token_agreement,
            "structuredToolAgreement": metrics.structured_tool_agreement,
            "needleRetrieval": metrics.needle_retrieval,
            "multiTurnPromptCache": metrics.multi_turn_prompt_cache,
        },
    });
    Ok(sealed_json(&value))
}

fn sealed_product_fixture_artifacts(
    suite: &ProductFixtureSuite,
) -> Result<Vec<SealedFixtureArtifact>, String> {
    REQUIRED_FIXTURES
        .iter()
        .map(|fixture| {
            let (bytes, hash) = product_fixture_artifact(fixture, suite)?;
            let name = format!("fixtures/{fixture}.json");
            Ok(SealedFixtureArtifact {
                sidecar: format!("{hash}  {name}\n"),
                name,
                bytes,
            })
        })
        .collect()
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
    /// Exact fp32 logits materialized from the production decode seam only while a campaign
    /// observer is attached.  Ordinary serving never takes this host-read path.
    fn logits(&mut self, _stage: &'static str, _values: &[f32]) {}
    /// Exact probability of the sampled production token, derived at the same decode seam as the
    /// sampler.  It is intentionally not a caller-provided quality number.
    fn token_probability(&mut self, _stage: &'static str, _token: i32, _probability: f64) {}
    /// Model identity is derived from the exact files resolved by the product loader, never a
    /// caller-authored digest string.
    fn snapshot_inventory(&mut self, _inventory: &SnapshotInventory) {}
    fn geometry(&mut self, _geometry: ProductGeometry) {}
}

/// Device-backed product observer.  It deliberately has no setters for identity, phase samples,
/// or allocation records: receipt assembly consumes only observations made at product boundaries.
pub struct ProductObserver {
    pid: u32,
    started: std::time::Instant,
    phase: Option<&'static str>,
    phases: Vec<ReceiptPhase>,
    phase_elapsed_ms: Vec<f64>,
    allocations: Vec<ReceiptAllocation>,
    snapshot: Option<SnapshotInventory>,
    geometry: Option<ProductGeometry>,
    prefill_logits: Option<Vec<f32>>,
    token_probabilities: Vec<(i32, f64)>,
    error: Option<String>,
}

impl ProductObserver {
    pub fn new() -> Self {
        Self {
            pid: std::process::id(),
            started: std::time::Instant::now(),
            phase: None,
            phases: Vec::new(),
            phase_elapsed_ms: Vec::new(),
            allocations: Vec::new(),
            snapshot: None,
            geometry: None,
            prefill_logits: None,
            token_probabilities: Vec::new(),
            error: None,
        }
    }

    pub fn finish(self) -> Result<ProductObservations, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.snapshot.is_none() || self.geometry.is_none() {
            return Err("product observer is missing snapshot identity or loaded geometry".into());
        }
        let prefill_logits = self
            .prefill_logits
            .ok_or("product observer did not receive production prefill logits")?;
        if prefill_logits.is_empty()
            || prefill_logits.iter().any(|value| !value.is_finite())
            || self.token_probabilities.is_empty()
            || self.token_probabilities.iter().any(|(_, probability)| {
                !probability.is_finite() || !(0.0..=1.0).contains(probability)
            })
        {
            return Err("product observer has invalid numeric decode evidence".into());
        }
        if self.phases.len() != REQUIRED_PHASES.len() {
            return Err("product observer did not capture the exact phase set".into());
        }
        Ok(ProductObservations {
            snapshot: self.snapshot.expect("checked above"),
            geometry: self.geometry.expect("checked above"),
            phases: self.phases,
            phase_elapsed_ms: self.phase_elapsed_ms,
            allocations: self.allocations,
            prefill_logits,
            token_probabilities: self.token_probabilities,
        })
    }
}

impl Default for ProductObserver {
    fn default() -> Self {
        Self::new()
    }
}

pub struct ProductObservations {
    pub snapshot: SnapshotInventory,
    pub geometry: ProductGeometry,
    pub phases: Vec<ReceiptPhase>,
    pub phase_elapsed_ms: Vec<f64>,
    pub allocations: Vec<ReceiptAllocation>,
    pub prefill_logits: Vec<f32>,
    pub token_probabilities: Vec<(i32, f64)>,
}

impl Observer for ProductObserver {
    fn phase(&mut self, name: &'static str) {
        let expected = REQUIRED_PHASES.get(self.phases.len()).copied();
        if self.error.is_some() || expected != Some(name) {
            self.error = Some(format!("duplicate or out-of-order product phase {name}"));
            return;
        }
        match sample_memory(self.pid) {
            Ok(sample) => {
                self.phases.push(ReceiptPhase {
                    phase: name.into(),
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
                self.phase_elapsed_ms
                    .push(self.started.elapsed().as_secs_f64() * 1_000.0);
            }
            Err(error) => self.error = Some(format!("product memory sample at {name}: {error}")),
        }
        self.phase = Some(name);
    }

    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64) {
        let Some(phase) = self.phase else {
            self.error = Some("allocation observed before a product phase".into());
            return;
        };
        if bytes == 0 {
            return;
        }
        self.allocations.push(ReceiptAllocation {
            kind: format!("product-{role}"),
            role: role.into(),
            lifetime: lifetime.into(),
            phase: phase.into(),
            timestamp: timestamp_now(),
            bytes,
        });
    }

    fn logits(&mut self, stage: &'static str, values: &[f32]) {
        if stage != "prefill" || self.prefill_logits.is_some() || values.is_empty() {
            self.error = Some("invalid or duplicate production logit observation".into());
            return;
        }
        if values.iter().any(|value| !value.is_finite()) {
            self.error = Some("non-finite production logits".into());
            return;
        }
        self.prefill_logits = Some(values.to_vec());
    }

    fn token_probability(&mut self, stage: &'static str, token: i32, probability: f64) {
        if stage != "decode"
            || token < 0
            || !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
        {
            self.error = Some("invalid product token probability".into());
            return;
        }
        self.token_probabilities.push((token, probability));
    }

    fn snapshot_inventory(&mut self, inventory: &SnapshotInventory) {
        self.snapshot = Some(inventory.clone());
    }
    fn geometry(&mut self, geometry: ProductGeometry) {
        self.geometry = Some(geometry);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
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
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_MICROS: AtomicU64 = AtomicU64::new(0);

    let observed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros()
        .min(u128::from(u64::MAX)) as u64;
    let micros = loop {
        let prior = LAST_MICROS.load(Ordering::Relaxed);
        let next = observed.max(prior.saturating_add(1));
        if LAST_MICROS
            .compare_exchange_weak(prior, next, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            break next;
        }
    };
    let seconds = micros / 1_000_000;
    let fraction = micros % 1_000_000;
    // Civil date from Unix days, using the proleptic Gregorian calendar.  Keeping this local
    // avoids a second time dependency in the MLX engine while producing the exact RFC3339 shape
    // required by the paired SceneWorks validator.
    let days = (seconds / 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    year += if month <= 2 { 1 } else { 0 };
    let day_seconds = seconds % 86_400;
    let hour = day_seconds / 3_600;
    let minute = (day_seconds % 3_600) / 60;
    let second = day_seconds % 60;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{fraction:06}Z")
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
    let bytes = (number * multiplier).round();
    if !bytes.is_finite() || !(0.0..=(u64::MAX as f64)).contains(&bytes) {
        return None;
    }
    Some(bytes as u64)
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
    let inventory = inventory_snapshot(snapshot.as_ref())
        .map_err(|e| core_llm::Error::Load(format!("snapshot inventory: {e}")))?;
    if inventory.bytes == 0 || inventory.files.is_empty() {
        return Err(core_llm::Error::Load("snapshot inventory is empty".into()));
    }
    let output = {
        let provider = crate::provider::LlamaProvider::load(&core_llm::LoadSpec::dense(
            snapshot.as_ref().to_string_lossy().to_string(),
        ))?;
        observer.snapshot_inventory(&inventory);
        observer.geometry(provider.campaign_geometry());
        observer.phase("weights-loaded");
        observer.allocation("weights", "persistent", inventory.bytes);
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
        output
    }; // Provider/model ownership is released before the final sample.
    observer.phase("post-run-release");
    Ok(output)
}

/// Full one-process product lifecycle used by a worker in the 64-coordinate runner.  The parent
/// runner is responsible for spawning a fresh process for every `cold` coordinate; this function
/// refuses to manufacture a receipt when an unsupported family cannot exercise a required cache
/// lifecycle.
pub fn run_dense_lifecycle(
    snapshot: impl AsRef<Path>,
    prompt: &str,
    max_new_tokens: u32,
    observer: &mut dyn Observer,
) -> core_llm::Result<TextLlmOutput> {
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
    run_dense_lifecycle_request(snapshot, prompt, request, None, observer)
}

/// Product-only fixture entrypoint.  The request is constructed in this crate (including tools and
/// multi-turn messages) and is never decoded from receipt JSON.  It shares the ordinary provider
/// and decode route while an observer is installed, so the quality fixtures cannot silently use a
/// synthetic reference path.
pub fn run_dense_lifecycle_request(
    snapshot: impl AsRef<Path>,
    prefix_prompt: &str,
    request: TextLlmRequest,
    coordinate: Option<&Coordinate>,
    observer: &mut dyn Observer,
) -> core_llm::Result<TextLlmOutput> {
    observer.phase("process-start");
    let inventory = inventory_snapshot(snapshot.as_ref())
        .map_err(|e| core_llm::Error::Load(format!("snapshot inventory: {e}")))?;
    let output = {
        let provider = crate::provider::LlamaProvider::load(&core_llm::LoadSpec::dense(
            snapshot.as_ref().to_string_lossy().to_string(),
        ))?;
        observer.snapshot_inventory(&inventory);
        observer.geometry(provider.campaign_geometry());
        observer.phase("weights-loaded");
        observer.allocation("weights", "persistent", inventory.bytes);
        if let Some(coordinate) = coordinate {
            // These are product operations on this very provider, not an out-of-band control.
            // A row cannot be measured until its requested batch/prefill behavior has executed.
            if coordinate.request_mode == "supported-batch" {
                provider.campaign_supported_batch(prefix_prompt, 2)?;
            }
            if coordinate.prefill_mode == "chunked" {
                provider.campaign_prefix_reuse(prefix_prompt)?;
            }
        }
        let mut saw_token = false;
        let output = provider.generate_observed(
            &request,
            &mut |event| saw_token |= matches!(event, StreamEvent::Token { .. }),
            observer,
        )?;
        if !saw_token {
            return Err(core_llm::Error::InvalidRequest(
                "dense campaign produced no first-token observation".into(),
            ));
        }
        provider.campaign_prefix_reuse(prefix_prompt)?;
        observer.phase("prompt-cache-reuse");
        provider.campaign_cancel_after_first_token(request)?;
        observer.phase("cancellation-cleanup");
        output
    };
    observer.phase("post-run-release");
    Ok(output)
}

/// Product-owned result of one independently executed frozen quality fixture.  The artifact name
/// is selected by the worker, but every numeric value and emitted tool/token fact comes from the
/// loaded provider through the observer and `TextLlmOutput`.
pub struct ProductFixtureResult {
    pub observation: ProductObservations,
    pub output: TextLlmOutput,
}

pub fn run_product_fixture(
    snapshot: impl AsRef<Path>,
    prefix_prompt: &str,
    request: TextLlmRequest,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureResult> {
    let mut observer = ProductObserver::new();
    let output = run_dense_lifecycle_request(
        snapshot,
        prefix_prompt,
        request,
        Some(coordinate),
        &mut observer,
    )?;
    let observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    Ok(ProductFixtureResult {
        observation,
        output,
    })
}

fn negative_log_likelihood(probabilities: &[(i32, f64)]) -> Result<f64, String> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|(_, probability)| !probability.is_finite() || *probability <= 0.0)
    {
        return Err("missing finite product token probabilities".into());
    }
    Ok(-probabilities
        .iter()
        .map(|(_, probability)| probability.ln())
        .sum::<f64>()
        / probabilities.len() as f64)
}

fn token_agreement(left: &[(i32, f64)], right: &[(i32, f64)]) -> (u64, u64) {
    let total = left.len().min(right.len()) as u64;
    let matches = left
        .iter()
        .zip(right)
        .filter(|((left, _), (right, _))| left == right)
        .count() as u64;
    (matches, total)
}

/// Convert four actual candidate/reference fixture pairs into the raw quality input consumed by
/// [`ReceiptBuilder`].  This helper intentionally rejects a fixture that did not execute its own
/// required behavior (tool parsing, needle recovery, or prefix result equality), rather than
/// converting a missing capability into a green ratio.
pub fn quality_from_product_fixtures(
    kernel_candidate: &ProductFixtureResult,
    kernel_reference: &ProductFixtureResult,
    tool_candidate: &ProductFixtureResult,
    tool_reference: &ProductFixtureResult,
    needle_candidate: &ProductFixtureResult,
    needle_reference: &ProductFixtureResult,
    cache_candidate: &ProductFixtureResult,
    cache_reference: &ProductFixtureResult,
    expected_needle: &str,
) -> Result<QualityObservation, String> {
    let candidate_logits = &kernel_candidate.observation.prefill_logits;
    let reference_logits = &kernel_reference.observation.prefill_logits;
    if candidate_logits.len() != reference_logits.len() || candidate_logits.is_empty() {
        return Err("candidate/reference production logits are not comparable".into());
    }
    let parity_errors = candidate_logits
        .iter()
        .zip(reference_logits)
        .map(|(candidate, reference)| f64::from((candidate - reference).abs()))
        .collect::<Vec<_>>();
    let (greedy_matches, greedy_total) = token_agreement(
        &kernel_candidate.observation.token_probabilities,
        &kernel_reference.observation.token_probabilities,
    );
    let (cache_matches, cache_total) = token_agreement(
        &cache_candidate.observation.token_probabilities,
        &cache_reference.observation.token_probabilities,
    );
    let tool_ok = !tool_candidate.output.tool_calls.is_empty()
        && tool_candidate.output.tool_calls == tool_reference.output.tool_calls;
    let needle_ok = needle_candidate.output.text.contains(expected_needle)
        && needle_reference.output.text.contains(expected_needle);
    if !tool_ok {
        return Err("structured-tool fixture produced no matching product tool call".into());
    }
    if !needle_ok {
        return Err("long-context needle fixture did not recover the product-owned needle".into());
    }
    Ok(QualityObservation {
        parity_errors,
        reference_perplexity: negative_log_likelihood(
            &kernel_reference.observation.token_probabilities,
        )?,
        candidate_perplexity: negative_log_likelihood(
            &kernel_candidate.observation.token_probabilities,
        )?,
        greedy_matches,
        greedy_total,
        tool_matches: 1,
        tool_total: 1,
        needle_matches: 1,
        needle_total: 1,
        cache_matches,
        cache_total,
    })
}

/// The fixed tool offer used by the independent structured-output fixture.  It lives beside the
/// product runner so a receipt cannot claim structured agreement for an unoffered or caller-made
/// schema.
pub fn structured_fixture_tool() -> ToolSpec {
    ToolSpec::new(
        "record_baseline_fact",
        "Record the exact requested baseline fact.",
        serde_json::json!({
            "type": "object",
            "properties": { "fact": { "type": "string" } },
            "required": ["fact"],
            "additionalProperties": false,
        }),
    )
}

/// All four frozen fixture pairs, run independently against the candidate and immutable fp32
/// reference snapshots.  The prompt construction is fixed in source; the worker can supply only
/// the audited model paths and the base scenario text.
pub struct ProductFixtureSuite {
    pub kernel_candidate: ProductFixtureResult,
    pub kernel_reference: ProductFixtureResult,
    pub tool_candidate: ProductFixtureResult,
    pub tool_reference: ProductFixtureResult,
    pub needle_candidate: ProductFixtureResult,
    pub needle_reference: ProductFixtureResult,
    pub cache_candidate: ProductFixtureResult,
    pub cache_reference: ProductFixtureResult,
    pub needle: String,
}

fn fixture_request(prompt: String, tools: Vec<ToolSpec>) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::text(Role::User, prompt)],
        tools,
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        },
        max_new_tokens: 64,
        seed: Some(0),
        ..Default::default()
    }
}

pub fn run_product_fixture_suite(
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureSuite> {
    let needle = "SC20671-NUMERIC-NEEDLE-9b7a2e".to_string();
    let kernel_prompt = format!("{prompt}\nReturn a concise deterministic answer.");
    let tool_prompt = format!(
        "{prompt}\nCall record_baseline_fact with fact exactly `SC20671 structured fixture`."
    );
    let needle_prompt = format!(
        "{prompt}\n{}\nThe required answer is the exact marker above: {needle}.",
        "context ".repeat(2_048)
    );
    let cache_prompt = format!("{prompt}\nRepeat the stable baseline fact.");
    let run_pair = |prefix: &str, request: TextLlmRequest| -> core_llm::Result<_> {
        Ok((
            run_product_fixture(candidate_snapshot, prefix, request.clone(), coordinate)?,
            run_product_fixture(reference_snapshot, prefix, request, coordinate)?,
        ))
    };
    let (kernel_candidate, kernel_reference) = run_pair(
        &kernel_prompt,
        fixture_request(kernel_prompt.clone(), Vec::new()),
    )?;
    let (tool_candidate, tool_reference) = run_pair(
        &tool_prompt,
        fixture_request(tool_prompt.clone(), vec![structured_fixture_tool()]),
    )?;
    let (needle_candidate, needle_reference) = run_pair(
        &needle_prompt,
        fixture_request(needle_prompt.clone(), Vec::new()),
    )?;
    let (cache_candidate, cache_reference) = run_pair(
        &cache_prompt,
        fixture_request(cache_prompt.clone(), Vec::new()),
    )?;
    Ok(ProductFixtureSuite {
        kernel_candidate,
        kernel_reference,
        tool_candidate,
        tool_reference,
        needle_candidate,
        needle_reference,
        cache_candidate,
        cache_reference,
        needle,
    })
}

impl ProductFixtureSuite {
    pub fn quality(&self) -> Result<QualityObservation, String> {
        quality_from_product_fixtures(
            &self.kernel_candidate,
            &self.kernel_reference,
            &self.tool_candidate,
            &self.tool_reference,
            &self.needle_candidate,
            &self.needle_reference,
            &self.cache_candidate,
            &self.cache_reference,
            &self.needle,
        )
    }
}

/// Derive one timing sample exclusively from product phase boundaries.  The first load interval is
/// retained as the cold compile-inclusive observation; subsequent samples in the same warm worker
/// retain their own load interval as warm compile-inclusive observations.  We do not pretend this
/// separates MLX compilation from model loading: the receipt labels the two measured process modes
/// and preserves every raw sample for later analysis.
pub fn timing_from_product_observation(
    observation: &ProductObservations,
    generated_tokens: usize,
    process_temperature: &str,
) -> Result<RawTiming, String> {
    if observation.phase_elapsed_ms.len() != REQUIRED_PHASES.len() || generated_tokens == 0 {
        return Err("timing requires eight product phases and generated tokens".into());
    }
    let elapsed = &observation.phase_elapsed_ms;
    if elapsed.windows(2).any(|window| window[1] <= window[0]) {
        return Err("product phase timing is not strictly increasing".into());
    }
    let positive_delta = |later: usize, earlier: usize| {
        let value = elapsed[later] - elapsed[earlier];
        (value.is_finite() && value > 0.0)
            .then_some(value)
            .ok_or_else(|| "nonpositive product phase duration".to_string())
    };
    let load_ms = positive_delta(1, 0)?;
    let prefill_ms = positive_delta(2, 1)?;
    let ttft_ms = positive_delta(3, 2)?;
    let first_token_ms = positive_delta(3, 0)?;
    let decode_ms = positive_delta(4, 3)?;
    let throughput = generated_tokens as f64 * 1_000.0 / decode_ms;
    if !throughput.is_finite() || throughput <= 0.0 {
        return Err("invalid product decode throughput".into());
    }
    let (cold_compile_ms, warm_compile_ms) = match process_temperature {
        "cold" => (load_ms, 0.000_001),
        "warm" => (0.000_001, load_ms),
        _ => return Err("unknown process temperature".into()),
    };
    Ok(RawTiming {
        load_ms,
        prefill_ms,
        ttft_ms,
        first_token_ms,
        decode_tokens_per_second: throughput,
        cold_compile_ms,
        warm_compile_ms,
    })
}

/// Freeze the cold-versus-steady timing attribution before receipt assembly.  The first real
/// product run in a fresh child is the cold sample; four subsequent runs in that same child are the
/// warm samples.  The observed excess of the cold load interval over the median warm interval is
/// the JIT/first-dispatch component.  A nonpositive excess fails closed rather than being rounded
/// into a made-up compile measurement.
pub fn timing_samples_from_product_repeats(
    runs: &[&ProductFixtureResult],
    process_temperature: &str,
) -> Result<Vec<RawTiming>, String> {
    if runs.len() != 5 {
        return Err("receipt requires exactly five product repeats".into());
    }
    let mut samples = runs
        .iter()
        .map(|run| {
            timing_from_product_observation(
                &run.observation,
                run.output.usage.generated_tokens as usize,
                process_temperature,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut warm_loads = samples[1..]
        .iter()
        .map(|sample| sample.load_ms)
        .collect::<Vec<_>>();
    warm_loads.sort_by(f64::total_cmp);
    let warm_load_ms = (warm_loads[1] + warm_loads[2]) / 2.0;
    let cold_jit_ms = samples[0].load_ms - warm_load_ms;
    if !warm_load_ms.is_finite()
        || warm_load_ms <= 0.0
        || !cold_jit_ms.is_finite()
        || cold_jit_ms <= 0.0
    {
        return Err(
            "fresh-process cold run did not expose a positive JIT/first-dispatch excess".into(),
        );
    }
    for sample in &mut samples {
        sample.cold_compile_ms = cold_jit_ms;
        sample.warm_compile_ms = warm_load_ms;
    }
    Ok(samples)
}

fn checked_git_revision(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["rev-parse", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;
    let revision = String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .trim()
        .to_string();
    if !output.status.success()
        || revision.len() != 40
        || !revision.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Err("cannot resolve immutable producer revision".into());
    }
    Ok(revision)
}

fn required_campaign_env(name: &str) -> Result<String, String> {
    let value =
        std::env::var(name).map_err(|_| format!("missing required campaign environment {name}"))?;
    if value.trim().is_empty() || value.contains("product-probed-at-worker") {
        return Err(format!(
            "campaign environment {name} is not a real probed value"
        ));
    }
    Ok(value)
}

fn probed_command(program: &str, args: &[&str], label: &str) -> Result<String, String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|e| format!("probe {label}: {e}"))?;
    if !output.status.success() {
        return Err(format!("probe {label} failed"));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|e| format!("probe {label} output: {e}"))?
        .trim()
        .to_string();
    if value.is_empty() || value.contains("product-probed-at-worker") {
        return Err(format!("probe {label} returned no real value"));
    }
    Ok(value)
}

/// `pmset -g therm` is verbose diagnostic text, never a receipt state.  Accept only an explicit
/// no-throttling/no-pressure observation and normalize it to the schema's semantic `nominal`.
fn normalize_pmset_thermal(value: &str) -> Result<String, String> {
    let normalized = value.to_ascii_lowercase();
    if normalized.contains("thermal pressure: 0")
        || normalized.contains("thermal level: 0")
        || normalized.contains("nominal")
    {
        return Ok("nominal".into());
    }
    Err("pmset thermal probe did not prove nominal thermal state".into())
}

fn product_receipt(
    coordinate: &Coordinate,
    suite: &ProductFixtureSuite,
    timings: Vec<RawTiming>,
    executable: &Path,
    fixtures: &[SealedFixtureArtifact],
    warmup_suite_sha256: Option<String>,
) -> Result<Receipt, String> {
    let quality = suite.quality()?;
    let observation = &suite.kernel_candidate.observation;
    let reference = &suite.kernel_reference.observation.snapshot;
    let model = &observation.snapshot;
    let inference_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let _ = fs::metadata(executable).map_err(|e| e.to_string())?;
    let scene_works_root = PathBuf::from(required_campaign_env("SCENEWORKS_ROOT")?);
    let scene_works_revision = checked_git_revision(&scene_works_root)?;
    let inference_revision = checked_git_revision(inference_root)?;
    let hardware = probed_command("sysctl", &["-n", "hw.model"], "hardware")?;
    let xcode = probed_command("xcodebuild", &["-version"], "xcode")?;
    let power_mode = probed_command("pmset", &["-g", "custom"], "power mode")?;
    let thermal_state = probed_command("pmset", &["-g", "therm"], "thermal state")?;
    let normalized_thermal_state = normalize_pmset_thermal(&thermal_state)?;
    let transcript = format!(
        "{}\n{}\n{}\n{}",
        suite.kernel_candidate.output.text,
        suite.tool_candidate.output.text,
        suite.needle_candidate.output.text,
        suite.cache_candidate.output.text
    );
    let mut fallback_reasons = std::collections::BTreeMap::new();
    for capability in [
        "trim",
        "rollback",
        "clear",
        "clone",
        "batchSplit",
        "batchMerge",
        "prefixCopyOnWrite",
        "pageImport",
        "pageExport",
        "serialization",
        "restore",
        "denseFallback",
    ] {
        fallback_reasons.insert(
            format!("{capability}FallbackReason"),
            "dense baseline route has no compressed lifecycle representation".into(),
        );
    }
    let cache_bytes = observation
        .allocations
        .iter()
        .filter(|e| e.role == "cache" && e.lifetime == "persistent")
        .map(|e| e.bytes)
        .sum::<u64>();
    let workspace = observation
        .allocations
        .iter()
        .filter(|e| e.lifetime == "transient")
        .map(|e| e.bytes)
        .sum::<u64>();
    let release = observation.phases.last().ok_or("release phase")?;
    let start = observation.phases.first().ok_or("start phase")?;
    let mut fixture_evidence = std::collections::BTreeMap::new();
    for name in REQUIRED_FIXTURES {
        let artifact_name = format!("fixtures/{name}.json");
        let artifact = fixtures
            .iter()
            .find(|artifact| artifact.name == artifact_name)
            .ok_or_else(|| format!("missing sealed fixture artifact {name}"))?;
        let hash = seal_bytes(&artifact.bytes);
        if artifact.sidecar != format!("{hash}  {}\n", artifact.name) {
            return Err(format!("fixture sidecar does not bind {name}"));
        }
        fixture_evidence.insert(
            name.into(),
            ReceiptFixture {
                passed: true,
                artifact_name,
                artifact_sha256: hash,
                artifact_sidecar_sha256: seal_bytes(artifact.sidecar.as_bytes()),
                independent_reference: format!("fp32-snapshot:{}", reference.sha256),
            },
        );
    }
    let template = Receipt {
        schema_version: 3, harness_version: "sc-20671-kv-baseline-v3".into(), run_id: seal_bytes(format!("{}:{}:{}", coordinate_slug(coordinate), model.sha256, seal_bytes(transcript.as_bytes())).as_bytes()), captured_at: release.timestamp.clone(), mode: "dense".into(), status: "complete".into(), contract_hash: QUALITY_CONTRACT_HASH.into(), receipt_sha256: String::new(),
        provenance: ReceiptProvenance { scene_works_revision, inference_revision, mlx_revision: format!("pmetal-lock:{}", seal_bytes(include_bytes!("../../../../Cargo.lock"))), dependency_lock_sha256: seal_bytes(include_bytes!("../../../../Cargo.lock")), os: std::env::consts::OS.into(), xcode, hardware, model_id: format!("{};tokenizer={};reference={}", model.root.display(), model.sha256, reference.sha256), model_file_sha256: model.sha256.clone(), model_file_bytes: model.bytes, power_mode, thermal_state: normalized_thermal_state, command_template: "sc20671-kv-baseline --mode {mode}".into(), command: "sc20671-kv-baseline --mode dense".into() },
        matrix: ReceiptMatrix { family: coordinate.family.into(), context_band: coordinate.context_band.into(), request_mode: coordinate.request_mode.into(), prefill_mode: coordinate.prefill_mode.into(), process_temperature: coordinate.process_temperature.into() },
        geometry: ReceiptGeometry { batch: if coordinate.request_mode == "single" {1} else {2}, query_heads: observation.geometry.query_heads, kv_heads: observation.geometry.kv_heads, head_dimension: observation.geometry.head_dimension, query_length: suite.kernel_candidate.output.usage.prompt_tokens as u64, kv_length: suite.kernel_candidate.output.usage.prompt_tokens as u64, layers: observation.geometry.layers, element_bytes: observation.geometry.element_bytes, capacity: suite.kernel_candidate.output.usage.prompt_tokens as u64 },
        memory: ReceiptMemory { model_weights_bytes: model.bytes, persistent_kv_bytes: cache_bytes, transient_workspace_bytes: workspace, dense_theoretical_kv_bytes: 0, phase_samples: vec![], allocation_events: vec![], reconciliation: ReceiptReconciliation { expected_dense_kv_bytes: 0, observed_persistent_kv_bytes: 0, tolerance_bytes: 0 }, release: ReceiptRelease { verified: release.phys_footprint_bytes <= start.phys_footprint_bytes && release.mlx.active_bytes <= start.mlx.active_bytes, phys_footprint_tolerance_bytes: 0, mlx_active_tolerance_bytes: 0, mlx_cache_tolerance_bytes: 0 } },
        timings: ReceiptTimings { load_ms: 0.0,prefill_ms:0.0,ttft_ms:0.0,first_token_ms:0.0,decode_tokens_per_second:0.0,cold_compile_ms:0.0,warm_compile_ms:0.0,samples:vec![],summary:ReceiptTimingSummary{decode_tokens_per_second_mean:0.0,decode_tokens_per_second_p95:0.0,decode_tokens_per_second_variance:0.0,decode_tokens_per_second_coefficient_of_variation:0.0,confidence_interval_low:0.0,confidence_interval_high:0.0}},
        quality: ReceiptQuality { parity_max_error:0.0,perplexity_delta:0.0,greedy_token_agreement:0.0,structured_tool_agreement:0.0,needle_retrieval:0.0,multi_turn_prompt_cache:0.0,statistics:ReceiptQualityStatistics{repeats:5,warmups:0,confidence_interval:"95% bootstrap".into(),outlier_policy:"report all samples; no silent deletion".into(),variance_policy:"all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum".into(),max_coefficient_of_variation:0.05},fixture_evidence}, lifecycle: ReceiptLifecycle { append:true,chunked_prefill:coordinate.prefill_mode=="chunked",single_shot_prefill:coordinate.prefill_mode=="single-shot",prompt_cache_reuse:true,trim:false,rollback:false,clear:false,cancel:true,clone:false,batch_split:false,batch_merge:false,prefix_copy_on_write:false,page_import:false,page_export:false,serialization:false,restore:false,dense_fallback:false,post_run_release:true,fallback_reasons }, cancellation: ReceiptCancellation{cleanup_verified:true}, warmup: ReceiptWarmup { required: coordinate.process_temperature == "warm", completed: warmup_suite_sha256.is_some(), worker_pid: std::process::id(), suite_sha256: warmup_suite_sha256.unwrap_or_default() } };
    ReceiptBuilder {
        template,
        phases: observation.phases.clone(),
        allocations: observation.allocations.clone(),
        timings,
        quality,
    }
    .finish()
}

/// Execute the coordinate-specific production control before receipt assembly.  A supported batch
/// is a real `generate_batch` dispatch (never serial aliases); chunked prefill is exercised through
/// the provider's real prefix cache path, which performs prefix/suffix prefill with an existing KV
/// cache and validates a hit.  Unsupported decoder contracts return typed errors before any receipt
/// is created.
pub fn verify_coordinate_product_controls(
    snapshot: &Path,
    coordinate: &Coordinate,
    prompt: &str,
) -> core_llm::Result<()> {
    let provider = crate::provider::LlamaProvider::load(&core_llm::LoadSpec::dense(
        snapshot.to_string_lossy().to_string(),
    ))?;
    if coordinate.request_mode == "supported-batch" {
        provider.campaign_supported_batch(prompt, 2)?;
    }
    if coordinate.prefill_mode == "chunked" {
        provider.campaign_prefix_reuse(prompt)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn formula_includes_batch_and_key_value_pair() {
        assert_eq!(dense_kv_bytes(2, 3, 4, 5, 6, 2), Ok(2880));
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
    fn canonical_json_is_the_paired_sceneworks_representation() {
        let value = serde_json::json!({ "z": 1, "a": { "y": 2, "x": 3 } });
        assert_eq!(
            String::from_utf8(canonical_json_bytes(&value).unwrap()).unwrap(),
            "{\n  \"a\": {\n    \"x\": 3,\n    \"y\": 2\n  },\n  \"z\": 1\n}"
        );
    }

    #[test]
    fn producer_timestamp_is_rfc3339_and_strictly_monotonic() {
        let first = timestamp_now();
        let second = timestamp_now();
        assert!(first.ends_with('Z'));
        assert!(second.ends_with('Z'));
        assert!(first < second);
    }

    #[test]
    fn product_observer_refuses_caller_supplied_partial_evidence() {
        let observer = ProductObserver::new();
        assert!(observer.finish().is_err());
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
    fn schedule_is_exact_and_rejects_duplicate_or_reused_cold_workers() {
        let schedule = required_schedule();
        assert_eq!(schedule.len(), 64);
        assert_eq!(
            schedule
                .iter()
                .filter(|row| row.discipline == ProcessDiscipline::FreshChild)
                .count(),
            32
        );
        let outcomes = schedule
            .iter()
            .enumerate()
            .map(|(index, row)| (row.coordinate.clone(), row.discipline, (index + 1) as u32))
            .collect::<Vec<_>>();
        assert!(validate_schedule_outcomes(&schedule, &outcomes).is_ok());
        let mut duplicate = outcomes.clone();
        duplicate[1].0 = duplicate[0].0.clone();
        assert!(validate_schedule_outcomes(&schedule, &duplicate).is_err());
        let first_cold = outcomes
            .iter()
            .position(|row| row.1 == ProcessDiscipline::FreshChild)
            .unwrap();
        let second_cold = outcomes
            .iter()
            .enumerate()
            .find(|(index, row)| *index != first_cold && row.1 == ProcessDiscipline::FreshChild)
            .map(|(index, _)| index)
            .unwrap();
        let mut reused_cold = outcomes;
        reused_cold[second_cold].2 = reused_cold[first_cold].2;
        assert!(validate_schedule_outcomes(&schedule, &reused_cold).is_err());
    }

    #[test]
    fn private_worker_seam_schedules_every_coordinate_once_with_declared_isolation() {
        let mut seen = Vec::new();
        let outcomes = execute_required_schedule(|row| {
            seen.push((row.coordinate.clone(), row.discipline));
            Ok((seen.len() + 100) as u32)
        })
        .unwrap();
        assert_eq!(seen.len(), 64);
        assert_eq!(outcomes.len(), 64);
        assert_eq!(
            seen.iter()
                .filter(|(_, discipline)| *discipline == ProcessDiscipline::ReusedWarmWorker)
                .count(),
            32
        );
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

    struct FakeSampler(u8);
    impl CampaignSampler for FakeSampler {
        fn sample(&mut self, _phase: &'static str) -> Result<MemorySample, String> {
            let second = self.0;
            self.0 += 1;
            Ok(MemorySample {
                captured_at: format!("2026-01-01T00:00:{second:02}.000Z"),
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
        let phases = run_lifecycle(FakeSampler(0), |recorder| {
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
        assert!(run_lifecycle(FakeSampler(0), |_recorder| Ok(())).is_err());
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
        let mut template = Receipt {
            schema_version: 3,
            harness_version: "sc-20671-kv-baseline-v3".into(),
            run_id: "run".into(),
            captured_at: "2026-01-01T00:00:00Z".into(),
            mode: "dense".into(),
            status: "complete".into(),
            contract_hash: QUALITY_CONTRACT_HASH.into(),
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
                transient_workspace_bytes: 1,
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
                    variance_policy: "all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum".into(),
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
            warmup: ReceiptWarmup {
                required: false,
                completed: false,
                worker_pid: 7,
                suite_sha256: String::new(),
            },
        };
        for name in REQUIRED_FIXTURES {
            template.quality.fixture_evidence.insert(
                (*name).into(),
                ReceiptFixture {
                    passed: true,
                    artifact_name: format!("fixtures/{name}.json"),
                    artifact_sha256: "a".repeat(64),
                    artifact_sidecar_sha256: "b".repeat(64),
                    independent_reference: "independent-reference".into(),
                },
            );
        }
        let phases = REQUIRED_PHASES
            .iter()
            .enumerate()
            .map(|(index, phase)| ReceiptPhase {
                phase: (*phase).into(),
                pid: 7,
                source: "footprint -p".into(),
                timestamp: format!("2026-01-01T00:00:0{index}.000Z"),
                phys_footprint_bytes: 100,
                phys_footprint_peak_bytes: 100,
                mlx: ReceiptMlx {
                    source: "mlx_rs::memory".into(),
                    active_bytes: 10,
                    cache_bytes: 1,
                    peak_bytes: 10,
                },
            })
            .collect();
        let allocations = vec![
            ReceiptAllocation {
                kind: "weights-allocation".into(),
                role: "weights".into(),
                lifetime: "persistent".into(),
                phase: "weights-loaded".into(),
                timestamp: "2026-01-01T00:00:01.100Z".into(),
                bytes: 1,
            },
            ReceiptAllocation {
                kind: "dense-kv-allocation".into(),
                role: "cache".into(),
                lifetime: "persistent".into(),
                phase: "prefill-peak".into(),
                timestamp: "2026-01-01T00:00:02.100Z".into(),
                bytes: 4,
            },
            ReceiptAllocation {
                kind: "attention-workspace".into(),
                role: "attention-workspace".into(),
                lifetime: "transient".into(),
                phase: "prefill-peak".into(),
                timestamp: "2026-01-01T00:00:02.200Z".into(),
                bytes: 1,
            },
        ];
        let timings = (0..5)
            .map(|_| RawTiming {
                load_ms: 1.0,
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 1.0,
                decode_tokens_per_second: 1.0,
                cold_compile_ms: 1.0,
                warm_compile_ms: 1.0,
            })
            .collect();
        let quality = QualityObservation {
            parity_errors: vec![0.0],
            reference_perplexity: 1.0,
            candidate_perplexity: 1.0,
            greedy_matches: 1,
            greedy_total: 1,
            tool_matches: 1,
            tool_total: 1,
            needle_matches: 1,
            needle_total: 1,
            cache_matches: 1,
            cache_total: 1,
        };
        let receipt = ReceiptBuilder {
            template,
            phases,
            allocations,
            timings,
            quality,
        }
        .finish()
        .expect("builder must produce a complete v3 receipt");
        let mut tampered = receipt.clone();
        tampered.geometry.capacity = 2;
        assert!(validate_receipt_semantics(&tampered).is_err());
        let mut timing_tampered = receipt.clone();
        timing_tampered.timings.decode_tokens_per_second += 1.0;
        assert!(validate_receipt_semantics(&timing_tampered).is_err());
        let mut lifecycle_tampered = receipt.clone();
        lifecycle_tampered.lifecycle.append = false;
        assert!(validate_receipt_semantics(&lifecycle_tampered).is_err());
        let mut phase_tampered = receipt.clone();
        phase_tampered.memory.phase_samples[3].source = "caller-authored".into();
        assert!(validate_receipt_semantics(&phase_tampered).is_err());
        let mut calendar_tampered = receipt.clone();
        calendar_tampered.memory.phase_samples[0].timestamp = "2026-02-31T00:00:00Z".into();
        assert!(validate_receipt_semantics(&calendar_tampered).is_err());
        let mut revision_tampered = receipt.clone();
        revision_tampered.provenance.scene_works_revision = "A".repeat(40);
        assert!(validate_receipt_semantics(&revision_tampered).is_err());
        let mut command_tampered = receipt.clone();
        command_tampered.provenance.command_template = "run --dense".into();
        command_tampered.provenance.command = "run --dense".into();
        assert!(validate_receipt_semantics(&command_tampered).is_err());
        let mut threshold_tampered = receipt.clone();
        threshold_tampered
            .memory
            .allocation_events
            .push(ReceiptAllocation {
                kind: "temporary-full-cache".into(),
                role: "cache".into(),
                lifetime: "transient".into(),
                phase: "prefill-peak".into(),
                timestamp: "2026-01-01T00:00:02.300Z".into(),
                bytes: 4,
            });
        assert!(validate_receipt_semantics(&threshold_tampered).is_err());
        let mut split_prefill = receipt.clone();
        split_prefill.memory.phase_samples[2].mlx.active_bytes = 3;
        split_prefill.memory.phase_samples[2].mlx.peak_bytes = 4;
        assert!(3 + 4 > 1 + 4 + 1);
        assert!(validate_receipt_semantics(&split_prefill).is_err());
        let mut fixture_tampered = receipt.clone();
        fixture_tampered
            .quality
            .fixture_evidence
            .remove(REQUIRED_FIXTURES[0]);
        assert!(validate_receipt_semantics(&fixture_tampered).is_err());
        let mut partial = receipt.clone();
        partial.timings.samples.pop();
        assert!(validate_receipt_semantics(&partial).is_err());
        let bundle = assemble_artifacts(receipt).expect("receipt bytes must assemble");
        assert!(
            validate_artifact_bundle(&bundle).is_err(),
            "campaign receipts require published fixture bytes"
        );
        let named = assemble_artifacts_named(
            serde_json::from_slice(&bundle.receipt).unwrap(),
            "baseline.json",
            "baseline.txt",
        )
        .unwrap();
        assert!(validate_artifact_bundle_named(&named, "baseline.json", "baseline.txt").is_err());
        let mut sealed_tampered = bundle.clone();
        sealed_tampered.receipt[0] ^= 1;
        assert!(validate_artifact_bundle(&sealed_tampered).is_err());
    }

    #[test]
    fn atomic_writer_publishes_named_bytes_and_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let receipt = b"receipt".to_vec();
        let human = b"human receipt".to_vec();
        let bundle = ArtifactBundle {
            receipt_name: "run.json".into(),
            human_name: "run.txt".into(),
            receipt_sidecar: format!("{}  run.json\n", seal_bytes(&receipt)),
            human_sidecar: format!("{}  run.txt\n", seal_bytes(&human)),
            receipt: receipt.clone(),
            human: human.clone(),
            fixtures: Vec::new(),
        };
        let output = dir.path().join("run");
        write_artifacts(&output, "run.json", "run.txt", &bundle).unwrap();
        assert_eq!(fs::read(output.join("run.json")).unwrap(), receipt);
        assert_eq!(
            fs::read_to_string(output.join("run.json.sha256")).unwrap(),
            bundle.receipt_sidecar
        );
        assert!(output.join("run.txt.sha256").is_file());
        assert!(write_artifacts(&output, "run.json", "run.txt", &bundle).is_err());
    }
}

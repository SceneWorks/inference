//! Sealed real-model evidence harness for SC-20676 packed Metal decode attention.
//!
//! The dense SC-20671 campaign remains its own frozen covering-set baseline. This module
//! consumes a completed baseline receipt and records the much narrower product comparison needed
//! for the opt-in packed reader: one Llama and one Qwen3 snapshot, each in fresh dense and packed
//! workers.  It deliberately never accepts caller-authored cache counters or provenance.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use core_llm::Tokenizer;
use serde::{Deserialize, Serialize};

use crate::campaign;
use crate::campaign_supervisor::{self, RunRequest, SystemProbe};
use crate::config::Architecture;
use crate::decode::{
    generate_from_prefill, generate_with_cache, CancelFlag, FinishReason, GenerationConfig,
    StreamEvent,
};
use crate::models::CausalLm;
use crate::primitives::nn::to_f32_host;
use crate::primitives::{
    input_ids, CacheRoute, CompiledKernelHandle, KvCache, PackedCacheEvidence, PackedMetalKernel,
    Weights,
};
use crate::{Error, ModelConfig, Result};

pub const SC20676_SCHEMA_VERSION: u32 = 3;
pub const SC20676_HARNESS_VERSION: &str = "sc-20676-packed-metal-evidence-v3";
/// The SC-20671 frozen contract is the policy identity; SC-20676 only narrows it with the
/// compressed-domain parity requirements below and never introduces a tunable caller threshold.
pub const SC20676_CONTRACT_HASH: &str = campaign::QUALITY_CONTRACT_HASH;
pub const SC20676_MAX_LOGIT_ABS_ERROR: f64 = campaign::COMPRESSED_PARITY_MAX_ERROR;
pub const SC20676_MIN_GREEDY_TOKEN_AGREEMENT: f64 = campaign::COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN;
pub const SC20676_MIN_LONG_CONTEXT_TOKENS: usize = 1_024;
pub const SC20676_NEEDLE: &str = "SC20676-NEEDLE-9b7a2e";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Thresholds {
    pub contract_hash: String,
    pub max_logit_abs_error: f64,
    pub min_greedy_token_agreement: f64,
}

impl Default for Sc20676Thresholds {
    fn default() -> Self {
        Self {
            contract_hash: SC20676_CONTRACT_HASH.into(),
            max_logit_abs_error: SC20676_MAX_LOGIT_ABS_ERROR,
            min_greedy_token_agreement: SC20676_MIN_GREEDY_TOKEN_AGREEMENT,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676BaselineBinding {
    pub receipt_sha256: String,
    pub campaign_manifest_sha256: String,
    pub campaign_schedule_version: u64,
    pub model_file_sha256: String,
    pub model_id: String,
    pub model_repository: String,
    pub model_revision: String,
    pub snapshot_inventory_sha256: String,
    pub scene_works_revision: String,
    pub inference_revision: String,
    pub campaign_session_id: String,
    pub campaign_global_identity_sha256: String,
    pub context_band: String,
    pub context_payload_tokens: u64,
    pub head_dimension: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676InputBinding {
    pub ids_sha256: String,
    pub ids_len: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676OutputBinding {
    pub logits_sha256: String,
    pub logits_len: u64,
    pub logits_shape: Vec<i32>,
    pub logits_dtype: String,
    pub tokens_sha256: String,
    pub token_len: u64,
    pub needle_expected: String,
    pub needle_output: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676TimingSample {
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub steady_decode_tokens_per_second: f64,
    pub packed_cold_dispatch_ms: f64,
    pub packed_warm_dispatch_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676TimingSummary {
    pub mean_tokens_per_second: f64,
    pub variance_tokens_per_second: f64,
    pub coefficient_of_variation: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676MemoryPhase {
    pub phase: String,
    pub captured_at: String,
    pub phys_footprint_bytes: u64,
    pub phys_footprint_peak_bytes: u64,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676PeakWindow {
    pub started_at: String,
    pub baseline_active_bytes: u64,
    pub baseline_cache_bytes: u64,
    pub reset_peak_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676MemoryEvidence {
    pub phases: Vec<Sc20676MemoryPhase>,
    pub decode_peak_window: Sc20676PeakWindow,
    pub dense_theoretical_kv_bytes: u64,
    pub observed_dense_kv_bytes: u64,
    pub packed_logical_payload_bytes: u64,
    pub packed_metadata_bytes: u64,
    pub packed_device_bytes: u64,
    pub transient_workspace_bytes: u64,
    /// Geometry-only comparison values; these are never physical allocations.
    /// Both dense and packed arms carry them; only live cache evidence establishes avoidance.
    pub theoretical_dense_reconstruction_bytes: u64,
    pub theoretical_score_matrix_bytes: u64,
    pub release_verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676KernelProfile {
    pub metal_device: String,
    pub gpu_family: String,
    pub qualification: String,
    pub head_dimension: u64,
    pub threads: u64,
    pub simd_groups: u64,
    pub values_per_thread: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Quality {
    pub max_logit_abs_error: f64,
    pub greedy_token_agreement: f64,
    /// Packed needle outcome relative to the same-weights dense run (quality contract v3).
    pub needle_retrieval: bool,
    /// False when the dense run itself missed the needle; the packed check then measures only
    /// exact agreement with the dense output and cannot detect KV-induced retrieval loss.
    pub needle_discriminating: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Fallback {
    pub route: String,
    pub reason: String,
    pub executed_dense_request: bool,
    pub packed_dispatch_attempts_before: u64,
    pub packed_dispatch_attempts_after: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Cancellation {
    pub finish_reason: String,
    pub emitted_tokens: u64,
    pub retained_before_reset: u64,
    pub retained_after_reset: u64,
    pub release_verified: bool,
    pub packed_before_reset: PackedCacheEvidence,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Provenance {
    pub scene_works_repository: String,
    pub scene_works_revision: String,
    pub inference_repository: String,
    pub inference_revision: String,
    pub dependency_lock_sha256: String,
    pub mlx_version: String,
    pub mlx_source: String,
    pub mlx_revision: String,
    pub os: String,
    pub os_version: String,
    pub xcode: String,
    pub hardware: String,
    pub metal_device: String,
    pub power_mode: String,
    pub thermal_state: String,
    pub source_tree_clean: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Arm {
    pub arm_sha256: String,
    pub run_nonce: String,
    pub worker_pid: u32,
    pub executable_sha256: String,
    pub family: String,
    pub mode: String,
    pub snapshot_inventory_sha256: String,
    pub prompt_tokens: u64,
    pub elapsed_ms: f64,
    pub peak_mlx_bytes: u64,
    pub peak_phys_footprint_bytes: u64,
    pub provenance: Sc20676Provenance,
    pub input: Sc20676InputBinding,
    pub output: Sc20676OutputBinding,
    pub timings: Vec<Sc20676TimingSample>,
    pub timing_summary: Sc20676TimingSummary,
    pub memory: Sc20676MemoryEvidence,
    pub route: String,
    pub kernel_profile: Option<Sc20676KernelProfile>,
    pub packed: Option<PackedCacheEvidence>,
    pub warm_packed: Option<PackedCacheEvidence>,
    pub continuation_dispatches: u64,
    pub quality: Option<Sc20676Quality>,
    pub fallback: Option<Sc20676Fallback>,
    pub cancellation: Option<Sc20676Cancellation>,
    /// Runtime guards the worker ran under, with the stated cap and static estimate.
    pub admission: campaign::ReceiptAdmission,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Sc20676Receipt {
    pub schema_version: u32,
    pub harness_version: String,
    pub receipt_sha256: String,
    pub baseline: Sc20676BaselineBinding,
    pub thresholds: Sc20676Thresholds,
    pub dense: Sc20676Arm,
    pub packed: Sc20676Arm,
}

impl Sc20676Receipt {
    pub fn bytes(&self) -> std::result::Result<Vec<u8>, serde_json::Error> {
        let mut bytes = campaign::canonical_json_bytes(&serde_json::to_value(self)?)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    pub fn finish(mut self) -> std::result::Result<Self, String> {
        self.schema_version = SC20676_SCHEMA_VERSION;
        self.harness_version = SC20676_HARNESS_VERSION.into();
        self.receipt_sha256.clear();
        validate_sc20676_receipt_core(&self)?;
        self.receipt_sha256 = campaign::seal_bytes(&self.bytes().map_err(|e| e.to_string())?);
        validate_sc20676_receipt(&self)?;
        Ok(self)
    }
}

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn is_revision(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn sha256_bytes(bytes: &[u8]) -> String {
    campaign::seal_bytes(bytes)
}

fn arm_semantic_seal(arm: &Sc20676Arm) -> std::result::Result<String, String> {
    let mut core = arm.clone();
    core.arm_sha256.clear();
    campaign::canonical_json_bytes(&serde_json::to_value(core).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
        .map(|bytes| sha256_bytes(&bytes))
}

fn finish_arm(mut arm: Sc20676Arm) -> std::result::Result<Sc20676Arm, String> {
    arm.arm_sha256.clear();
    arm.arm_sha256 = arm_semantic_seal(&arm)?;
    Ok(arm)
}

/// The live worker invokes the pinned snapshot validator both before loading and after all model
/// work. Keeping the equality rule pure makes replacement races testable without model weights.
fn validate_snapshot_inventory_unchanged(
    initial: &campaign::SnapshotInventory,
    final_inventory: &campaign::SnapshotInventory,
) -> std::result::Result<(), String> {
    if initial.sha256 != final_inventory.sha256 || initial.bytes != final_inventory.bytes {
        return Err("SC-20676 candidate snapshot changed while worker was running".into());
    }
    Ok(())
}

fn validate_worker_binding(
    arm: &Sc20676Arm,
    family: &str,
    mode: &str,
    nonce: &str,
    executable_sha256: &str,
    provenance: &Sc20676Provenance,
    seen_pids: &mut std::collections::BTreeSet<u32>,
) -> std::result::Result<(), String> {
    validate_arm(arm)?;
    if arm.run_nonce != nonce
        || arm.executable_sha256 != executable_sha256
        || arm.family != family
        || arm.mode != mode
        || &arm.provenance != provenance
        || !seen_pids.insert(arm.worker_pid)
    {
        return Err("SC-20676 worker nonce/PID/executable/provenance binding failed".into());
    }
    Ok(())
}

/// Consume the exact arm values which were just sealed and parent-validated.  Keeping this as a
/// value-only operation prevents a second worker-file read from replacing the packed arm between
/// validation and receipt assembly.
fn take_validated_family_arms(
    arms: Vec<Sc20676Arm>,
) -> std::result::Result<(Sc20676Arm, Sc20676Arm), String> {
    let mut dense = None;
    let mut packed = None;
    for arm in arms {
        match arm.mode.as_str() {
            "dense" if dense.is_none() => dense = Some(arm),
            "packed" if packed.is_none() => packed = Some(arm),
            _ => {
                return Err(
                    "validated worker arms must contain one dense and one packed value".into(),
                )
            }
        }
    }
    Ok((
        dense.ok_or("missing dense worker arm")?,
        packed.ok_or("missing packed worker arm")?,
    ))
}

fn validate_complete_matrix_receipts(
    receipts: &[Sc20676Receipt],
) -> std::result::Result<(), String> {
    for receipt in receipts {
        validate_sc20676_receipt(receipt)?;
    }
    let families = receipts
        .iter()
        .map(|receipt| receipt.dense.family.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let baseline_campaigns = receipts
        .iter()
        .map(|receipt| {
            (
                receipt.baseline.campaign_manifest_sha256.as_str(),
                receipt.baseline.campaign_schedule_version,
                receipt.baseline.campaign_global_identity_sha256.as_str(),
                receipt.baseline.scene_works_revision.as_str(),
                receipt.baseline.inference_revision.as_str(),
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    if receipts.len() != 2 || families != std::collections::BTreeSet::from(["llama", "qwen"]) {
        return Err(
            "SC-20676 complete matrix requires exactly one Llama and one Qwen receipt".into(),
        );
    }
    if baseline_campaigns.len() != 1 {
        return Err(
            "SC-20676 complete matrix must bind one SC-20671 campaign and source closure".into(),
        );
    }
    Ok(())
}

fn timing_summary(
    samples: &[Sc20676TimingSample],
) -> std::result::Result<Sc20676TimingSummary, String> {
    if samples.len() != 5
        || samples
            .iter()
            .any(|sample| !finite_positive(sample.steady_decode_tokens_per_second))
    {
        return Err(
            "SC-20676 requires exactly five finite positive measured timing samples".into(),
        );
    }
    let mean = samples
        .iter()
        .map(|sample| sample.steady_decode_tokens_per_second)
        .sum::<f64>()
        / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample.steady_decode_tokens_per_second - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    Ok(Sc20676TimingSummary {
        mean_tokens_per_second: mean,
        variance_tokens_per_second: variance,
        coefficient_of_variation: variance.sqrt() / mean,
    })
}

fn validate_memory(memory: &Sc20676MemoryEvidence) -> std::result::Result<(), String> {
    const PHASES: [&str; 6] = [
        "process-start",
        "weights-loaded",
        "prefill-cache-resident",
        "decode-window-start",
        "decode-complete",
        "reset-release",
    ];
    if memory.phases.len() != PHASES.len()
        || memory.phases.iter().zip(PHASES).any(|(sample, expected)| {
            sample.phase != expected
                || !campaign::valid_utc_timestamp(&sample.captured_at)
                || sample.phys_footprint_peak_bytes < sample.phys_footprint_bytes
        })
        || memory.phases.windows(2).any(|window| {
            window[1].phys_footprint_peak_bytes < window[0].phys_footprint_peak_bytes
                || campaign::compare_utc_timestamps(&window[0].captured_at, &window[1].captured_at)
                    == Some(std::cmp::Ordering::Greater)
        })
        || memory.dense_theoretical_kv_bytes == 0
        || memory.observed_dense_kv_bytes == 0
        || memory.transient_workspace_bytes == 0
        || memory.theoretical_dense_reconstruction_bytes == 0
        || memory.theoretical_score_matrix_bytes == 0
        || !memory.release_verified
    {
        return Err("SC-20676 memory phase evidence is incomplete".into());
    }
    let prefill = &memory.phases[2];
    let window_start = &memory.phases[3];
    let decode = &memory.phases[4];
    if !campaign::valid_utc_timestamp(&memory.decode_peak_window.started_at)
        || memory.decode_peak_window.started_at != window_start.captured_at
        || !campaign::utc_timestamp_before(&prefill.captured_at, &window_start.captured_at)
        || !campaign::utc_timestamp_before(&window_start.captured_at, &decode.captured_at)
        || memory.decode_peak_window.baseline_active_bytes != window_start.mlx_active_bytes
        || memory.decode_peak_window.baseline_cache_bytes != window_start.mlx_cache_bytes
        || window_start.mlx_active_bytes != prefill.mlx_active_bytes
        || window_start.mlx_cache_bytes != prefill.mlx_cache_bytes
        || memory.decode_peak_window.reset_peak_bytes != 0
        || window_start.mlx_peak_bytes != memory.decode_peak_window.reset_peak_bytes
        || window_start.mlx_active_bytes == 0
    {
        return Err("SC-20676 decode peak window is not a sealed phase-local reset".into());
    }
    let weights = &memory.phases[1];
    let release = &memory.phases[5];
    if release.mlx_active_bytes != weights.mlx_active_bytes
        || release.mlx_cache_bytes != weights.mlx_cache_bytes
        || release.phys_footprint_bytes
            > weights
                .phys_footprint_bytes
                .saturating_add(campaign::POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES)
    {
        return Err("SC-20676 release did not return to the weights-only boundary".into());
    }
    if memory.transient_workspace_bytes
        != decode
            .mlx_peak_bytes
            .saturating_sub(decode.mlx_active_bytes)
    {
        return Err(
            "SC-20676 transient workspace is not the measured peak above resident cache".into(),
        );
    }
    if memory.theoretical_dense_reconstruction_bytes != memory.dense_theoretical_kv_bytes {
        return Err("SC-20676 dense comparison geometry does not match dense KV geometry".into());
    }
    Ok(())
}

fn validate_arm(arm: &Sc20676Arm) -> std::result::Result<(), String> {
    let phase_peak_mlx_bytes = arm
        .memory
        .phases
        .iter()
        .map(|phase| phase.mlx_peak_bytes)
        .max()
        .unwrap_or(0);
    let phase_peak_phys_footprint_bytes = arm
        .memory
        .phases
        .iter()
        .map(|phase| phase.phys_footprint_peak_bytes)
        .max()
        .unwrap_or(0);
    let shape_elements = arm
        .output
        .logits_shape
        .iter()
        .try_fold(1_u64, |total, dimension| {
            let dimension = u64::try_from(*dimension)
                .map_err(|_| "logits shape contains a negative dimension")?;
            total
                .checked_mul(dimension)
                .ok_or("logits shape overflows u64")
        })?;
    if !is_digest(&arm.arm_sha256)
        || arm.arm_sha256 != arm_semantic_seal(arm)?
        || !is_digest(&arm.run_nonce)
        || arm.worker_pid == 0
        || !is_digest(&arm.executable_sha256)
        || !["llama", "qwen"].contains(&arm.family.as_str())
        || !["dense", "packed"].contains(&arm.mode.as_str())
        || arm.input.ids_len == 0
        || !is_digest(&arm.input.ids_sha256)
        || !finite_positive(arm.elapsed_ms)
        || arm.peak_mlx_bytes != phase_peak_mlx_bytes
        || arm.peak_phys_footprint_bytes != phase_peak_phys_footprint_bytes
        || arm.output.logits_len == 0
        || arm.output.logits_shape.is_empty()
        || shape_elements != arm.output.logits_len
        || arm.output.logits_dtype.is_empty()
        || !is_digest(&arm.output.logits_sha256)
        || arm.output.token_len == 0
        || !is_digest(&arm.output.tokens_sha256)
        || arm.output.needle_expected != SC20676_NEEDLE
        || !arm.provenance.source_tree_clean
        || arm.provenance.scene_works_repository != campaign::SCENEWORKS_REPOSITORY
        || arm.provenance.inference_repository != campaign::INFERENCE_REPOSITORY
        || arm.provenance.inference_revision.len() != 40
        || arm.provenance.scene_works_revision.len() != 40
        || !is_digest(&arm.provenance.dependency_lock_sha256)
        || arm.provenance.mlx_version.is_empty()
        || arm.provenance.mlx_source.is_empty()
        || arm.provenance.mlx_revision.is_empty()
        || arm.provenance.os_version.is_empty()
        || arm.provenance.metal_device.is_empty()
        || arm.provenance.thermal_state != "nominal"
    {
        return Err("SC-20676 worker arm identity/input/output provenance is incomplete".into());
    }
    arm.admission.validate()?;
    let summary = timing_summary(&arm.timings)?;
    if !finite_positive(summary.mean_tokens_per_second)
        || summary.coefficient_of_variation > 0.05
        || arm.timing_summary != summary
        || arm.timings.iter().any(|sample| {
            !finite_positive(sample.prefill_ms)
                || !finite_positive(sample.ttft_ms)
                || !finite_positive(sample.first_token_ms)
                || (sample.first_token_ms - (sample.prefill_ms + sample.ttft_ms)).abs()
                    > f64::EPSILON * (sample.first_token_ms.abs() + 1.0)
                || (arm.mode == "dense"
                    && (sample.packed_cold_dispatch_ms != 0.0
                        || sample.packed_warm_dispatch_ms != 0.0))
                || (arm.mode == "packed"
                    && (sample.packed_cold_dispatch_ms != 0.0
                        || !finite_positive(sample.packed_warm_dispatch_ms)))
        })
    {
        return Err("SC-20676 timing evidence is incomplete or unstable".into());
    }
    validate_memory(&arm.memory)?;
    if arm.mode == "dense" {
        if arm.kernel_profile.is_some()
            || arm.packed.is_some()
            || arm.warm_packed.is_some()
            || arm.continuation_dispatches != 0
            || arm.quality.is_some()
            || arm.fallback.is_some()
            || arm.cancellation.is_some()
            || arm.memory.packed_logical_payload_bytes != 0
            || arm.memory.packed_metadata_bytes != 0
            || arm.memory.packed_device_bytes != 0
        {
            return Err("dense arm contains packed-only evidence".into());
        }
        return Ok(());
    }
    let packed = arm
        .packed
        .as_ref()
        .ok_or("packed arm lacks primary representation evidence")?;
    let profile = arm
        .kernel_profile
        .as_ref()
        .ok_or("packed arm lacks device-bound kernel profile evidence")?;
    let expected_values_per_thread = profile.head_dimension.checked_div(32).unwrap_or(0);
    if profile.metal_device != arm.provenance.metal_device
        || profile.gpu_family != "conservative-unknown-apple"
        || profile.qualification != "conservative-default"
        || ![64, 128, 256].contains(&profile.head_dimension)
        || profile.threads != 32
        || profile.simd_groups != 1
        || profile.values_per_thread != expected_values_per_thread
    {
        return Err("SC-20676 packed kernel profile is not bound to the measured device".into());
    }
    let warm = arm
        .warm_packed
        .as_ref()
        .ok_or("packed arm lacks warm representation evidence")?;
    if packed.representation_identity != "sc-20676-packed-group-affine-v1"
        || packed.representation_version != 2
        || packed.bits != 2
        || packed.quantization_group_size != 32
        || packed.accepted_direct_calls == 0
        || packed.full_cache_dequantizations != 0
        || packed.dense_active
        || packed.failed_dispatches != 0
        || packed.compile_jit_attempts != 1
        || !packed.kernel_warmed
        || packed.cold_dispatches != 1
        || packed.steady_dispatches == 0
        || !finite_positive(packed.cold_elapsed_ms)
        || !finite_positive(packed.steady_elapsed_ms)
        || packed.retained_device_packed_logical_bytes == 0
        || packed.peak_packed_argument_logical_bytes == 0
        || packed.peak_packed_transient_logical_bytes == 0
        || packed.retained_device_code_bytes == 0
        || packed.retained_device_metadata_bytes == 0
        || packed
            .retained_device_code_bytes
            .saturating_add(packed.retained_device_metadata_bytes)
            != packed.retained_device_packed_logical_bytes
        || arm.memory.packed_logical_payload_bytes == 0
        || arm.memory.packed_metadata_bytes == 0
        || arm.memory.packed_device_bytes == 0
        || arm.continuation_dispatches == 0
        || !warm.kernel_warmed
        || warm.accepted_direct_calls == 0
        || warm.full_cache_dequantizations != 0
        || warm.dense_active
        || warm.failed_dispatches != 0
        || warm.compile_jit_attempts != 0
        || warm.cold_dispatches != 0
        || warm.steady_dispatches == 0
        || !finite_positive(warm.steady_elapsed_ms)
        || !warm.fallback_reasons.is_empty()
    {
        return Err("SC-20676 packed representation contract is not proven".into());
    }
    let resident = &arm.memory.phases[4];
    let weights = &arm.memory.phases[1];
    if arm.memory.packed_logical_payload_bytes != packed.retained_device_code_bytes
        || arm.memory.packed_metadata_bytes != packed.retained_device_metadata_bytes
        || arm.memory.packed_device_bytes != active_resident_delta_from_phases(weights, resident)
        || arm.memory.packed_device_bytes < packed.retained_device_packed_logical_bytes
    {
        return Err(
            "SC-20676 packed storage mixes logical representation with allocator residency".into(),
        );
    }
    let cancel = arm
        .cancellation
        .as_ref()
        .ok_or("packed arm lacks cancellation evidence")?;
    if cancel.finish_reason != "cancelled"
        || cancel.emitted_tokens != 1
        || cancel.retained_before_reset == 0
        || cancel.retained_after_reset != 0
        || !cancel.release_verified
        || cancel.packed_before_reset.accepted_direct_calls == 0
        || cancel.packed_before_reset.full_cache_dequantizations != 0
        || cancel.packed_before_reset.dense_active
        || cancel.packed_before_reset.failed_dispatches != 0
    {
        return Err("SC-20676 cancellation release is unproven".into());
    }
    let fallback = arm
        .fallback
        .as_ref()
        .ok_or("packed arm lacks fallback evidence")?;
    if fallback.route != "dense-fallback"
        || fallback.reason != "additive attention masks require dense fallback"
        || !fallback.executed_dense_request
        || fallback.packed_dispatch_attempts_before != 0
        || fallback.packed_dispatch_attempts_after != 0
    {
        return Err("SC-20676 fallback is not the precise pre-mutation additive-mask route".into());
    }
    Ok(())
}

/// Reject malformed or deceptively dense evidence while constructing the semantic core.
fn validate_sc20676_receipt_core(receipt: &Sc20676Receipt) -> std::result::Result<(), String> {
    if receipt.schema_version != SC20676_SCHEMA_VERSION
        || receipt.harness_version != SC20676_HARNESS_VERSION
        || receipt.thresholds.contract_hash != SC20676_CONTRACT_HASH
        || !is_digest(&receipt.baseline.receipt_sha256)
        || !is_digest(&receipt.baseline.campaign_manifest_sha256)
        || ![1, 2].contains(&receipt.baseline.campaign_schedule_version)
        || !is_digest(&receipt.baseline.model_file_sha256)
        || !is_digest(&receipt.baseline.snapshot_inventory_sha256)
        || receipt.baseline.model_id.is_empty()
        || receipt.baseline.model_repository.is_empty()
        || !is_revision(&receipt.baseline.model_revision)
        || !is_revision(&receipt.baseline.scene_works_revision)
        || !is_revision(&receipt.baseline.inference_revision)
        || !is_digest(&receipt.baseline.campaign_session_id)
        || !is_digest(&receipt.baseline.campaign_global_identity_sha256)
        || receipt.baseline.context_band != campaign::SC20676_BASELINE_CONTEXT_BAND
        || receipt.baseline.context_payload_tokens < SC20676_MIN_LONG_CONTEXT_TOKENS as u64
        || ![64, 128, 256].contains(&receipt.baseline.head_dimension)
        || receipt.thresholds.max_logit_abs_error != SC20676_MAX_LOGIT_ABS_ERROR
        || receipt.thresholds.min_greedy_token_agreement != SC20676_MIN_GREEDY_TOKEN_AGREEMENT
    {
        return Err("SC-20676 receipt identity or thresholds are malformed".into());
    }
    validate_arm(&receipt.dense)?;
    validate_arm(&receipt.packed)?;
    if receipt.dense.mode != "dense"
        || receipt.packed.mode != "packed"
        || receipt.dense.family != receipt.packed.family
    {
        return Err("SC-20676 receipt arms are not a same-family dense/packed pair".into());
    }
    if receipt
        .packed
        .kernel_profile
        .as_ref()
        .is_none_or(|profile| profile.head_dimension != receipt.baseline.head_dimension)
    {
        return Err("SC-20676 packed kernel profile does not match baseline model geometry".into());
    }
    if receipt.dense.prompt_tokens < SC20676_MIN_LONG_CONTEXT_TOKENS as u64
        || receipt.dense.prompt_tokens != receipt.baseline.context_payload_tokens
        || receipt.packed.prompt_tokens != receipt.baseline.context_payload_tokens
        || receipt.dense.input.ids_len != receipt.baseline.context_payload_tokens
        || receipt.packed.input.ids_len != receipt.baseline.context_payload_tokens
        || receipt.dense.run_nonce != receipt.packed.run_nonce
        || receipt.dense.executable_sha256 != receipt.packed.executable_sha256
        || !is_digest(&receipt.dense.snapshot_inventory_sha256)
        || receipt.dense.snapshot_inventory_sha256 != receipt.packed.snapshot_inventory_sha256
        || receipt.dense.snapshot_inventory_sha256 != receipt.baseline.snapshot_inventory_sha256
        || receipt.dense.input != receipt.packed.input
        || receipt.dense.output.logits_len != receipt.packed.output.logits_len
        || receipt.dense.output.logits_shape != receipt.packed.output.logits_shape
        || receipt.dense.output.logits_dtype != receipt.packed.output.logits_dtype
        || receipt.dense.output.token_len != receipt.packed.output.token_len
        || receipt.dense.output.tokens_sha256 != receipt.packed.output.tokens_sha256
        || !receipt.dense.elapsed_ms.is_finite()
        || !receipt.packed.elapsed_ms.is_finite()
    {
        return Err("SC-20676 receipt arm evidence is incomplete".into());
    }
    let model = campaign::benchmark_model(&receipt.dense.family, false)?;
    let expected_model_id = format!(
        "{}@{};architecture={};inventory={}",
        model.repository,
        model.revision,
        model.architecture,
        receipt.baseline.snapshot_inventory_sha256
    );
    if receipt.baseline.model_repository != model.repository
        || receipt.baseline.model_revision != model.revision
        || receipt.baseline.model_id != expected_model_id
    {
        return Err("SC-20676 baseline model binding does not match the receipt family".into());
    }
    if receipt.dense.provenance != receipt.packed.provenance
        || !is_revision(&receipt.dense.provenance.inference_revision)
        || !is_revision(&receipt.dense.provenance.scene_works_revision)
        || receipt.dense.provenance.inference_revision != receipt.baseline.inference_revision
        || receipt.dense.provenance.scene_works_revision != receipt.baseline.scene_works_revision
        || !is_digest(&receipt.dense.provenance.dependency_lock_sha256)
        || [
            receipt.dense.provenance.os.as_str(),
            receipt.dense.provenance.xcode.as_str(),
            receipt.dense.provenance.hardware.as_str(),
            receipt.dense.provenance.power_mode.as_str(),
            receipt.dense.provenance.thermal_state.as_str(),
        ]
        .iter()
        .any(|field| field.is_empty())
        || receipt.dense.provenance.thermal_state != "nominal"
    {
        return Err("SC-20676 worker provenance is incomplete or differs by arm".into());
    }
    let packed = receipt.packed.packed.as_ref().expect("checked above");
    if packed.accepted_direct_calls == 0
        || packed.full_cache_dequantizations != 0
        || packed.dense_active
        || packed.retained_device_packed_logical_bytes == 0
        || receipt.packed.continuation_dispatches == 0
        || receipt
            .packed
            .warm_packed
            .as_ref()
            .is_none_or(|warm| !warm.kernel_warmed || warm.steady_dispatches == 0)
    {
        return Err("SC-20676 packed arm did not prove direct warm compressed decode".into());
    }
    let quality = receipt
        .packed
        .quality
        .as_ref()
        .ok_or("missing SC-20676 quality evidence")?;
    let (needle_retrieval, needle_discriminating) = sc20676_needle_observation(
        &receipt.dense.output.needle_output,
        &receipt.packed.output.needle_output,
    );
    if !quality.max_logit_abs_error.is_finite()
        || quality.max_logit_abs_error > receipt.thresholds.max_logit_abs_error
        || !(0.0..=1.0).contains(&quality.greedy_token_agreement)
        || quality.greedy_token_agreement < receipt.thresholds.min_greedy_token_agreement
        || quality.needle_retrieval != needle_retrieval
        || quality.needle_discriminating != needle_discriminating
        || !quality.needle_retrieval
    {
        return Err("SC-20676 packed quality is outside declared bounds".into());
    }
    if receipt.packed.timing_summary.mean_tokens_per_second
        < receipt.dense.timing_summary.mean_tokens_per_second
    {
        return Err("SC-20676 packed warm steady decode regressed the dense baseline".into());
    }
    let fallback = receipt
        .packed
        .fallback
        .as_ref()
        .ok_or("missing SC-20676 fallback evidence")?;
    if fallback.route != "dense-fallback"
        || fallback.reason.is_empty()
        || !fallback.executed_dense_request
    {
        return Err("SC-20676 fallback did not execute a pre-mutation dense request".into());
    }
    let cancel = receipt
        .packed
        .cancellation
        .as_ref()
        .ok_or("missing SC-20676 cancellation evidence")?;
    if cancel.finish_reason != "cancelled"
        || cancel.emitted_tokens == 0
        || !cancel.release_verified
        || cancel.retained_after_reset != 0
    {
        return Err("SC-20676 cancellation cleanup is unproven".into());
    }
    Ok(())
}

/// Validate a published receipt, including its mandatory semantic-core seal. Construction uses
/// the private core validator above; every consumer and publication path uses this sealed form.
pub fn validate_sc20676_receipt(receipt: &Sc20676Receipt) -> std::result::Result<(), String> {
    validate_sc20676_receipt_core(receipt)?;
    if !is_digest(&receipt.receipt_sha256) {
        return Err("SC-20676 receipt seal is missing or malformed".into());
    }
    let mut core = receipt.clone();
    core.receipt_sha256.clear();
    if campaign::seal_bytes(&core.bytes().map_err(|e| e.to_string())?) != receipt.receipt_sha256 {
        return Err("SC-20676 receipt seal does not bind its contents".into());
    }
    Ok(())
}

/// Bind the frozen long-context row from a complete SC-20671 campaign to the exact immutable
/// candidate snapshot being tested.  A standalone receipt, a different coordinate, or a model
/// inventory substituted after baseline publication fails closed.
pub fn bind_sc20671_baseline(
    campaign_directory: impl AsRef<Path>,
    snapshot: &Path,
    family: &str,
) -> std::result::Result<Sc20676BaselineBinding, String> {
    let campaign_directory = campaign_directory.as_ref();
    let baseline = campaign::select_sc20676_baseline_row(campaign_directory, family)?;
    let manifest_bytes =
        fs::read(campaign_directory.join("campaign.json")).map_err(|e| e.to_string())?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    let campaign_schedule_version = manifest
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        .filter(|version| [1, 2].contains(version))
        .ok_or("SC-20671 baseline has no supported complete schedule version")?;
    let receipt = baseline.receipt;
    let campaign_global_identity_sha256 =
        campaign::seal_bytes(&campaign::campaign_global_identity(&receipt)?);
    let inventory = campaign::validate_sc20676_candidate_snapshot(family, snapshot)?;
    let spec = campaign::benchmark_model(family, false)?;
    let expected_model_id = format!(
        "{}@{};architecture={};inventory={}",
        spec.repository, spec.revision, spec.architecture, inventory.sha256
    );
    if receipt.provenance.model_id != expected_model_id
        || receipt.provenance.model_file_sha256 != inventory.sha256
        || receipt.provenance.model_file_bytes != inventory.bytes
        || baseline.coordinate.context_band != campaign::SC20676_BASELINE_CONTEXT_BAND
    {
        return Err(
            "SC-20671 baseline family/model/revision/inventory does not match tested snapshot"
                .into(),
        );
    }
    Ok(Sc20676BaselineBinding {
        receipt_sha256: receipt.receipt_sha256,
        campaign_manifest_sha256: campaign::seal_bytes(&manifest_bytes),
        campaign_schedule_version,
        model_file_sha256: receipt.provenance.model_file_sha256,
        model_id: receipt.provenance.model_id.clone(),
        model_repository: spec.repository.into(),
        model_revision: spec.revision.into(),
        snapshot_inventory_sha256: inventory.sha256,
        scene_works_revision: receipt.provenance.scene_works_revision,
        inference_revision: receipt.provenance.inference_revision,
        campaign_session_id: receipt.provenance.campaign_session_id,
        campaign_global_identity_sha256,
        context_band: baseline.coordinate.context_band.into(),
        context_payload_tokens: receipt.geometry.context_payload_tokens,
        head_dimension: receipt.geometry.head_dimension,
    })
}

fn command_provenance(
    command: &str,
    args: &[&str],
    label: &str,
) -> std::result::Result<String, String> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|e| format!("{label}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{label}: command failed"));
    }
    let value = String::from_utf8(output.stdout)
        .map_err(|e| format!("{label}: {e}"))?
        .trim()
        .to_owned();
    if value.is_empty() {
        return Err(format!("{label}: command returned empty output"));
    }
    Ok(value)
}

fn checked_clean_revision(root: &Path, label: &str) -> std::result::Result<String, String> {
    let root = root.to_str().ok_or("non-UTF8 repository root")?;
    let output = Command::new("git")
        .args(["-C", root, "status", "--porcelain"])
        .output()
        .map_err(|e| format!("{label}: {e}"))?;
    if !output.status.success() {
        return Err(format!("{label}: cannot inspect source tree"));
    }
    let status = String::from_utf8(output.stdout).map_err(|e| format!("{label}: {e}"))?;
    if !status.is_empty() {
        return Err(format!("{label} source tree is dirty"));
    }
    let revision = command_provenance("git", &["-C", root, "rev-parse", "HEAD"], label)?;
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{label} revision is not a full commit id"));
    }
    Ok(revision)
}

fn worker_provenance() -> std::result::Result<Sc20676Provenance, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let scene_works_root = PathBuf::from(
        std::env::var("SCENEWORKS_ROOT")
            .map_err(|_| "SCENEWORKS_ROOT is required for SC-20676 provenance")?,
    );
    campaign::checked_repository_identity(root, campaign::INFERENCE_REPOSITORY)?;
    campaign::checked_repository_identity(&scene_works_root, campaign::SCENEWORKS_REPOSITORY)?;
    let inference_revision = checked_clean_revision(root, "inference")?;
    let scene_works_revision = checked_clean_revision(&scene_works_root, "SceneWorks")?;
    let thermal = command_provenance("pmset", &["-g", "therm"], "thermal state")?;
    let thermal_state = campaign::normalize_pmset_thermal(&thermal)?;
    let mlx = campaign::locked_mlx_identity(include_bytes!("../../../../Cargo.lock"))?;
    Ok(Sc20676Provenance {
        scene_works_repository: campaign::SCENEWORKS_REPOSITORY.into(),
        scene_works_revision,
        inference_repository: campaign::INFERENCE_REPOSITORY.into(),
        inference_revision,
        dependency_lock_sha256: campaign::seal_bytes(include_bytes!("../../../../Cargo.lock")),
        mlx_version: mlx.version,
        mlx_source: mlx.source,
        mlx_revision: mlx.revision,
        os: std::env::consts::OS.into(),
        os_version: command_provenance("sw_vers", &["-productVersion"], "OS version")?,
        xcode: command_provenance("xcodebuild", &["-version"], "xcode")?,
        hardware: command_provenance("sysctl", &["-n", "hw.model"], "hardware")?,
        metal_device: command_provenance(
            "system_profiler",
            &["SPDisplaysDataType"],
            "Metal device",
        )?,
        power_mode: command_provenance("pmset", &["-g", "custom"], "power mode")?,
        thermal_state,
        source_tree_clean: true,
    })
}

fn max_abs(a: &[f32], b: &[f32]) -> std::result::Result<f64, String> {
    if a.is_empty() || a.len() != b.len() {
        return Err("SC-20676 logit vectors are empty or have different lengths".into());
    }
    let mut maximum = 0.0_f64;
    for (dense, packed) in a.iter().zip(b) {
        let dense = f64::from(*dense);
        let packed = f64::from(*packed);
        if !dense.is_finite() || !packed.is_finite() {
            return Err("SC-20676 logits contain a non-finite value".into());
        }
        maximum = maximum.max((dense - packed).abs());
    }
    Ok(maximum)
}

fn agreement(a: &[i32], b: &[i32]) -> f64 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).filter(|(a, b)| a == b).count() as f64 / a.len() as f64
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn input_binding(ids: &[i32]) -> Sc20676InputBinding {
    Sc20676InputBinding {
        ids_sha256: sha256_bytes(&i32_bytes(ids)),
        ids_len: ids.len() as u64,
    }
}

fn output_binding(
    logits: &[f32],
    shape: &[i32],
    dtype: &str,
    tokens: &[i32],
    needle_expected: &str,
    needle_output: String,
) -> Sc20676OutputBinding {
    Sc20676OutputBinding {
        logits_sha256: sha256_bytes(&f32_bytes(logits)),
        logits_len: logits.len() as u64,
        logits_shape: shape.to_vec(),
        logits_dtype: dtype.into(),
        tokens_sha256: sha256_bytes(&i32_bytes(tokens)),
        token_len: tokens.len() as u64,
        needle_expected: needle_expected.into(),
        needle_output,
    }
}

fn phase(name: &str, sample: &campaign::MemorySample) -> Sc20676MemoryPhase {
    Sc20676MemoryPhase {
        phase: name.into(),
        captured_at: sample.captured_at.clone(),
        phys_footprint_bytes: sample.current_bytes,
        phys_footprint_peak_bytes: sample.peak_bytes,
        mlx_active_bytes: sample.mlx_active_bytes,
        mlx_cache_bytes: sample.mlx_cache_bytes,
        mlx_peak_bytes: sample.mlx_peak_bytes,
    }
}

fn begin_decode_peak_window(
) -> std::result::Result<(Sc20676PeakWindow, campaign::MemorySample), String> {
    mlx_rs::memory::reset_peak_memory();
    let sample = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    if sample.mlx_active_bytes == 0 || sample.mlx_peak_bytes != 0 {
        return Err(
            "SC-20676 decode peak-window reset did not produce a positive active baseline and zero peak"
                .into(),
        );
    }
    let window = Sc20676PeakWindow {
        started_at: sample.captured_at.clone(),
        baseline_active_bytes: sample.mlx_active_bytes,
        baseline_cache_bytes: sample.mlx_cache_bytes,
        reset_peak_bytes: sample.mlx_peak_bytes,
    };
    Ok((window, sample))
}

fn file_sha256(path: &Path) -> std::result::Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn run_nonce(executable_sha256: &str) -> std::result::Result<String, String> {
    let mut bytes = [0_u8; 32];
    fs::File::open("/dev/urandom")
        .map_err(|e| e.to_string())?
        .read_exact(&mut bytes)
        .map_err(|e| e.to_string())?;
    Ok(sha256_bytes(
        &[bytes.as_slice(), executable_sha256.as_bytes()].concat(),
    ))
}

fn context_ids(tokenizer: &Tokenizer, cfg: &ModelConfig, target_tokens: u64) -> Result<Vec<i32>> {
    let limit = usize::try_from(cfg.max_position_embeddings)
        .map_err(|_| Error::Config("negative context window".into()))?;
    let target = usize::try_from(target_tokens)
        .map_err(|_| Error::Config("SC-20676 baseline context does not fit usize".into()))?;
    if target < SC20676_MIN_LONG_CONTEXT_TOKENS || target > limit {
        return Err(Error::Config(
            "SC-20676 baseline context is outside the model window".into(),
        ));
    }
    let needle = SC20676_NEEDLE;
    let suffix = campaign::NEEDLE_FIXTURE_QUESTION;
    let question = suffix.split_inclusive('?').next().unwrap_or(suffix);
    let mut ids = tokenizer
        .encode(
            &format!("{}{needle}. ", campaign::NEEDLE_FIXTURE_STATEMENT_PREFIX),
            false,
        )
        .map_err(|e| Error::Msg(e.to_string()))?;
    let suffix_ids = tokenizer
        .encode(suffix, false)
        .map_err(|e| Error::Msg(e.to_string()))?;
    let filler = tokenizer
        .encode(" packed-cache evidence ", false)
        .map_err(|e| Error::Msg(e.to_string()))?;
    if filler.is_empty()
        || ids
            .len()
            .checked_add(suffix_ids.len())
            .is_none_or(|minimum| minimum > target)
    {
        return Err(Error::Config(
            "SC-20676 tokenizer cannot construct the exact frozen baseline context".into(),
        ));
    }
    let filler_tokens = target - ids.len() - suffix_ids.len();
    ids.extend(filler.iter().copied().cycle().take(filler_tokens));
    ids.extend(suffix_ids);
    if ids.len() != target || ids.len() > limit {
        return Err(Error::Config(
            "SC-20676 tokenizer did not preserve the exact frozen baseline context".into(),
        ));
    }
    let decoded = tokenizer
        .decode(&ids, true)
        .map_err(|e| Error::Msg(e.to_string()))?;
    if !decoded.contains(needle) || !decoded.contains(question) {
        return Err(Error::Config(
            "SC-20676 exact token construction lost the needle/question contract".into(),
        ));
    }
    Ok(ids.into_iter().map(|id| id as i32).collect())
}

fn generation_config() -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: 16,
        seed: Some(0),
        stop_tokens: Vec::new(),
        ..Default::default()
    }
}

fn route_name(route: &CacheRoute) -> String {
    match route {
        CacheRoute::ExperimentalPacked => "experimental-packed".into(),
        CacheRoute::DenseFallback { .. } => "dense-fallback".into(),
    }
}

fn packed_cache(
    model: &CausalLm,
    handle: CompiledKernelHandle,
    prompt_tokens: usize,
    has_mask: bool,
) -> (CacheRoute, Box<dyn KvCache>) {
    let selection = model.select_cache_with_packed_reader(handle, 1, prompt_tokens, has_mask);
    (selection.route().clone(), selection.into_cache())
}

/// SC-20676 uses the bf16 decode cache used by the causal decoder.  Keep the calculation next to
/// the runtime capture so a receipt cannot smuggle in a caller-selected geometry value.
fn dense_kv_geometry(cfg: &ModelConfig, capacity: usize) -> std::result::Result<u64, String> {
    campaign::dense_kv_bytes(
        1,
        u64::try_from(cfg.num_layers).map_err(|_| "layer count does not fit u64")?,
        u64::try_from(cfg.num_kv_heads).map_err(|_| "negative KV-head count")?,
        u64::try_from(capacity).map_err(|_| "context capacity does not fit u64")?,
        u64::try_from(cfg.head_dim).map_err(|_| "negative head dimension")?,
        2,
    )
}

/// Comparison-only dense attention score storage for one decode query across the loaded decoder.
/// This is derived from request geometry and is deliberately kept distinct from allocator samples.
fn score_matrix_geometry(cfg: &ModelConfig, capacity: usize) -> std::result::Result<u64, String> {
    [
        u64::try_from(cfg.num_layers).map_err(|_| "layer count does not fit u64")?,
        u64::try_from(cfg.num_heads).map_err(|_| "negative query-head count")?,
        u64::try_from(capacity).map_err(|_| "context capacity does not fit u64")?,
        1,
        u64::try_from(std::mem::size_of::<f32>())
            .map_err(|_| "score element size does not fit u64")?,
    ]
    .into_iter()
    .try_fold(1_u64, |bytes, factor| bytes.checked_mul(factor))
    .ok_or("dense score-matrix geometry overflows u64".into())
}

fn observed_kv_bytes(weights: &campaign::MemorySample, resident: &campaign::MemorySample) -> u64 {
    resident
        .mlx_active_bytes
        .saturating_sub(weights.mlx_active_bytes)
        .max(
            resident
                .mlx_cache_bytes
                .saturating_sub(weights.mlx_cache_bytes),
        )
}

fn active_resident_delta(
    weights: &campaign::MemorySample,
    resident: &campaign::MemorySample,
) -> u64 {
    resident
        .mlx_active_bytes
        .saturating_sub(weights.mlx_active_bytes)
}

fn active_resident_delta_from_phases(
    weights: &Sc20676MemoryPhase,
    resident: &Sc20676MemoryPhase,
) -> u64 {
    resident
        .mlx_active_bytes
        .saturating_sub(weights.mlx_active_bytes)
}

fn transient_peak_above_active(peak: &campaign::MemorySample) -> u64 {
    peak.mlx_peak_bytes.saturating_sub(peak.mlx_active_bytes)
}

/// Two warmups are deliberately discarded; all five stored samples are independent cache
/// allocations and decode operations.  This prevents a single worker elapsed time from becoming
/// the performance gate.
fn dense_timing_trial(
    model: &CausalLm,
    ids: &[i32],
    config: &GenerationConfig,
) -> std::result::Result<Sc20676TimingSample, String> {
    let mut cache: Box<dyn KvCache> = Box::new(model.new_cache());
    let prefill_started = Instant::now();
    let logits = model
        .decode_logits(&input_ids(ids), cache.as_mut(), 0)
        .map_err(|e| e.to_string())?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let output = generate_from_prefill(
        model,
        cache.as_mut(),
        logits,
        ids.to_vec(),
        config,
        &CancelFlag::new(),
        &mut |event| {
            if first_token_ms.is_none() && matches!(event, StreamEvent::Token { .. }) {
                first_token_ms = Some(decode_started.elapsed().as_secs_f64() * 1000.0);
            }
        },
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    cache.reset().map_err(|e| e.to_string())?;
    let ttft_ms = first_token_ms.ok_or("dense timing trial emitted no token")?;
    let steady_ms = decode_ms - ttft_ms;
    if output.tokens.len() < 2 || !finite_positive(steady_ms) {
        return Err("dense timing trial has no steady decode evidence".into());
    }
    Ok(Sc20676TimingSample {
        prefill_ms,
        ttft_ms,
        first_token_ms: prefill_ms + ttft_ms,
        steady_decode_tokens_per_second: (output.tokens.len() - 1) as f64 / (steady_ms / 1000.0),
        packed_cold_dispatch_ms: 0.0,
        packed_warm_dispatch_ms: 0.0,
    })
}

fn packed_timing_trial(
    model: &CausalLm,
    ids: &[i32],
    config: &GenerationConfig,
    retained: &CompiledKernelHandle,
) -> std::result::Result<Sc20676TimingSample, String> {
    let (route, mut cache) = packed_cache(model, retained.clone(), ids.len(), false);
    if route != CacheRoute::ExperimentalPacked {
        return Err("timing trial did not select packed cache".into());
    }
    let prefill_started = Instant::now();
    let logits = model
        .decode_logits(&input_ids(ids), cache.as_mut(), 0)
        .map_err(|e| e.to_string())?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
    let decode_started = Instant::now();
    let mut first_token_ms = None;
    let output = generate_from_prefill(
        model,
        cache.as_mut(),
        logits,
        ids.to_vec(),
        config,
        &CancelFlag::new(),
        &mut |event| {
            if first_token_ms.is_none() && matches!(event, StreamEvent::Token { .. }) {
                first_token_ms = Some(decode_started.elapsed().as_secs_f64() * 1000.0);
            }
        },
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
    let evidence = model
        .packed_cache_evidence(cache.as_ref())
        .ok_or("timing cache did not expose packed evidence")?;
    cache.reset().map_err(|e| e.to_string())?;
    let ttft_ms = first_token_ms.ok_or("packed timing trial emitted no token")?;
    let steady_ms = decode_ms - ttft_ms;
    if output.tokens.len() < 2
        || !finite_positive(steady_ms)
        || evidence.accepted_direct_calls == 0
        || evidence.full_cache_dequantizations != 0
        || evidence.dense_active
        || evidence.failed_dispatches != 0
        || evidence.compile_jit_attempts != 0
        || !evidence.kernel_warmed
        || evidence.cold_dispatches != 0
        || evidence.steady_dispatches == 0
        || !finite_positive(evidence.steady_elapsed_ms)
        || !evidence.fallback_reasons.is_empty()
    {
        return Err("packed timing trial has incomplete dispatch evidence".into());
    }
    Ok(Sc20676TimingSample {
        prefill_ms,
        ttft_ms,
        first_token_ms: prefill_ms + ttft_ms,
        steady_decode_tokens_per_second: (output.tokens.len() - 1) as f64 / (steady_ms / 1000.0),
        // The retained handle was warmed once by the primary measured arm. These five independent
        // cache trials are steady-only; the real cold/JIT observation remains in `arm.packed`.
        packed_cold_dispatch_ms: 0.0,
        packed_warm_dispatch_ms: evidence.steady_elapsed_ms / evidence.steady_dispatches as f64,
    })
}

fn measured_dense_timings(
    model: &CausalLm,
    ids: &[i32],
    config: &GenerationConfig,
) -> std::result::Result<Vec<Sc20676TimingSample>, String> {
    for _ in 0..2 {
        dense_timing_trial(model, ids, config)?;
    }
    (0..5)
        .map(|_| dense_timing_trial(model, ids, config))
        .collect()
}

fn measured_packed_timings(
    model: &CausalLm,
    ids: &[i32],
    config: &GenerationConfig,
    retained: &CompiledKernelHandle,
) -> std::result::Result<Vec<Sc20676TimingSample>, String> {
    for _ in 0..2 {
        packed_timing_trial(model, ids, config, retained)?;
    }
    (0..5)
        .map(|_| packed_timing_trial(model, ids, config, retained))
        .collect()
}

fn sc20676_admitted_tokens(
    mode: &str,
    target_prompt_tokens: u64,
    policy: &campaign::CampaignSafetyPolicy,
) -> std::result::Result<(u64, u64), String> {
    let request = target_prompt_tokens
        .checked_add(16)
        .ok_or("SC-20676 prompt plus generation overflows")?;
    // The packed proof retains its measured cache while each warm/fallback/cancellation control
    // runs in a second cache. Those controls are sequential, so two is the maximum live count.
    let total = match mode {
        "dense" => request,
        "packed" => request
            .checked_mul(2)
            .ok_or("SC-20676 packed live context overflows")?,
        _ => return Err("SC-20676 worker mode must be dense or packed".into()),
    };
    if request > policy.max_request_tokens || total > policy.max_context_tokens {
        return Err(format!("SC-20676 {mode} requires request {request} and total-live {total} tokens; mandatory safety ceiling refuses"));
    }
    Ok((total, request))
}

/// Admit one arm by its runtime guards (supervised worker, footprint watchdog cap, host reserve,
/// deadline, sampling) instead of an unobtainable static whole-process peak proof. The pinned
/// candidate's conservative load-plus-KV floor is recorded as the arm's estimate and still refuses
/// before spawn when it alone exceeds the child cap.
fn sc20676_runtime_admission(
    family: &str,
    snapshot: &Path,
    total_tokens: u64,
    request_tokens: u64,
    policy: &campaign::CampaignSafetyPolicy,
) -> std::result::Result<campaign::ReceiptAdmission, String> {
    let floor = campaign::static_role_footprint_budget(
        campaign::benchmark_model(family, false)?,
        snapshot,
        total_tokens,
        request_tokens,
    )?;
    campaign::runtime_guarded_admission(policy, floor)
}

/// Packed needle outcome relative to the same-weights dense output: exact recovery when the dense
/// run recovered the needle, otherwise exact agreement with the dense output, flagged as
/// non-discriminating so a shared miss is never counted as retrieval.
fn sc20676_needle_observation(dense_output: &str, packed_output: &str) -> (bool, bool) {
    if dense_output == SC20676_NEEDLE {
        (packed_output == SC20676_NEEDLE, true)
    } else {
        (packed_output == dense_output, false)
    }
}

/// Execute one fresh worker.  This is intentionally the only live model entry point; parent mode
/// merely spawns separate processes and seals their output.
#[allow(clippy::too_many_arguments)] // The worker boundary names every sealed parent input.
pub fn run_sc20676_worker(
    snapshot: &Path,
    family: &str,
    mode: &str,
    nonce: &str,
    expected_executable_sha256: &str,
    target_prompt_tokens: u64,
    policy: &campaign::CampaignSafetyPolicy,
    captured_provenance: &Sc20676Provenance,
) -> std::result::Result<Sc20676Arm, String> {
    let started = Instant::now();
    if !is_digest(nonce) || !is_digest(expected_executable_sha256) {
        return Err("worker nonce or executable seal is malformed".into());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let executable_sha256 = file_sha256(&executable)?;
    if executable_sha256 != expected_executable_sha256 {
        return Err("worker executable does not match parent-sealed executable".into());
    }
    let provenance = captured_provenance.clone();
    let before = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    let cfg = ModelConfig::from_dir(snapshot).map_err(|e| e.to_string())?;
    let (total_tokens, request_tokens) =
        sc20676_admitted_tokens(mode, target_prompt_tokens, policy)?;
    let admission =
        sc20676_runtime_admission(family, snapshot, total_tokens, request_tokens, policy)?;
    let native = u64::try_from(cfg.max_position_embeddings)
        .map_err(|_| "SC-20676 model native context is invalid")?;
    if target_prompt_tokens
        .checked_add(16)
        .is_none_or(|total| total > native)
    {
        return Err("SC-20676 prompt plus generation exceeds pinned native context".into());
    }
    let actual_family = match cfg.architecture {
        Architecture::Llama => "llama",
        Architecture::Qwen3 => "qwen",
        _ => return Err("SC-20676 only accepts causal Llama or Qwen3 snapshots".into()),
    };
    if actual_family != family {
        return Err(format!(
            "requested {family} worker for {actual_family} snapshot"
        ));
    }
    let snapshot_inventory = campaign::validate_sc20676_candidate_snapshot(family, snapshot)?;
    let tokenizer =
        Tokenizer::from_file(snapshot.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let ids = context_ids(&tokenizer, &cfg, target_prompt_tokens).map_err(|e| e.to_string())?;
    if u64::try_from(ids.len()).map_err(|_| "prompt length does not fit u64")?
        != target_prompt_tokens
    {
        return Err("SC-20676 worker prompt length does not match frozen baseline".into());
    }
    let model = CausalLm::from_weights(
        &Weights::from_dir(snapshot).map_err(|e| e.to_string())?,
        "",
        cfg.clone(),
    )
    .map_err(|e| e.to_string())?;
    // MLX loads lazily. Force a real product decode and host materialization before declaring the
    // weights-only boundary; every later cache delta and release check is relative to this fence.
    let mut materialization_cache: Box<dyn KvCache> = Box::new(model.new_cache());
    let materialized = model
        .decode_logits(&input_ids(&ids[..1]), materialization_cache.as_mut(), 0)
        .map_err(|e| e.to_string())?;
    let _ = to_f32_host(&materialized).map_err(|e| e.to_string())?;
    drop(materialized);
    materialization_cache.reset().map_err(|e| e.to_string())?;
    drop(materialization_cache);
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let config = generation_config();

    if mode == "dense" {
        let weights_loaded =
            campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
        let mut cache: Box<dyn KvCache> = Box::new(model.new_cache());
        let logits = model
            .decode_logits(&input_ids(&ids), cache.as_mut(), 0)
            .map_err(|e| e.to_string())?;
        let logit_values = to_f32_host(&logits).map_err(|e| e.to_string())?;
        let logit_shape = logits.shape().to_vec();
        let logit_dtype = format!("{:?}", logits.dtype());
        let resident = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
        let (decode_peak_window, decode_window_start) = begin_decode_peak_window()?;
        let out = generate_from_prefill(
            &model,
            cache.as_mut(),
            logits,
            ids.clone(),
            &config,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
        )
        .map_err(|e| e.to_string())?;
        if out.tokens.is_empty() {
            return Err("dense SC-20676 control emitted no tokens".into());
        }
        let decode_peak = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
        let needle_output = tokenizer
            .decode(
                &out.tokens.iter().map(|id| *id as u32).collect::<Vec<_>>(),
                true,
            )
            .map_err(|e| e.to_string())?
            .trim()
            .to_owned();
        let output = output_binding(
            &logit_values,
            &logit_shape,
            &logit_dtype,
            &out.tokens,
            SC20676_NEEDLE,
            needle_output,
        );
        let dense_theoretical_kv_bytes = dense_kv_geometry(&cfg, ids.len())?;
        let theoretical_score_matrix_bytes = score_matrix_geometry(&cfg, ids.len())?;
        let observed_dense_kv_bytes = observed_kv_bytes(&weights_loaded, &resident);
        if observed_dense_kv_bytes == 0 {
            return Err("dense cache did not increase the MLX resident boundary".into());
        }
        cache.reset().map_err(|e| e.to_string())?;
        drop(cache);
        let timings = measured_dense_timings(&model, &ids, &config)?;
        mlx_rs::memory::clear_cache();
        let after = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
        let final_snapshot_inventory =
            campaign::validate_sc20676_candidate_snapshot(family, snapshot)?;
        validate_snapshot_inventory_unchanged(&snapshot_inventory, &final_snapshot_inventory)?;
        return finish_arm(Sc20676Arm {
            arm_sha256: String::new(),
            run_nonce: nonce.into(),
            worker_pid: std::process::id(),
            executable_sha256,
            family: family.into(),
            mode: mode.into(),
            snapshot_inventory_sha256: final_snapshot_inventory.sha256,
            prompt_tokens: ids.len() as u64,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            peak_mlx_bytes: [
                &before,
                &weights_loaded,
                &resident,
                &decode_window_start,
                &decode_peak,
                &after,
            ]
            .into_iter()
            .map(|sample| sample.mlx_peak_bytes)
            .max()
            .unwrap_or(0),
            peak_phys_footprint_bytes: [
                &before,
                &weights_loaded,
                &resident,
                &decode_window_start,
                &decode_peak,
                &after,
            ]
            .into_iter()
            .map(|sample| sample.peak_bytes)
            .max()
            .unwrap_or(0),
            provenance,
            input: input_binding(&ids),
            output,
            timings: timings.clone(),
            timing_summary: timing_summary(&timings)?,
            memory: Sc20676MemoryEvidence {
                phases: vec![
                    phase("process-start", &before),
                    phase("weights-loaded", &weights_loaded),
                    phase("prefill-cache-resident", &resident),
                    phase("decode-window-start", &decode_window_start),
                    phase("decode-complete", &decode_peak),
                    phase("reset-release", &after),
                ],
                decode_peak_window,
                dense_theoretical_kv_bytes,
                observed_dense_kv_bytes,
                packed_logical_payload_bytes: 0,
                packed_metadata_bytes: 0,
                packed_device_bytes: 0,
                transient_workspace_bytes: transient_peak_above_active(&decode_peak),
                theoretical_dense_reconstruction_bytes: dense_theoretical_kv_bytes,
                theoretical_score_matrix_bytes,
                release_verified: after.mlx_active_bytes == weights_loaded.mlx_active_bytes
                    && after.mlx_cache_bytes == weights_loaded.mlx_cache_bytes
                    && after.current_bytes
                        <= weights_loaded
                            .current_bytes
                            .saturating_add(campaign::POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES),
            },
            route: "dense".into(),
            kernel_profile: None,
            packed: None,
            warm_packed: None,
            continuation_dispatches: 0,
            quality: None,
            fallback: None,
            cancellation: None,
            admission,
        });
    }
    if mode != "packed" {
        return Err("SC-20676 worker mode must be dense or packed".into());
    }

    let dense_memory_before =
        campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    let dense_logits = {
        let mut dense = model.new_cache();
        let logits = model
            .decode_logits(&input_ids(&ids), &mut dense, 0)
            .map_err(|e| e.to_string())?;
        let values = to_f32_host(&logits).map_err(|e| e.to_string())?;
        let dense_memory_resident =
            campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
        dense.reset().map_err(|e| e.to_string())?;
        (values, dense_memory_resident)
    };
    let dense_observed_kv_bytes = observed_kv_bytes(&dense_memory_before, &dense_logits.1);
    if dense_observed_kv_bytes == 0 {
        return Err("dense parity control did not increase the MLX resident boundary".into());
    }
    let dense_tokens = {
        let mut dense = model.new_cache();
        let generated = generate_with_cache(
            &model,
            &ids,
            &mut dense,
            &config,
            &CancelFlag::new(),
            &mut |_| {},
        )
        .map_err(|e| e.to_string())?;
        dense.reset().map_err(|e| e.to_string())?;
        generated.tokens
    };
    // The dense parity controls are deliberately outside the packed arm's measurement window.
    // Their cache objects are dropped, allocator cache is purged, and MLX high-water is reset
    // before the packed weights-only boundary is sampled.
    mlx_rs::memory::clear_cache();
    mlx_rs::memory::reset_peak_memory();
    let weights_loaded = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    // The cache clone/snapshot contract owns this non-threaded Metal object through its existing
    // `Arc` handle; it is never sent across threads by this single-worker harness.
    let kernel = PackedMetalKernel::new().map_err(|e| e.to_string())?;
    let head_dimension =
        usize::try_from(cfg.head_dim).map_err(|_| "SC-20676 head dimension does not fit usize")?;
    let tuning = kernel
        .tuning_profile(head_dimension)
        .ok_or("SC-20676 packed kernel has no tuning profile for the model head dimension")?;
    let kernel_profile = Sc20676KernelProfile {
        metal_device: provenance.metal_device.clone(),
        gpu_family: tuning.gpu_family.into(),
        qualification: "conservative-default".into(),
        head_dimension: u64::try_from(head_dimension)
            .map_err(|_| "SC-20676 head dimension does not fit u64")?,
        threads: u64::try_from(tuning.threads)
            .map_err(|_| "SC-20676 thread count does not fit u64")?,
        simd_groups: u64::try_from(tuning.simd_groups)
            .map_err(|_| "SC-20676 SIMD-group count does not fit u64")?,
        values_per_thread: u64::try_from(tuning.values_per_thread)
            .map_err(|_| "SC-20676 values-per-thread count does not fit u64")?,
    };
    #[allow(clippy::arc_with_non_send_sync)]
    let retained = CompiledKernelHandle::new(Arc::new(kernel));
    let (route, mut cache) = packed_cache(&model, retained.clone(), ids.len(), false);
    if route != CacheRoute::ExperimentalPacked {
        return Err(format!("packed selection refused real request: {route:?}"));
    }
    let first_logits = model
        .decode_logits(&input_ids(&ids), cache.as_mut(), 0)
        .map_err(|e| e.to_string())?;
    let packed_prefill = model
        .packed_cache_evidence(cache.as_ref())
        .ok_or("packed cache did not expose model evidence")?;
    let packed_logits = to_f32_host(&first_logits).map_err(|e| e.to_string())?;
    let packed_logit_shape = first_logits.shape().to_vec();
    let packed_logit_dtype = format!("{:?}", first_logits.dtype());
    let resident = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    let (decode_peak_window, decode_window_start) = begin_decode_peak_window()?;
    let out = generate_from_prefill(
        &model,
        cache.as_mut(),
        first_logits,
        ids.clone(),
        &config,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let decode_peak = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    let packed = model
        .packed_cache_evidence(cache.as_ref())
        .ok_or("packed cache lost model evidence")?;
    let continuation_dispatches = packed
        .accepted_direct_calls
        .saturating_sub(packed_prefill.accepted_direct_calls)
        as u64;
    if continuation_dispatches == 0 || packed.full_cache_dequantizations != 0 || packed.dense_active
    {
        return Err("packed continuation left compressed direct path".into());
    }
    let needle_output = tokenizer
        .decode(
            &out.tokens.iter().map(|v| *v as u32).collect::<Vec<_>>(),
            true,
        )
        .map_err(|e| e.to_string())?
        .trim()
        .to_owned();
    let dense_needle_output = tokenizer
        .decode(
            &dense_tokens.iter().map(|v| *v as u32).collect::<Vec<_>>(),
            true,
        )
        .map_err(|e| e.to_string())?
        .trim()
        .to_owned();
    let (needle_retrieval, needle_discriminating) =
        sc20676_needle_observation(&dense_needle_output, &needle_output);
    let quality = Sc20676Quality {
        max_logit_abs_error: max_abs(&dense_logits.0, &packed_logits)?,
        greedy_token_agreement: agreement(&dense_tokens, &out.tokens),
        needle_retrieval,
        needle_discriminating,
    };
    let output = output_binding(
        &packed_logits,
        &packed_logit_shape,
        &packed_logit_dtype,
        &out.tokens,
        SC20676_NEEDLE,
        needle_output,
    );

    let (_, mut warm_cache) = packed_cache(&model, retained.clone(), ids.len(), false);
    let _warm = generate_with_cache(
        &model,
        &ids,
        warm_cache.as_mut(),
        &config,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .map_err(|e| e.to_string())?;
    let warm_packed = model
        .packed_cache_evidence(warm_cache.as_ref())
        .ok_or("warm cache did not expose evidence")?;
    warm_cache.reset().map_err(|e| e.to_string())?;
    drop(warm_cache);

    let (fallback_route, mut fallback_cache) =
        packed_cache(&model, retained.clone(), ids.len(), true);
    let fallback_reason = match &fallback_route {
        CacheRoute::DenseFallback { reason } => reason.clone(),
        CacheRoute::ExperimentalPacked => {
            return Err("additive-mask fallback was accepted as packed".into())
        }
    };
    let fallback_before = model
        .packed_cache_evidence(fallback_cache.as_ref())
        .map(|evidence| evidence.dispatch_attempts)
        .unwrap_or(0);
    let fallback_out = generate_with_cache(
        &model,
        &ids,
        fallback_cache.as_mut(),
        &config,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .map_err(|e| e.to_string())?;
    let fallback_after = model
        .packed_cache_evidence(fallback_cache.as_ref())
        .map(|evidence| evidence.dispatch_attempts)
        .unwrap_or(0);
    fallback_cache.reset().map_err(|e| e.to_string())?;
    drop(fallback_cache);

    let (_, mut cancel_cache) = packed_cache(&model, retained.clone(), ids.len(), false);
    let cancel = CancelFlag::new();
    let mut emitted = 0_u64;
    let cancelled = generate_with_cache(
        &model,
        &ids,
        cancel_cache.as_mut(),
        &config,
        &cancel,
        &mut |event| {
            if matches!(event, StreamEvent::Token { .. }) {
                emitted += 1;
                cancel.cancel();
            }
        },
    )
    .map_err(|e| e.to_string())?;
    let cancel_evidence = model
        .packed_cache_evidence(cancel_cache.as_ref())
        .ok_or("cancel cache did not expose evidence")?;
    let before_reset = cancel_evidence.retained_device_packed_logical_bytes;
    cancel_cache.reset().map_err(|e| e.to_string())?;
    let after_reset = model
        .packed_cache_evidence(cancel_cache.as_ref())
        .ok_or("reset cache did not expose evidence")?
        .retained_device_packed_logical_bytes;
    drop(cancel_cache);
    cache.reset().map_err(|e| e.to_string())?;
    drop(cache);
    let timings = measured_packed_timings(&model, &ids, &config, &retained)?;
    drop(retained);
    mlx_rs::memory::clear_cache();
    let after = campaign::sample_memory(std::process::id()).map_err(|e| e.to_string())?;
    let dense_theoretical_kv_bytes = dense_kv_geometry(&cfg, ids.len())?;
    let theoretical_score_matrix_bytes = score_matrix_geometry(&cfg, ids.len())?;
    // The component evidence below is read from this same post-decode cache state. Using the
    // prefill boundary here would join different sequence lengths and could falsely validate only
    // because allocator slack hid the mismatch.
    let packed_device_bytes = active_resident_delta(&weights_loaded, &decode_peak);
    let transient_workspace_bytes = transient_peak_above_active(&decode_peak);
    if packed_device_bytes == 0 || transient_workspace_bytes == 0 {
        return Err("packed allocator evidence has no resident or transient delta".into());
    }
    let physical_release_verified = after.mlx_active_bytes == weights_loaded.mlx_active_bytes
        && after.mlx_cache_bytes == weights_loaded.mlx_cache_bytes
        && after.current_bytes
            <= weights_loaded
                .current_bytes
                .saturating_add(campaign::POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES);
    let final_snapshot_inventory = campaign::validate_sc20676_candidate_snapshot(family, snapshot)?;
    validate_snapshot_inventory_unchanged(&snapshot_inventory, &final_snapshot_inventory)?;
    finish_arm(Sc20676Arm {
        arm_sha256: String::new(),
        run_nonce: nonce.into(),
        worker_pid: std::process::id(),
        executable_sha256,
        family: family.into(),
        mode: mode.into(),
        snapshot_inventory_sha256: final_snapshot_inventory.sha256,
        prompt_tokens: ids.len() as u64,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        peak_mlx_bytes: [
            &before,
            &weights_loaded,
            &resident,
            &decode_window_start,
            &decode_peak,
            &after,
        ]
        .into_iter()
        .map(|sample| sample.mlx_peak_bytes)
        .max()
        .unwrap_or(0),
        peak_phys_footprint_bytes: [
            &before,
            &weights_loaded,
            &resident,
            &decode_window_start,
            &decode_peak,
            &after,
        ]
        .into_iter()
        .map(|sample| sample.peak_bytes)
        .max()
        .unwrap_or(0),
        provenance,
        input: input_binding(&ids),
        output,
        timings: timings.clone(),
        timing_summary: timing_summary(&timings)?,
        memory: Sc20676MemoryEvidence {
            phases: vec![
                phase("process-start", &before),
                phase("weights-loaded", &weights_loaded),
                phase("prefill-cache-resident", &resident),
                phase("decode-window-start", &decode_window_start),
                phase("decode-complete", &decode_peak),
                phase("reset-release", &after),
            ],
            decode_peak_window,
            dense_theoretical_kv_bytes,
            observed_dense_kv_bytes: dense_observed_kv_bytes,
            packed_logical_payload_bytes: packed.retained_device_code_bytes,
            packed_metadata_bytes: packed.retained_device_metadata_bytes,
            packed_device_bytes,
            transient_workspace_bytes,
            theoretical_dense_reconstruction_bytes: dense_theoretical_kv_bytes,
            theoretical_score_matrix_bytes,
            release_verified: physical_release_verified,
        },
        route: route_name(&route),
        kernel_profile: Some(kernel_profile),
        packed: Some(packed),
        warm_packed: Some(warm_packed),
        continuation_dispatches,
        quality: Some(quality),
        fallback: Some(Sc20676Fallback {
            route: route_name(&fallback_route),
            reason: fallback_reason,
            executed_dense_request: !fallback_out.tokens.is_empty(),
            packed_dispatch_attempts_before: fallback_before,
            packed_dispatch_attempts_after: fallback_after,
        }),
        cancellation: Some(Sc20676Cancellation {
            finish_reason: match cancelled.finish_reason {
                FinishReason::Cancelled => "cancelled".into(),
                _ => "unexpected".into(),
            },
            emitted_tokens: emitted,
            retained_before_reset: before_reset,
            retained_after_reset: after_reset,
            release_verified: after_reset == 0 && physical_release_verified,
            packed_before_reset: cancel_evidence,
        }),
        admission,
    })
}

fn required_flag(args: &[String], name: &str) -> std::result::Result<String, String> {
    args.windows(2)
        .find_map(|pair| (pair[0] == name).then(|| pair[1].clone()))
        .ok_or_else(|| format!("missing {name}"))
}

/// Validate every file the parent is about to publish.  The manifest is not trusted just because
/// it has a sidecar: it must name the two sealed receipts in this exact staging directory, bind
/// the current nonce/executable, and contain no extra row that could survive from another run.
fn validate_complete_matrix_directory(
    directory: &Path,
    nonce: &str,
    executable_sha256: &str,
) -> std::result::Result<(), String> {
    let names = fs::read_dir(directory)
        .map_err(|e| e.to_string())?
        .map(|entry| {
            entry
                .map_err(|e| e.to_string())
                .map(|entry| entry.file_name().to_string_lossy().into_owned())
        })
        .collect::<std::result::Result<std::collections::BTreeSet<_>, _>>()?;
    let expected = [
        "complete-matrix.json".to_owned(),
        "complete-matrix.json.sha256".to_owned(),
        "llama-receipt.json".to_owned(),
        "qwen-receipt.json".to_owned(),
    ]
    .into_iter()
    .collect::<std::collections::BTreeSet<_>>();
    if names != expected {
        return Err("SC-20676 staging directory is partial or contains stale artifacts".into());
    }
    let matrix_bytes =
        fs::read(directory.join("complete-matrix.json")).map_err(|e| e.to_string())?;
    let sidecar = fs::read_to_string(directory.join("complete-matrix.json.sha256"))
        .map_err(|e| e.to_string())?;
    if sidecar
        != format!(
            "{}  complete-matrix.json\n",
            campaign::seal_bytes(&matrix_bytes)
        )
    {
        return Err("SC-20676 complete matrix sidecar mismatch".into());
    }
    let matrix: serde_json::Value =
        serde_json::from_slice(&matrix_bytes).map_err(|e| e.to_string())?;
    if matrix
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(SC20676_SCHEMA_VERSION.into())
        || matrix.get("kind").and_then(serde_json::Value::as_str)
            != Some("sc-20676-complete-real-model-matrix")
        || matrix.get("runNonce").and_then(serde_json::Value::as_str) != Some(nonce)
        || matrix
            .get("executableSha256")
            .and_then(serde_json::Value::as_str)
            != Some(executable_sha256)
    {
        return Err("SC-20676 complete matrix identity is invalid".into());
    }
    let rows = matrix
        .get("receipts")
        .and_then(serde_json::Value::as_array)
        .ok_or("SC-20676 complete matrix lacks receipt rows")?;
    if rows.len() != 2 {
        return Err("SC-20676 complete matrix must contain Llama and Qwen only".into());
    }
    let mut seen = std::collections::BTreeSet::new();
    let mut validated_receipts = Vec::with_capacity(2);
    for row in rows {
        let family = row
            .get("family")
            .and_then(serde_json::Value::as_str)
            .ok_or("SC-20676 matrix row lacks family")?;
        if !["llama", "qwen"].contains(&family) || !seen.insert(family) {
            return Err("SC-20676 complete matrix has duplicate or unknown family".into());
        }
        let receipt_file_sha256 = row
            .get("receiptFileSha256")
            .and_then(serde_json::Value::as_str)
            .ok_or("SC-20676 matrix row lacks receipt-file seal")?;
        let receipt_bytes = fs::read(directory.join(format!("{family}-receipt.json")))
            .map_err(|e| e.to_string())?;
        let receipt = parse_staged_receipt_bytes(&receipt_bytes, receipt_file_sha256)?;
        if receipt.dense.family != family
            || receipt.dense.run_nonce != nonce
            || receipt.packed.run_nonce != nonce
            || receipt.dense.executable_sha256 != executable_sha256
            || receipt.packed.executable_sha256 != executable_sha256
            || row.get("receiptSha256").and_then(serde_json::Value::as_str)
                != Some(receipt.receipt_sha256.as_str())
            || row
                .get("baselineReceiptSha256")
                .and_then(serde_json::Value::as_str)
                != Some(receipt.baseline.receipt_sha256.as_str())
        {
            return Err("SC-20676 matrix row does not bind its staged receipt".into());
        }
        validated_receipts.push(receipt);
    }
    if seen.len() != 2 {
        return Err("SC-20676 complete matrix lacks a required family".into());
    }
    validate_complete_matrix_receipts(&validated_receipts)?;
    Ok(())
}

fn publish_complete_matrix(
    staging: &Path,
    destination: &Path,
    nonce: &str,
    executable_sha256: &str,
) -> std::result::Result<(), String> {
    if destination.exists() {
        return Err(
            "SC-20676 final destination already exists; refusing stale receipt reuse".into(),
        );
    }
    validate_complete_matrix_directory(staging, nonce, executable_sha256)?;
    fs::rename(staging, destination).map_err(|e| e.to_string())
}

fn parse_staged_receipt_bytes(
    bytes: &[u8],
    expected_file_sha256: &str,
) -> std::result::Result<Sc20676Receipt, String> {
    if !is_digest(expected_file_sha256) || campaign::seal_bytes(bytes) != expected_file_sha256 {
        return Err("SC-20676 staged receipt bytes do not match the complete-matrix seal".into());
    }
    let receipt = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    validate_sc20676_receipt(&receipt)?;
    Ok(receipt)
}

struct ArmFamilyInput {
    family: &'static str,
    snapshot: PathBuf,
    baseline: Sc20676BaselineBinding,
    snapshot_bytes: u64,
    baseline_manifest_sha256: String,
}

fn arm_resume_identity(
    inputs: &[ArmFamilyInput],
    executable_sha256: &str,
    policy_sha256: &str,
    provenance: &Sc20676Provenance,
    nonce: &str,
) -> std::result::Result<serde_json::Value, String> {
    let families = inputs
        .iter()
        .map(|input| {
            serde_json::json!({
                "family": input.family,
                "snapshotInventorySha256": input.baseline.snapshot_inventory_sha256,
                "snapshotBytes": input.snapshot_bytes,
                "baselineReceiptSha256": input.baseline.receipt_sha256,
                "baselineManifestSha256": input.baseline_manifest_sha256,
                "baseline": input.baseline,
            })
        })
        .collect::<Vec<_>>();
    Ok(serde_json::json!({
        "schemaVersion": 1,
        "kind": "sc-20676-arm-resume",
        "schedule": ["llama-dense", "llama-packed", "qwen-dense", "qwen-packed"],
        "executableSha256": executable_sha256,
        "policySha256": policy_sha256,
        "runNonce": nonce,
        "provenance": provenance,
        "families": families,
    }))
}

fn arm_identity_bytes(value: &serde_json::Value) -> std::result::Result<Vec<u8>, String> {
    campaign::canonical_json_bytes(value).map_err(|e| e.to_string())
}

fn prepare_arm_resume(
    root: &Path,
    identity: &mut serde_json::Value,
) -> std::result::Result<(String, String, Sc20676Provenance), String> {
    if root.exists() {
        let prior: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("identity.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        for key in ["sceneWorksRevision", "inferenceRevision"] {
            identity["provenance"][key] = prior
                .get("provenance")
                .and_then(|value| value.get(key))
                .cloned()
                .ok_or("SC-20676 resume identity lacks captured source revision")?;
        }
        identity["runNonce"] = prior
            .get("runNonce")
            .cloned()
            .ok_or("SC-20676 resume identity lacks run nonce")?;
    }
    let bytes = arm_identity_bytes(identity)?;
    let sha = campaign::seal_bytes(&bytes);
    let sidecar = format!("{sha}  identity.json\n");
    if root.exists() {
        if !root.is_dir()
            || fs::read(root.join("identity.json")).map_err(|e| e.to_string())? != bytes
            || fs::read_to_string(root.join("identity.json.sha256")).map_err(|e| e.to_string())?
                != sidecar
        {
            return Err("SC-20676 resume identity differs from executable, models, baseline, schedule, environment, or policy".into());
        }
    } else {
        fs::create_dir(root).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json"), bytes).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json.sha256"), sidecar).map_err(|e| e.to_string())?;
    }
    let nonce = identity
        .get("runNonce")
        .and_then(serde_json::Value::as_str)
        .filter(|nonce| is_digest(nonce))
        .ok_or("SC-20676 resume nonce invalid")?
        .to_owned();
    let provenance = serde_json::from_value(
        identity
            .get("provenance")
            .cloned()
            .ok_or("SC-20676 resume provenance missing")?,
    )
    .map_err(|e| e.to_string())?;
    Ok((sha, nonce, provenance))
}

fn arm_binding(identity_sha256: &str, arm: &Sc20676Arm, bytes: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1,
        "resumeIdentitySha256": identity_sha256,
        "armSha256": arm.arm_sha256,
        "fileSha256": campaign::seal_bytes(bytes),
    })
}

#[allow(clippy::too_many_arguments)] // A resumed arm must check every independent binding.
fn read_bound_arm(
    root: &Path,
    input: &ArmFamilyInput,
    mode: &str,
    identity_sha256: &str,
    nonce: &str,
    executable_sha256: &str,
    provenance: &Sc20676Provenance,
    seen_pids: &mut std::collections::BTreeSet<u32>,
) -> std::result::Result<Sc20676Arm, String> {
    let slug = format!("{}-{mode}", input.family);
    let bytes = fs::read(root.join(format!("{slug}.json"))).map_err(|e| e.to_string())?;
    let arm: Sc20676Arm = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_worker_binding(
        &arm,
        input.family,
        mode,
        nonce,
        executable_sha256,
        provenance,
        seen_pids,
    )?;
    if arm.snapshot_inventory_sha256 != input.baseline.snapshot_inventory_sha256 {
        return Err(format!(
            "SC-20676 resumed arm {slug} has stale snapshot inventory"
        ));
    }
    let expected = arm_identity_bytes(&arm_binding(identity_sha256, &arm, &bytes))?;
    if fs::read(root.join(format!("{slug}.binding.json"))).map_err(|e| e.to_string())? != expected {
        return Err(format!(
            "SC-20676 resumed arm {slug} has stale identity binding"
        ));
    }
    Ok(arm)
}

/// Parent spawns a fresh dense and packed child for each family.  Workers write untrusted arm
/// files; only this parent binds them to an already-sealed SC-20671 baseline and produces a seal.
pub fn sc20676_cli(args: &[String]) -> std::result::Result<(), String> {
    let mode = args
        .first()
        .map(String::as_str)
        .ok_or("usage: sc20676-packed-evidence parent|worker ...")?;
    if mode == "worker" {
        let policy = campaign::load_campaign_safety_policy(Path::new(&required_flag(
            args,
            "--safety-policy",
        )?))?;
        if policy.seal()? != required_flag(args, "--policy-sha256")? {
            return Err("SC-20676 worker safety policy seal mismatch".into());
        }
        let requested_tokens: u64 = required_flag(args, "--context-payload-tokens")?
            .parse()
            .map_err(|_| "invalid --context-payload-tokens")?;
        if requested_tokens == 0 {
            return Err("SC-20676 requested context is zero".into());
        }
        sc20676_admitted_tokens(&required_flag(args, "--mode")?, requested_tokens, &policy)?;
        let identity_bytes = fs::read(required_flag(args, "--resume-identity")?)
            .map_err(|e| format!("read SC-20676 parent identity: {e}"))?;
        if campaign::seal_bytes(&identity_bytes) != required_flag(args, "--resume-identity-sha256")?
        {
            return Err("SC-20676 worker resume identity seal mismatch".into());
        }
        let identity: serde_json::Value =
            serde_json::from_slice(&identity_bytes).map_err(|e| e.to_string())?;
        if identity
            .get("policySha256")
            .and_then(serde_json::Value::as_str)
            != Some(policy.seal()?.as_str())
            || identity
                .get("executableSha256")
                .and_then(serde_json::Value::as_str)
                != Some(required_flag(args, "--executable-sha256")?.as_str())
            || identity.get("runNonce").and_then(serde_json::Value::as_str)
                != Some(required_flag(args, "--run-nonce")?.as_str())
        {
            return Err(
                "SC-20676 worker policy, executable, or nonce differs from resume identity".into(),
            );
        }
        let captured: Sc20676Provenance = serde_json::from_value(
            identity
                .get("provenance")
                .cloned()
                .ok_or("SC-20676 resume provenance missing")?,
        )
        .map_err(|e| e.to_string())?;
        let mut current = worker_provenance()?;
        current
            .scene_works_revision
            .clone_from(&captured.scene_works_revision);
        current
            .inference_revision
            .clone_from(&captured.inference_revision);
        if current != captured {
            return Err(
                "SC-20676 worker environment differs from captured resume provenance".into(),
            );
        }
        let arm = run_sc20676_worker(
            Path::new(&required_flag(args, "--snapshot")?),
            &required_flag(args, "--family")?,
            &required_flag(args, "--mode")?,
            &required_flag(args, "--run-nonce")?,
            &required_flag(args, "--executable-sha256")?,
            requested_tokens,
            &policy,
            &captured,
        )?;
        validate_arm(&arm)?;
        let output = PathBuf::from(required_flag(args, "--out")?);
        if output.exists() {
            return Err("worker output path already exists".into());
        }
        let staging = output.with_extension(format!("json.staging-{}", std::process::id()));
        if staging.exists() {
            return Err("SC-20676 worker staging path already exists".into());
        }
        fs::write(
            &staging,
            campaign::canonical_json_bytes(&serde_json::to_value(arm).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::rename(staging, output).map_err(|e| e.to_string())?;
        return Ok(());
    }
    if mode != "parent" {
        return Err("usage: sc20676-packed-evidence parent|worker ...".into());
    }
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let policy_path = PathBuf::from(required_flag(args, "--safety-policy")?);
    let policy = campaign::load_campaign_safety_policy(&policy_path)?;
    let policy_sha256 = policy.seal()?;
    let executable_sha256 = file_sha256(&executable)?;
    let proposed_nonce = run_nonce(&executable_sha256)?;
    let current_provenance = worker_provenance()?;
    let destination = PathBuf::from(required_flag(args, "--out")?);
    let worker_root = PathBuf::from(required_flag(args, "--resume-dir")?);
    if !destination.is_absolute() || !worker_root.is_absolute() || destination == worker_root {
        return Err("SC-20676 destination and distinct resume directory must be absolute".into());
    }
    if destination.exists() {
        return Err(
            "SC-20676 final destination already exists; refusing stale receipt reuse".into(),
        );
    }
    let parent = destination.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let mut inputs = Vec::new();
    for family in ["llama", "qwen"] {
        let snapshot = PathBuf::from(required_flag(args, &format!("--{family}-snapshot"))?);
        let baseline_campaign = PathBuf::from(required_flag(
            args,
            &format!("--{family}-baseline-campaign"),
        )?);
        let baseline = bind_sc20671_baseline(&baseline_campaign, &snapshot, family)?;
        let inventory = campaign::validate_sc20676_candidate_snapshot(family, &snapshot)?;
        let native = campaign::benchmark_model(family, false)?.native_context_tokens;
        if baseline
            .context_payload_tokens
            .checked_add(16)
            .is_none_or(|tokens| tokens > native)
        {
            return Err(format!(
                "SC-20676 {family} baseline plus generation exceeds native context"
            ));
        }
        inputs.push(ArmFamilyInput {
            family,
            snapshot,
            baseline,
            snapshot_bytes: inventory.bytes,
            baseline_manifest_sha256: file_sha256(&baseline_campaign.join("campaign.json"))?,
        });
    }
    let mut identity = arm_resume_identity(
        &inputs,
        &executable_sha256,
        &policy_sha256,
        &current_provenance,
        &proposed_nonce,
    )?;
    let (identity_sha256, nonce, expected_provenance) =
        prepare_arm_resume(&worker_root, &mut identity)?;
    let staging = parent.join(format!(
        ".{}.sc20676-staging-{nonce}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("evidence")
    ));
    if staging.exists() {
        return Err(
            "SC-20676 partial final staging exists; remove it explicitly before resume".into(),
        );
    }
    let allowed = ["llama-dense", "llama-packed", "qwen-dense", "qwen-packed"];
    for entry in fs::read_dir(&worker_root).map_err(|e| e.to_string())? {
        let name = entry
            .map_err(|e| e.to_string())?
            .file_name()
            .to_string_lossy()
            .to_string();
        if !["identity.json", "identity.json.sha256", "logs"].contains(&name.as_str())
            && !allowed.iter().any(|slug| {
                name == format!("{slug}.json") || name == format!("{slug}.binding.json")
            })
        {
            return Err(format!(
                "SC-20676 unexpected or partial resume artifact: {name}"
            ));
        }
    }
    let result = (|| -> std::result::Result<(), String> {
        let mut receipts = Vec::new();
        let mut worker_pids = std::collections::BTreeSet::new();
        for input in &inputs {
            let family = input.family;
            let target_prompt_tokens = input.baseline.context_payload_tokens.to_string();
            let mut arms = Vec::new();
            for worker_mode in ["dense", "packed"] {
                let slug = format!("{family}-{worker_mode}");
                let arm_path = worker_root.join(format!("{slug}.json"));
                let binding_path = worker_root.join(format!("{slug}.binding.json"));
                if arm_path.exists() || binding_path.exists() {
                    if !arm_path.exists() || !binding_path.exists() {
                        return Err(format!(
                            "SC-20676 partial resume arm {slug} must be repaired explicitly"
                        ));
                    }
                    arms.push(read_bound_arm(
                        &worker_root,
                        input,
                        worker_mode,
                        &identity_sha256,
                        &nonce,
                        &executable_sha256,
                        &expected_provenance,
                        &mut worker_pids,
                    )?);
                    eprintln!("SC-20676 resumed validated arm {slug}");
                    continue;
                }
                let (total_tokens, request_tokens) = sc20676_admitted_tokens(
                    worker_mode,
                    input.baseline.context_payload_tokens,
                    &policy,
                )?;
                let admission = sc20676_runtime_admission(
                    family,
                    &input.snapshot,
                    total_tokens,
                    request_tokens,
                    &policy,
                )?;
                let mut command = Command::new(&executable);
                command
                    .arg("worker")
                    .arg("--snapshot")
                    .arg(&input.snapshot)
                    .arg("--family")
                    .arg(family)
                    .arg("--mode")
                    .arg(worker_mode)
                    .arg("--run-nonce")
                    .arg(&nonce)
                    .arg("--executable-sha256")
                    .arg(&executable_sha256)
                    .arg("--context-payload-tokens")
                    .arg(&target_prompt_tokens)
                    .arg("--out")
                    .arg(&arm_path)
                    .arg("--safety-policy")
                    .arg(&policy_path)
                    .arg("--policy-sha256")
                    .arg(&policy_sha256)
                    .arg("--resume-identity")
                    .arg(worker_root.join("identity.json"))
                    .arg("--resume-identity-sha256")
                    .arg(&identity_sha256);
                let logs = worker_root.join("logs");
                let log_prefix = (0_u64..)
                    .map(|attempt| format!("{slug}.attempt-{attempt}"))
                    .find(|prefix| {
                        !logs.join(format!("{prefix}.stdout.log")).exists()
                            && !logs.join(format!("{prefix}.stderr.log")).exists()
                    })
                    .ok_or("SC-20676 no unused bounded worker log path")?;
                let request = RunRequest {
                    context_tokens: total_tokens,
                    request_tokens,
                    stdout_path: logs.join(format!("{log_prefix}.stdout.log")),
                    stderr_path: logs.join(format!("{log_prefix}.stderr.log")),
                };
                let unaccepted = logs.join(format!("{log_prefix}.unaccepted.json"));
                let status = match campaign_supervisor::run_guarded(
                    &mut command,
                    &request,
                    &policy.supervisor(),
                    &mut SystemProbe,
                ) {
                    Ok(status) => status,
                    Err(failure) => {
                        let reason = format!("{:?}", failure.reason);
                        campaign::write_unaccepted_row_record(
                            &unaccepted,
                            "sc-20676-unaccepted-arm",
                            &slug,
                            &admission,
                            (&reason, &failure.detail, failure.pid),
                        )?;
                        return Err(format!(
                            "SC-20676 {slug} worker stopped ({reason}): {}; child {:?} reaped; stderr {}; not accepted ({}); valid arms remain in {}",
                            failure.detail, failure.pid, request.stderr_path.display(), unaccepted.display(), worker_root.display(),
                        ));
                    }
                };
                if !status.success() {
                    campaign::write_unaccepted_row_record(
                        &unaccepted,
                        "sc-20676-unaccepted-arm",
                        &slug,
                        &admission,
                        ("ChildExit", &status.to_string(), None),
                    )?;
                    return Err(format!(
                        "SC-20676 {slug} worker failed with {status}; stderr {}; not accepted ({}); valid arms remain in {}",
                        request.stderr_path.display(), unaccepted.display(), worker_root.display(),
                    ));
                }
                let bytes = fs::read(&arm_path).map_err(|e| e.to_string())?;
                let arm: Sc20676Arm = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
                let mut candidate_pids = worker_pids.clone();
                validate_worker_binding(
                    &arm,
                    family,
                    worker_mode,
                    &nonce,
                    &executable_sha256,
                    &expected_provenance,
                    &mut candidate_pids,
                )?;
                if arm.snapshot_inventory_sha256 != input.baseline.snapshot_inventory_sha256 {
                    return Err(format!("SC-20676 {slug} snapshot inventory changed"));
                }
                let binding = arm_identity_bytes(&arm_binding(&identity_sha256, &arm, &bytes))?;
                let staging_binding =
                    binding_path.with_extension(format!("json.staging-{}", std::process::id()));
                if staging_binding.exists() {
                    return Err(format!("SC-20676 partial binding staging for {slug}"));
                }
                fs::write(&staging_binding, binding).map_err(|e| e.to_string())?;
                fs::rename(staging_binding, &binding_path).map_err(|e| e.to_string())?;
                arms.push(read_bound_arm(
                    &worker_root,
                    input,
                    worker_mode,
                    &identity_sha256,
                    &nonce,
                    &executable_sha256,
                    &expected_provenance,
                    &mut worker_pids,
                )?);
                eprintln!("SC-20676 accepted validated arm {slug}");
            }
            let (dense, packed) = take_validated_family_arms(arms)?;
            let receipt = Sc20676Receipt {
                schema_version: 0,
                harness_version: String::new(),
                receipt_sha256: String::new(),
                baseline: input.baseline.clone(),
                thresholds: Sc20676Thresholds::default(),
                dense,
                packed,
            }
            .finish()?;
            receipts.push(receipt);
        }
        validate_complete_matrix_receipts(&receipts)?;
        fs::create_dir(&staging).map_err(|e| e.to_string())?;
        let mut receipt_file_sha256 = std::collections::BTreeMap::new();
        for receipt in &receipts {
            let bytes = receipt.bytes().map_err(|e| e.to_string())?;
            receipt_file_sha256.insert(receipt.dense.family.clone(), campaign::seal_bytes(&bytes));
            fs::write(
                staging.join(format!("{}-receipt.json", receipt.dense.family)),
                bytes,
            )
            .map_err(|e| e.to_string())?;
        }
        let mut matrix_rows = Vec::with_capacity(receipts.len());
        for receipt in &receipts {
            let file_sha256 = receipt_file_sha256
                .get(&receipt.dense.family)
                .ok_or("SC-20676 staged receipt file seal is missing")?;
            matrix_rows.push(serde_json::json!({"family": receipt.dense.family, "receiptSha256": receipt.receipt_sha256, "receiptFileSha256": file_sha256, "baselineReceiptSha256": receipt.baseline.receipt_sha256}));
        }
        let matrix = serde_json::json!({ "schemaVersion": SC20676_SCHEMA_VERSION, "kind": "sc-20676-complete-real-model-matrix", "runNonce": nonce, "executableSha256": executable_sha256, "receipts": matrix_rows });
        let matrix_bytes = campaign::canonical_json_bytes(&matrix).map_err(|e| e.to_string())?;
        fs::write(staging.join("complete-matrix.json"), &matrix_bytes)
            .map_err(|e| e.to_string())?;
        fs::write(
            staging.join("complete-matrix.json.sha256"),
            format!(
                "{}  complete-matrix.json\n",
                campaign::seal_bytes(&matrix_bytes)
            ),
        )
        .map_err(|e| e.to_string())?;
        publish_complete_matrix(&staging, &destination, &nonce, &executable_sha256)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest() -> String {
        "a".repeat(64)
    }
    fn inventory(sha256: &str, bytes: u64) -> campaign::SnapshotInventory {
        campaign::SnapshotInventory {
            root: std::path::PathBuf::new(),
            files: Vec::new(),
            bytes,
            sha256: sha256.into(),
        }
    }
    fn provenance() -> Sc20676Provenance {
        Sc20676Provenance {
            scene_works_repository: campaign::SCENEWORKS_REPOSITORY.into(),
            scene_works_revision: "b".repeat(40),
            inference_repository: campaign::INFERENCE_REPOSITORY.into(),
            inference_revision: "b".repeat(40),
            dependency_lock_sha256: digest(),
            mlx_version: "0.1".into(),
            mlx_source: "git".into(),
            mlx_revision: "b".repeat(40),
            os: "macos".into(),
            os_version: "15.0".into(),
            xcode: "Xcode".into(),
            hardware: "Mac".into(),
            metal_device: "Apple GPU".into(),
            power_mode: "nominal".into(),
            thermal_state: "nominal".into(),
            source_tree_clean: true,
        }
    }
    fn timing(packed: bool) -> Vec<Sc20676TimingSample> {
        vec![
            Sc20676TimingSample {
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 2.0,
                steady_decode_tokens_per_second: 10.0,
                packed_cold_dispatch_ms: 0.0,
                packed_warm_dispatch_ms: if packed { 1.0 } else { 0.0 }
            };
            5
        ]
    }
    fn memory() -> Sc20676MemoryEvidence {
        let sample = |index: usize, phase: &str| Sc20676MemoryPhase {
            phase: phase.into(),
            captured_at: format!("2026-01-01T00:00:{index:02}Z"),
            phys_footprint_bytes: 10,
            phys_footprint_peak_bytes: 10,
            mlx_active_bytes: 2,
            mlx_cache_bytes: 3,
            mlx_peak_bytes: if ["weights-loaded", "decode-window-start"].contains(&phase) {
                0
            } else {
                4
            },
        };
        Sc20676MemoryEvidence {
            phases: [
                "process-start",
                "weights-loaded",
                "prefill-cache-resident",
                "decode-window-start",
                "decode-complete",
                "reset-release",
            ]
            .into_iter()
            .enumerate()
            .map(|(index, phase)| sample(index, phase))
            .collect(),
            decode_peak_window: Sc20676PeakWindow {
                started_at: "2026-01-01T00:00:03Z".into(),
                baseline_active_bytes: 2,
                baseline_cache_bytes: 3,
                reset_peak_bytes: 0,
            },
            dense_theoretical_kv_bytes: 2,
            observed_dense_kv_bytes: 1,
            packed_logical_payload_bytes: 1,
            packed_metadata_bytes: 1,
            packed_device_bytes: 0,
            transient_workspace_bytes: 2,
            theoretical_dense_reconstruction_bytes: 2,
            theoretical_score_matrix_bytes: 1,
            release_verified: true,
        }
    }
    fn packed_evidence() -> PackedCacheEvidence {
        PackedCacheEvidence {
            representation_identity: "sc-20676-packed-group-affine-v1".into(),
            representation_version: 2,
            bits: 2,
            quantization_group_size: 32,
            accepted_direct_calls: 2,
            compile_jit_attempts: 1,
            kernel_warmed: true,
            cold_dispatches: 1,
            steady_dispatches: 1,
            cold_elapsed_ms: 1.0,
            steady_elapsed_ms: 1.0,
            retained_device_code_bytes: 1,
            retained_device_metadata_bytes: 1,
            retained_device_packed_logical_bytes: 2,
            peak_packed_argument_logical_bytes: 1,
            peak_packed_transient_logical_bytes: 1,
            ..Default::default()
        }
    }
    fn arm(mode: &str) -> Sc20676Arm {
        let packed = mode == "packed";
        let input_tokens = vec![1_i32; 1024];
        let output_tokens = vec![1_i32];
        let logits = vec![1_f32];
        let timings = timing(packed);
        let mut arm_memory = memory();
        if packed {
            arm_memory.phases[2].mlx_active_bytes = 3;
            arm_memory.phases[2].mlx_peak_bytes = 6;
            arm_memory.phases[3].mlx_active_bytes = 3;
            arm_memory.phases[3].mlx_peak_bytes = 0;
            arm_memory.phases[4].mlx_active_bytes = 4;
            arm_memory.phases[4].mlx_peak_bytes = 6;
            arm_memory.phases[5].mlx_peak_bytes = 6;
            arm_memory.decode_peak_window.baseline_active_bytes = 3;
            arm_memory.packed_device_bytes = 2;
        } else {
            arm_memory.packed_logical_payload_bytes = 0;
            arm_memory.packed_metadata_bytes = 0;
            arm_memory.packed_device_bytes = 0;
        }
        let mut arm = Sc20676Arm {
            arm_sha256: String::new(),
            run_nonce: digest(),
            worker_pid: if packed { 2 } else { 1 },
            executable_sha256: digest(),
            family: "llama".into(),
            mode: mode.into(),
            snapshot_inventory_sha256: digest(),
            prompt_tokens: 1024,
            elapsed_ms: 1.0,
            peak_mlx_bytes: if packed { 6 } else { 4 },
            peak_phys_footprint_bytes: 10,
            provenance: provenance(),
            input: input_binding(&input_tokens),
            output: output_binding(
                &logits,
                &[1, 1],
                "Float32",
                &output_tokens,
                SC20676_NEEDLE,
                SC20676_NEEDLE.into(),
            ),
            timings: timings.clone(),
            timing_summary: timing_summary(&timings).unwrap(),
            memory: arm_memory,
            route: if packed {
                "experimental-packed"
            } else {
                "dense"
            }
            .into(),
            kernel_profile: packed.then_some(Sc20676KernelProfile {
                metal_device: "Apple GPU".into(),
                gpu_family: "conservative-unknown-apple".into(),
                qualification: "conservative-default".into(),
                head_dimension: 64,
                threads: 32,
                simd_groups: 1,
                values_per_thread: 2,
            }),
            packed: packed.then(packed_evidence),
            warm_packed: packed.then_some(PackedCacheEvidence {
                accepted_direct_calls: 2,
                kernel_warmed: true,
                steady_dispatches: 1,
                steady_elapsed_ms: 1.0,
                ..Default::default()
            }),
            continuation_dispatches: u64::from(packed),
            quality: packed.then_some(Sc20676Quality {
                max_logit_abs_error: 0.0,
                greedy_token_agreement: 1.0,
                needle_retrieval: true,
                needle_discriminating: true,
            }),
            fallback: packed.then_some(Sc20676Fallback {
                route: "dense-fallback".into(),
                reason: "additive attention masks require dense fallback".into(),
                executed_dense_request: true,
                packed_dispatch_attempts_before: 0,
                packed_dispatch_attempts_after: 0,
            }),
            cancellation: packed.then_some(Sc20676Cancellation {
                finish_reason: "cancelled".into(),
                emitted_tokens: 1,
                retained_before_reset: 1,
                retained_after_reset: 0,
                release_verified: true,
                packed_before_reset: packed_evidence(),
            }),
            admission: campaign::ReceiptAdmission {
                mode: campaign::RUNTIME_GUARDED_ADMISSION.into(),
                child_footprint_cap_bytes: 1 << 30,
                host_free_reserve_bytes: 1 << 30,
                static_footprint_floor_bytes: 1 << 20,
            },
        };
        arm.arm_sha256 = arm_semantic_seal(&arm).unwrap();
        arm
    }
    fn reseal_arm(arm: &mut Sc20676Arm) {
        arm.arm_sha256 = arm_semantic_seal(arm).unwrap();
    }
    fn receipt() -> Sc20676Receipt {
        let model = campaign::benchmark_model("llama", false).unwrap();
        let inventory_sha256 = digest();
        Sc20676Receipt {
            schema_version: SC20676_SCHEMA_VERSION,
            harness_version: SC20676_HARNESS_VERSION.into(),
            receipt_sha256: String::new(),
            baseline: Sc20676BaselineBinding {
                receipt_sha256: digest(),
                campaign_manifest_sha256: digest(),
                campaign_schedule_version: 2,
                model_file_sha256: digest(),
                model_id: format!(
                    "{}@{};architecture={};inventory={}",
                    model.repository, model.revision, model.architecture, inventory_sha256
                ),
                model_repository: model.repository.into(),
                model_revision: model.revision.into(),
                snapshot_inventory_sha256: inventory_sha256,
                scene_works_revision: "b".repeat(40),
                inference_revision: "b".repeat(40),
                campaign_session_id: digest(),
                campaign_global_identity_sha256: digest(),
                context_band: campaign::SC20676_BASELINE_CONTEXT_BAND.into(),
                context_payload_tokens: 1024,
                head_dimension: 64,
            },
            thresholds: Sc20676Thresholds::default(),
            dense: arm("dense"),
            packed: arm("packed"),
        }
    }
    #[test]
    fn seal_nonce_pid_and_input_tampering_fail_closed() {
        let sealed = receipt().finish().unwrap();
        validate_sc20676_receipt(&sealed).unwrap();
        for mutate in [0, 1, 2] {
            let mut bad = sealed.clone();
            if mutate == 0 {
                bad.packed.run_nonce = "c".repeat(64);
            }
            if mutate == 1 {
                bad.packed.worker_pid = bad.dense.worker_pid;
            }
            if mutate == 2 {
                bad.packed.input.ids_len += 1;
            }
            reseal_arm(&mut bad.packed);
            assert!(validate_sc20676_receipt(&bad).is_err());
        }
    }
    #[test]
    fn worker_binding_rejects_stale_or_reused_rows() {
        let dense = arm("dense");
        let mut pids = std::collections::BTreeSet::new();
        validate_worker_binding(
            &dense,
            "llama",
            "dense",
            &dense.run_nonce,
            &dense.executable_sha256,
            &dense.provenance,
            &mut pids,
        )
        .unwrap();
        assert!(validate_worker_binding(
            &dense,
            "llama",
            "dense",
            &dense.run_nonce,
            &dense.executable_sha256,
            &dense.provenance,
            &mut pids
        )
        .is_err());
        let mut stale = arm("dense");
        stale.run_nonce = "c".repeat(64);
        reseal_arm(&mut stale);
        assert!(validate_worker_binding(
            &stale,
            "llama",
            "dense",
            &dense.run_nonce,
            &dense.executable_sha256,
            &dense.provenance,
            &mut std::collections::BTreeSet::new()
        )
        .is_err());
    }
    #[test]
    fn arm_pair_and_cross_arm_seals_reject_replacement() {
        let dense = arm("dense");
        let packed = arm("packed");
        assert!(take_validated_family_arms(vec![dense.clone(), packed.clone()]).is_ok());
        assert!(take_validated_family_arms(vec![dense.clone(), dense]).is_err());
        let mut bad = receipt();
        bad.packed.run_nonce = "c".repeat(64);
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed.executable_sha256 = "c".repeat(64);
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn representation_fallback_cancellation_and_thresholds_fail_closed() {
        let mut bad = receipt();
        bad.packed.packed.as_mut().unwrap().bits = 0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed.fallback.as_mut().unwrap().reason = "late".into();
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed
            .cancellation
            .as_mut()
            .unwrap()
            .retained_after_reset = 1;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.thresholds.contract_hash = digest();
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn timing_memory_output_and_baseline_tampering_fail_closed() {
        let mut bad = receipt();
        bad.packed.timings[0].steady_decode_tokens_per_second = 100.0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed.memory.phases.pop();
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed.output.logits_len = 0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt().finish().unwrap();
        bad.baseline.receipt_sha256 = "c".repeat(64);
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn non_finite_logits_never_reduce_to_a_passing_parity_scalar() {
        assert_eq!(max_abs(&[1.0, -2.0], &[1.5, -1.0]).unwrap(), 1.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(max_abs(&[value], &[0.0]).is_err());
            assert!(max_abs(&[0.0], &[value]).is_err());
        }
        assert!(max_abs(&[], &[]).is_err());
        assert!(max_abs(&[0.0], &[0.0, 1.0]).is_err());
    }
    #[test]
    fn baseline_source_and_complete_campaign_identity_fail_closed() {
        let mut stale_inference = receipt();
        stale_inference.baseline.inference_revision = "c".repeat(40);
        assert!(validate_sc20676_receipt_core(&stale_inference).is_err());

        let mut stale_scene_works = receipt();
        stale_scene_works.baseline.scene_works_revision = "c".repeat(40);
        assert!(validate_sc20676_receipt_core(&stale_scene_works).is_err());

        let llama = receipt().finish().unwrap();
        let mut qwen = receipt();
        let model = campaign::benchmark_model("qwen", false).unwrap();
        qwen.baseline.model_repository = model.repository.into();
        qwen.baseline.model_revision = model.revision.into();
        qwen.baseline.model_id = format!(
            "{}@{};architecture={};inventory={}",
            model.repository,
            model.revision,
            model.architecture,
            qwen.baseline.snapshot_inventory_sha256
        );
        // Each sealed SC-20671 family row ran in a distinct product child/session.
        qwen.baseline.campaign_session_id = "d".repeat(64);
        qwen.dense.family = "qwen".into();
        qwen.packed.family = "qwen".into();
        reseal_arm(&mut qwen.dense);
        reseal_arm(&mut qwen.packed);
        let qwen = qwen.finish().unwrap();
        assert!(validate_complete_matrix_receipts(&[llama.clone(), qwen.clone()]).is_ok());

        let mut mixed_campaign = qwen;
        mixed_campaign.baseline.campaign_schedule_version = 1;
        let mixed_campaign = mixed_campaign.finish().unwrap();
        assert!(validate_complete_matrix_receipts(&[llama, mixed_campaign]).is_err());
    }
    #[test]
    fn packed_allocator_residency_uses_the_final_decode_cache_boundary() {
        let valid = receipt();
        assert_ne!(
            valid.packed.memory.phases[2].mlx_active_bytes,
            valid.packed.memory.phases[4].mlx_active_bytes
        );
        validate_sc20676_receipt_core(&valid).unwrap();

        let mut prefill_join = valid;
        prefill_join.packed.memory.packed_device_bytes = prefill_join.packed.memory.phases[2]
            .mlx_active_bytes
            .saturating_sub(prefill_join.packed.memory.phases[1].mlx_active_bytes);
        reseal_arm(&mut prefill_join.packed);
        assert!(validate_sc20676_receipt_core(&prefill_join).is_err());
    }
    #[test]
    fn partial_staging_never_publishes_a_final_matrix() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let staging = root.join("staging");
        let destination = root.join("published");
        fs::create_dir_all(&staging).unwrap();
        fs::write(staging.join("complete-matrix.json"), b"{}").unwrap();
        assert!(publish_complete_matrix(&staging, &destination, &digest(), &digest()).is_err());
        assert!(!destination.exists());
    }
    #[test]
    fn arm_policy_counts_generated_tokens_and_retained_packed_cache() {
        let mut policy = campaign::CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1,
            child_footprint_cap_bytes: 2,
            max_context_tokens: 1040,
            max_request_tokens: 1040,
            stdout_cap_bytes: 100,
            stderr_cap_bytes: 100,
        };
        assert_eq!(
            sc20676_admitted_tokens("dense", 1024, &policy).unwrap(),
            (1040, 1040)
        );
        assert!(sc20676_admitted_tokens("packed", 1024, &policy).is_err());
        policy.max_context_tokens = 2080;
        assert_eq!(
            sc20676_admitted_tokens("packed", 1024, &policy).unwrap(),
            (2080, 1040)
        );
        assert!(sc20676_admitted_tokens("packed", u64::MAX, &policy).is_err());
        // No unconditional long-context refusal: the arm reaches the pinned snapshot's load-plus-KV
        // floor, and admission is otherwise the runtime guards.
        let error = sc20676_runtime_admission(
            "llama",
            Path::new("/nonexistent-snapshot"),
            2080,
            1040,
            &policy,
        )
        .unwrap_err();
        assert!(!error.contains("prefill peak bound"), "{error}");
        let temporary = tempfile::tempdir().unwrap();
        let snapshot = temporary.path().join("llama");
        campaign::tests::write_stub_dtype_snapshot(
            &snapshot,
            campaign::benchmark_model("llama", false).unwrap(),
            "F16",
        );
        policy.child_footprint_cap_bytes = 64 << 30;
        let admission = sc20676_runtime_admission("llama", &snapshot, 2080, 1040, &policy).unwrap();
        assert_eq!(admission.mode, campaign::RUNTIME_GUARDED_ADMISSION);
        assert_eq!(admission.child_footprint_cap_bytes, 64 << 30);
        assert!(admission.static_footprint_floor_bytes > 0);
        policy.child_footprint_cap_bytes = admission.static_footprint_floor_bytes - 1;
        assert!(sc20676_runtime_admission("llama", &snapshot, 2080, 1040, &policy).is_err());
        policy.child_footprint_cap_bytes = 64 << 30;
        policy.host_free_reserve_bytes = 0;
        assert!(sc20676_runtime_admission("llama", &snapshot, 2080, 1040, &policy).is_err());
    }

    #[test]
    fn packed_needle_is_gated_against_the_same_weights_dense_output() {
        assert_eq!(
            sc20676_needle_observation(SC20676_NEEDLE, SC20676_NEEDLE),
            (true, true)
        );
        assert_eq!(
            sc20676_needle_observation(SC20676_NEEDLE, "x"),
            (false, true)
        );
        assert_eq!(
            sc20676_needle_observation("I cannot help.", "I cannot help."),
            (true, false)
        );
        assert_eq!(
            sc20676_needle_observation("I cannot help.", "other"),
            (false, false)
        );
        // A dense arm that missed the needle is valid evidence, and the packed arm's identical
        // output is accepted only when flagged non-discriminating.
        let mut shared_miss = receipt();
        for arm in [&mut shared_miss.dense, &mut shared_miss.packed] {
            arm.output.needle_output = "I cannot help.".into();
        }
        let quality = shared_miss.packed.quality.as_mut().unwrap();
        quality.needle_discriminating = false;
        reseal_arm(&mut shared_miss.dense);
        reseal_arm(&mut shared_miss.packed);
        validate_arm(&shared_miss.dense).unwrap();
        let flagged = shared_miss.clone().finish().unwrap();
        assert!(
            !flagged
                .packed
                .quality
                .as_ref()
                .unwrap()
                .needle_discriminating
        );
        let mut hidden = shared_miss;
        hidden
            .packed
            .quality
            .as_mut()
            .unwrap()
            .needle_discriminating = true;
        reseal_arm(&mut hidden.packed);
        assert!(hidden.finish().is_err());
    }

    #[test]
    fn sealed_arm_resume_rejects_partial_stale_and_corrupt_rows() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("resume");
        let input = ArmFamilyInput {
            family: "llama",
            snapshot: PathBuf::new(),
            baseline: receipt().baseline,
            snapshot_bytes: 100,
            baseline_manifest_sha256: digest(),
        };
        let mut identity =
            arm_resume_identity(&[input], &digest(), &digest(), &provenance(), &digest()).unwrap();
        let (sha, nonce, captured) = prepare_arm_resume(&root, &mut identity).unwrap();
        let mut moved_ref = identity.clone();
        moved_ref["provenance"]["inferenceRevision"] = serde_json::json!("c".repeat(40));
        assert_eq!(prepare_arm_resume(&root, &mut moved_ref).unwrap().0, sha);
        let mut changed_exe = identity.clone();
        changed_exe["executableSha256"] = serde_json::json!("d".repeat(64));
        assert!(prepare_arm_resume(&root, &mut changed_exe).is_err());
        let input = ArmFamilyInput {
            family: "llama",
            snapshot: PathBuf::new(),
            baseline: receipt().baseline,
            snapshot_bytes: 100,
            baseline_manifest_sha256: digest(),
        };
        let arm = arm("dense");
        let bytes = campaign::canonical_json_bytes(&serde_json::to_value(&arm).unwrap()).unwrap();
        fs::write(root.join("llama-dense.json"), &bytes).unwrap();
        assert!(read_bound_arm(
            &root,
            &input,
            "dense",
            &sha,
            &nonce,
            &digest(),
            &captured,
            &mut std::collections::BTreeSet::new()
        )
        .is_err());
        let binding = arm_identity_bytes(&arm_binding(&sha, &arm, &bytes)).unwrap();
        fs::write(root.join("llama-dense.binding.json"), binding).unwrap();
        assert!(read_bound_arm(
            &root,
            &input,
            "dense",
            &sha,
            &nonce,
            &digest(),
            &captured,
            &mut std::collections::BTreeSet::new()
        )
        .is_ok());
        fs::write(root.join("llama-dense.binding.json"), b"corrupt").unwrap();
        assert!(read_bound_arm(
            &root,
            &input,
            "dense",
            &sha,
            &nonce,
            &digest(),
            &captured,
            &mut std::collections::BTreeSet::new()
        )
        .is_err());
    }
    #[test]
    fn staged_receipt_parser_requires_the_semantic_core_seal() {
        let unsealed = receipt();
        let bytes = unsealed.bytes().unwrap();
        let file_sha256 = campaign::seal_bytes(&bytes);
        assert!(parse_staged_receipt_bytes(&bytes, &file_sha256).is_err());

        let sealed = receipt().finish().unwrap();
        let bytes = sealed.bytes().unwrap();
        let file_sha256 = campaign::seal_bytes(&bytes);
        assert!(parse_staged_receipt_bytes(&bytes, &file_sha256).is_ok());
    }
    #[test]
    fn output_hash_and_frozen_threshold_tampering_fail_closed() {
        let mut bad = receipt();
        bad.packed.output.tokens_sha256 = digest();
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.thresholds.max_logit_abs_error = 0.1;
        assert!(validate_sc20676_receipt(&bad).is_err());
        let mut bad = receipt();
        bad.packed.memory.observed_dense_kv_bytes = 0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn storage_components_allocator_delta_and_comparison_geometry_fail_closed() {
        let mut bad = receipt();
        bad.packed
            .packed
            .as_mut()
            .unwrap()
            .retained_device_metadata_bytes = 0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        let packed = bad.packed.packed.as_mut().unwrap();
        packed.accepted_uploaded_packed_bytes = 7;
        bad.packed.memory.packed_logical_payload_bytes = packed.accepted_uploaded_packed_bytes;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.packed.memory.phases[4].mlx_active_bytes = 3;
        bad.packed.memory.phases[4].mlx_peak_bytes = 5;
        bad.packed.memory.packed_device_bytes = 1;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.packed.memory.theoretical_dense_reconstruction_bytes = 1;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn arm_phase_peaks_and_timing_labels_fail_closed() {
        let mut bad = receipt();
        bad.dense.peak_mlx_bytes = 3;
        reseal_arm(&mut bad.dense);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.packed.peak_phys_footprint_bytes = 9;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.dense.elapsed_ms = 0.0;
        reseal_arm(&mut bad.dense);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.packed.timings[0].first_token_ms = 1.0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.dense.timings[0].packed_cold_dispatch_ms = 1.0;
        reseal_arm(&mut bad.dense);
        assert!(validate_sc20676_receipt(&bad).is_err());

        let mut bad = receipt();
        bad.packed.timings[0].packed_warm_dispatch_ms = 0.0;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt(&bad).is_err());
    }
    #[test]
    fn decode_peak_attribution_requires_a_sealed_reset_after_prefill() {
        let mut valid = receipt();
        valid.packed.memory.phases[2].mlx_peak_bytes = 100;
        valid.packed.peak_mlx_bytes = 100;
        reseal_arm(&mut valid.packed);
        validate_sc20676_receipt_core(&valid).unwrap();

        let mut bad = valid.clone();
        bad.packed.memory.decode_peak_window.reset_peak_bytes = 100;
        bad.packed.memory.phases[3].mlx_peak_bytes = 100;
        reseal_arm(&mut bad.packed);
        assert!(validate_sc20676_receipt_core(&bad).is_err());

        let mut stale_window = valid;
        stale_window.packed.memory.decode_peak_window.started_at =
            stale_window.packed.memory.phases[2].captured_at.clone();
        reseal_arm(&mut stale_window.packed);
        assert!(validate_sc20676_receipt_core(&stale_window).is_err());
    }
    #[test]
    fn packed_kernel_profile_is_bound_to_device_and_dispatch_geometry() {
        let valid = receipt();
        validate_sc20676_receipt_core(&valid).unwrap();

        let mut wrong_device = valid.clone();
        wrong_device
            .packed
            .kernel_profile
            .as_mut()
            .unwrap()
            .metal_device = "another device".into();
        reseal_arm(&mut wrong_device.packed);
        assert!(validate_sc20676_receipt_core(&wrong_device).is_err());

        let mut wrong_family = valid.clone();
        wrong_family
            .packed
            .kernel_profile
            .as_mut()
            .unwrap()
            .gpu_family = "apple7-or-newer".into();
        reseal_arm(&mut wrong_family.packed);
        assert!(validate_sc20676_receipt_core(&wrong_family).is_err());

        let mut wrong_geometry = valid;
        wrong_geometry
            .packed
            .kernel_profile
            .as_mut()
            .unwrap()
            .threads = 64;
        reseal_arm(&mut wrong_geometry.packed);
        assert!(validate_sc20676_receipt_core(&wrong_geometry).is_err());
    }
    #[test]
    fn post_model_snapshot_inventory_must_match_the_initial_validation() {
        let initial = inventory(&digest(), 42);
        assert!(validate_snapshot_inventory_unchanged(&initial, &inventory(&digest(), 42)).is_ok());
        assert!(
            validate_snapshot_inventory_unchanged(&initial, &inventory(&"b".repeat(64), 42))
                .is_err()
        );
        assert!(
            validate_snapshot_inventory_unchanged(&initial, &inventory(&digest(), 43)).is_err()
        );
    }
    #[test]
    fn canonical_origin_rejects_wrong_repository_identity() {
        let origin =
            campaign::canonical_github_repository("git@github.com:SceneWorks/inference.git")
                .unwrap();
        assert_eq!(origin, campaign::INFERENCE_REPOSITORY);
        assert_ne!(origin, campaign::SCENEWORKS_REPOSITORY);
        assert!(campaign::canonical_github_repository(
            "https://example.invalid/SceneWorks/inference"
        )
        .is_err());
    }
    #[test]
    fn exact_baseline_length_and_receipt_bytes_tampering_fail_closed() {
        let mut bad = receipt();
        bad.dense.prompt_tokens += 1;
        reseal_arm(&mut bad.dense);
        assert!(validate_sc20676_receipt(&bad).is_err());
        let sealed = receipt().finish().unwrap();
        let bytes = sealed.bytes().unwrap();
        let seal = campaign::seal_bytes(&bytes);
        assert!(parse_staged_receipt_bytes(&bytes, &seal).is_ok());
        let mut whitespace_tampered = bytes;
        whitespace_tampered.push(b' ');
        assert!(parse_staged_receipt_bytes(&whitespace_tampered, &seal).is_err());
    }
}

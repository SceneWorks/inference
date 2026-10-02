//! Source-owned evidence primitives for the SC-20671 dense baseline campaign.
//!
//! This module deliberately keeps the receipt producer beside the product decoder.  The JSON
//! harness can validate evidence, but it must not be the component inventing model identity,
//! geometry, or lifecycle observations.  Device collection is supplied by the campaign runner;
//! these helpers make its inputs deterministic and testable without weights or Metal.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use crate::campaign_supervisor::{self, RunRequest, SafetyPolicy, SystemProbe};

pub const SC20671_SCHEDULE_VERSION: u64 = 2;
pub const SC20671_CAMPAIGN_KIND: &str = "sc-20671-complete-covering-set";
/// Manifest kind of an `--only-coordinate` run: one scheduled row, never a campaign. Every
/// complete-campaign loader (here and in SceneWorks) refuses it by kind.
pub const SC20671_PARTIAL_RUN_KIND: &str = "sc-20671-partial-coordinate-run";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CampaignSafetyPolicy {
    pub schema_version: u64,
    pub row_deadline_seconds: u64,
    pub poll_millis: u64,
    pub term_grace_millis: u64,
    pub host_free_reserve_bytes: u64,
    pub child_footprint_cap_bytes: u64,
    pub max_context_tokens: u64,
    pub max_request_tokens: u64,
    pub stdout_cap_bytes: u64,
    pub stderr_cap_bytes: u64,
}

impl CampaignSafetyPolicy {
    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.row_deadline_seconds == 0
            || self.poll_millis == 0
            || self.term_grace_millis == 0
            || self.host_free_reserve_bytes == 0
            || self.child_footprint_cap_bytes == 0
            || self.max_context_tokens == 0
            || self.max_request_tokens == 0
            || self.stdout_cap_bytes == 0
            || self.stderr_cap_bytes == 0
            || self
                .host_free_reserve_bytes
                .checked_add(self.child_footprint_cap_bytes)
                .is_none()
            || self.poll_millis >= self.row_deadline_seconds.saturating_mul(1_000)
            || self.term_grace_millis >= self.row_deadline_seconds.saturating_mul(1_000)
        {
            return Err("SC-20671/76 safety policy is missing, unsupported, or invalid".into());
        }
        Ok(())
    }

    pub fn supervisor(&self) -> SafetyPolicy {
        SafetyPolicy {
            deadline: Duration::from_secs(self.row_deadline_seconds),
            poll_interval: Duration::from_millis(self.poll_millis),
            term_grace: Duration::from_millis(self.term_grace_millis),
            host_free_reserve_bytes: self.host_free_reserve_bytes,
            child_footprint_cap_bytes: self.child_footprint_cap_bytes,
            max_context_tokens: self.max_context_tokens,
            max_request_tokens: self.max_request_tokens,
            stdout_cap_bytes: self.stdout_cap_bytes,
            stderr_cap_bytes: self.stderr_cap_bytes,
        }
    }

    pub fn seal(&self) -> Result<String, String> {
        let value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        Ok(seal_bytes(
            &canonical_json_bytes(&value).map_err(|e| e.to_string())?,
        ))
    }
}

pub fn load_campaign_safety_policy(path: &Path) -> Result<CampaignSafetyPolicy, String> {
    let bytes = fs::read(path).map_err(|e| format!("read mandatory safety policy: {e}"))?;
    let policy: CampaignSafetyPolicy = serde_json::from_slice(&bytes)
        .map_err(|e| format!("decode mandatory safety policy: {e}"))?;
    policy.validate()?;
    Ok(policy)
}

use core_llm::{
    Message, Role, Sampling, StreamEvent, TextLlmOutput, TextLlmRequest, ThinkingMode, Tokenizer,
    ToolSpec,
};

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
/// Receipt schema. v5 (sc-20671): per-repeat decode throughput is a dedicated fixed-length steady
/// decode recorded beside each sample, and provenance records the host's power mode and thermal
/// state at row start and end. v6 (sc-20671 hardware audit): real-hardware observations are
/// recorded instead of refused: host state per timing sample and thermal/power change flags (only
/// a throttled row start refuses), the dense-KV share of the prefill footprint, post-release MLX
/// residuals within a defined slack, teacher-forced greedy agreement with the free-running
/// divergence, and compile cost against a noise band.
/// v7 (quality contract v4): the multi-turn prompt-cache fixture is a real two-turn conversation
/// whose turn 2 is served by a prompt-cache hit, its agreement is teacher-forced (compressed rows on
/// a turn-2 forced continuation), and each arm's per-turn cache record is sealed.
pub const RECEIPT_SCHEMA_VERSION: u32 = 7;
pub const RECEIPT_HARNESS_VERSION: &str = "sc-20671-kv-baseline-v7";
/// Exact-byte SHA-256 of SceneWorks `config/kv-baseline-quality-contract.json` (contract v5).
pub const QUALITY_CONTRACT_HASH: &str =
    "8461b072e36493f4f4559e7fc858a286a239a43e8ca8d8ea93179bcf33f344ca";
/// Quality measurements per arm per process (contract v5 `statistics.qualityMeasuredOnce`): every
/// fixture, forced continuation and teacher-forced pass is deterministic within a process, so a
/// row measures quality once and gates that one sealed measurement.
pub const QUALITY_MEASUREMENTS: usize = 1;
/// Timing repeats (contract `statistics.repeats`): each is the kernel fixture's coordinate
/// operation and a fixed-length steady decode, feeding the coefficient-of-variation rule.
pub const TIMING_REPEATS: usize = 5;
/// Timing warmups of a warm row (contract `statistics.warmups`).
pub const TIMING_WARMUPS: usize = 2;
/// Frozen compressed-domain parity contract, shared by SC-20671 and the SC-20676 product proof.
pub const COMPRESSED_PARITY_MAX_ERROR: f64 = 0.0001;
pub const COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN: f64 = 0.999;
/// Frozen compressed perplexity-delta maximum (contract v3 `thresholds.perplexityDelta`).
pub const COMPRESSED_PERPLEXITY_DELTA_MAX: f64 = 0.01;

/// Direction of a frozen quality threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityGateComparison {
    /// The measured value must be at least the threshold.
    Minimum,
    /// The measured value must be at most the threshold.
    Maximum,
}

impl QualityGateComparison {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimum => "minimum",
            Self::Maximum => "maximum",
        }
    }

    fn misses(self, value: f64, threshold: f64) -> bool {
        match self {
            Self::Minimum => value < threshold,
            Self::Maximum => value > threshold,
        }
    }
}

/// One gated quality metric: its fixture, frozen threshold, and direction.
pub struct QualityGateMetric {
    pub metric: &'static str,
    pub fixture: &'static str,
    pub threshold: f64,
    pub comparison: QualityGateComparison,
    read: fn(&QualityMetrics) -> f64,
}

/// The measured-quality gate of a compressed row: each contract v3 threshold, evaluated for every
/// measured repeat against the same-weights dense-KV run. A miss is recorded evidence for the
/// SC-20678 Go/No-Go decision, never a refused row. Kernel parity (`parityMaxError`) is
/// deliberately absent: it compares the fused reader with its independent host-fp32
/// dequantize-then-attend reference over the exact stored codes, a kernel-correctness check that
/// still refuses the row.
pub const QUALITY_GATE_METRICS: [QualityGateMetric; 5] = [
    QualityGateMetric {
        metric: "greedyTokenAgreement",
        fixture: "kernel-fp32-reference",
        threshold: COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN,
        comparison: QualityGateComparison::Minimum,
        read: |metrics| metrics.greedy_token_agreement,
    },
    QualityGateMetric {
        metric: "perplexityDelta",
        fixture: "kernel-fp32-reference",
        threshold: COMPRESSED_PERPLEXITY_DELTA_MAX,
        comparison: QualityGateComparison::Maximum,
        read: |metrics| metrics.perplexity_delta,
    },
    QualityGateMetric {
        metric: "structuredToolAgreement",
        fixture: "structured-tool-call",
        threshold: 1.0,
        comparison: QualityGateComparison::Minimum,
        read: |metrics| metrics.structured_tool_agreement,
    },
    QualityGateMetric {
        metric: "needleRetrieval",
        fixture: "long-context-needle",
        threshold: 1.0,
        comparison: QualityGateComparison::Minimum,
        read: |metrics| metrics.needle_retrieval,
    },
    QualityGateMetric {
        metric: "multiTurnPromptCache",
        fixture: "multi-turn-prompt-cache",
        threshold: COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
        comparison: QualityGateComparison::Minimum,
        read: |metrics| metrics.multi_turn_prompt_cache,
    },
];

/// Evaluate the compressed quality gate over the row's quality measurement(s), in order (contract
/// v5: exactly one).
/// Contract v5: a non-discriminating needle (the same-weights dense run missed it) is an
/// observation only, so `needleRetrieval` is evaluated only when `needle_discriminating`.
pub fn quality_gate_from_repeats(
    repeats: &[QualityMetrics],
    needle_discriminating: bool,
) -> ReceiptQualityGate {
    let failures = repeats
        .iter()
        .enumerate()
        .flat_map(|(repeat, metrics)| {
            QUALITY_GATE_METRICS.iter().filter_map(move |spec| {
                let value = (spec.read)(metrics);
                let gated = needle_discriminating || spec.metric != "needleRetrieval";
                (gated && spec.comparison.misses(value, spec.threshold)).then(|| {
                    ReceiptQualityGateFailure {
                        metric: spec.metric.into(),
                        fixture: spec.fixture.into(),
                        repeat: repeat as u64,
                        value,
                        threshold: spec.threshold,
                        comparison: spec.comparison.as_str().into(),
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    ReceiptQualityGate {
        passed: failures.is_empty(),
        failures,
    }
}

/// Human-readable outcome of a quality gate, for diagnostics and the human receipt.
pub fn quality_gate_summary(gate: &ReceiptQualityGate) -> String {
    if gate.passed {
        return "passed".into();
    }
    let failures = gate
        .failures
        .iter()
        .map(|failure| {
            format!(
                "{} repeat {} = {} ({} {}, fixture {})",
                failure.metric,
                failure.repeat,
                failure.value,
                failure.comparison,
                failure.threshold,
                failure.fixture
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!("FAILED: {failures}")
}
/// Greedy agreement under quality contract v3 is teacher-forced (decided before any compressed
/// result existed): the candidate's greedy choice at each position of the reference stream.
pub const GREEDY_AGREEMENT_METHOD: &str = "teacher-forced";
/// Quality contract v3: the only denominator of a compressed receipt's quality gate is the
/// dense-KV run on the same weights. The bf16 model remains weight-quantization characterization.
pub const COMPRESSED_QUALITY_REFERENCE: &str = "dense-kv-same-weights";
/// Quality contract v3 needle wording (RULER/NIAH style). The v2 credential-like "passphrase"
/// wording made the dense Q4 Llama baseline answer with a safety refusal. The needle token itself
/// is still matched exactly.
pub const NEEDLE_FIXTURE_STATEMENT_PREFIX: &str = "The special magic identifier is ";
pub const NEEDLE_FIXTURE_QUESTION: &str =
    "What is the special magic identifier mentioned in the text above? Reply with only the identifier.";
/// Every campaign row is admitted by its runtime guards (supervised worker, footprint watchdog cap,
/// host reserve, deadline, sampling), never by a static whole-process peak proof.
pub const RUNTIME_GUARDED_ADMISSION: &str = "runtime-guarded";
/// Admission estimate source of an SC-20671 row: `static_row_footprint_budget`, the larger of
/// the candidate and reference roles' model-load (2x payload) plus KV, fused prefill tile and
/// tiled-prefill activation budget.
pub const SC20671_ESTIMATE_SOURCE: &str = "sc20671-static-row-footprint-budget";
/// Estimate source of a noise-floor row: one dense candidate session (`static_role_footprint_budget`).
pub const NOISE_FLOOR_ESTIMATE_SOURCE: &str = "sc20669-noise-floor-dense-session-budget";
/// Allocation kinds that explicitly witness a dense full-cache temporary. Mirrors SceneWorks'
/// `detectFullCacheTemporary`; a compressed receipt carrying either is rejected regardless of size.
pub const FULL_CACHE_MATERIALIZATION_KIND: &str = "full_cache_materialization";
pub const DENSE_CACHE_TEMPORARY_KIND: &str = "dense_cache_temporary";
/// Compressed-arm operations the compressed representation has no route for. They run on the
/// explicit dense path and every execution is recorded as a reasoned fallback in the receipt.
pub const COMPRESSED_PREFIX_REUSE_FALLBACK_REASON: &str = "the provider prefix cache computes and stores shared prefixes as dense contiguous K/V; only a compressed cache-hit decode imports a reused prefix (quantize-on-append)";
pub const COMPRESSED_BATCH_FALLBACK_REASON: &str = "batched prefill attends through additive padding masks, which the compressed fused reader cannot apply; the synchronous batch decoder keeps its dense cache";

pub const CONTEXT_BANDS: [&str; 4] = ["short", "medium", "memory-material", "fit-boundary"];
pub const MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS: u64 = 1_000;
pub const FIT_BOUNDARY_MIN_CONTEXT_BPS: u64 = 9_000;
/// Fixed allowance for process-resident Metal/JIT runtime pages after MLX live tensors and its
/// allocator cache have returned to the loaded-model boundary.
pub const POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES: u64 = 512 * 1024 * 1024;
/// Floor of the post-release MLX allocator slack: real hardware leaves a few small allocator blocks
/// (scalars, RNG state, compiled-kernel constants) that are not a leak.
pub const POST_RELEASE_MLX_SLACK_FLOOR_BYTES: u64 = 1024 * 1024;

/// Post-release MLX active/cache slack over a loaded-model `baseline`: `max(1 MiB, 0.1% of
/// baseline)`, rounded up. A residual above it is a material leak and still fails.
pub fn post_release_mlx_slack_bytes(baseline: u64) -> u64 {
    POST_RELEASE_MLX_SLACK_FLOOR_BYTES.max(baseline.div_ceil(1_000))
}
pub const SCENEWORKS_REPOSITORY: &str = "github.com/SceneWorks/SceneWorks";
pub const INFERENCE_REPOSITORY: &str = "github.com/SceneWorks/inference";
pub const PMETAL_MLX_REPOSITORY: &str = "https://github.com/michaeltrefry/mlx-rs";
pub const SC20671_MIN_NATIVE_CONTEXT_TOKENS: u64 = 32 * 1024;

pub(crate) fn valid_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    let fixed = |index: usize| bytes.get(index).is_some_and(u8::is_ascii_digit);
    let shape = (bytes.len() == 20
        || (bytes.len() >= 22 && bytes[19] == b'.' && bytes[bytes.len() - 1] == b'Z'))
        && bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && bytes.get(10) == Some(&b'T')
        && bytes.get(13) == Some(&b':')
        && bytes.get(16) == Some(&b':')
        && bytes.last() == Some(&b'Z')
        && (0..4).all(fixed)
        && (5..7).all(fixed)
        && (8..10).all(fixed)
        && (11..13).all(fixed)
        && (14..16).all(fixed)
        && (17..19).all(fixed)
        && (bytes.len() == 20 || (20..bytes.len() - 1).all(fixed));
    if !shape {
        return false;
    }
    let parse =
        |range: std::ops::Range<usize>| value.get(range).and_then(|part| part.parse::<u32>().ok());
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
}

/// Seconds since the Unix epoch of a validated UTC timestamp (fraction included).
pub(crate) fn utc_timestamp_seconds(value: &str) -> Option<f64> {
    if !valid_utc_timestamp(value) {
        return None;
    }
    let parse = |range: std::ops::Range<usize>| value.get(range)?.parse::<i64>().ok();
    let (year, month, day) = (parse(0..4)?, parse(5..7)?, parse(8..10)?);
    let (hour, minute, second) = (parse(11..13)?, parse(14..16)?, parse(17..19)?);
    // Days from civil (proleptic Gregorian), the inverse of `timestamp_now`'s conversion.
    let shifted = if month <= 2 { year - 1 } else { year };
    let era = if shifted >= 0 { shifted } else { shifted - 399 } / 400;
    let year_of_era = shifted - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let fraction = if value.len() > 20 {
        format!("0.{}", &value[20..value.len() - 1])
            .parse::<f64>()
            .ok()?
    } else {
        0.0
    };
    Some((days * 86_400 + hour * 3_600 + minute * 60 + second) as f64 + fraction)
}

pub(crate) fn compare_utc_timestamps(left: &str, right: &str) -> Option<std::cmp::Ordering> {
    if !valid_utc_timestamp(left) || !valid_utc_timestamp(right) {
        return None;
    }
    let whole_seconds = left[..19].cmp(&right[..19]);
    if whole_seconds != std::cmp::Ordering::Equal {
        return Some(whole_seconds);
    }
    let left_fraction = if left.len() == 20 {
        &[][..]
    } else {
        &left.as_bytes()[20..left.len() - 1]
    };
    let right_fraction = if right.len() == 20 {
        &[][..]
    } else {
        &right.as_bytes()[20..right.len() - 1]
    };
    for index in 0..left_fraction.len().max(right_fraction.len()) {
        let ordering = left_fraction
            .get(index)
            .copied()
            .unwrap_or(b'0')
            .cmp(&right_fraction.get(index).copied().unwrap_or(b'0'));
        if ordering != std::cmp::Ordering::Equal {
            return Some(ordering);
        }
    }
    Some(std::cmp::Ordering::Equal)
}

pub(crate) fn utc_timestamp_before(left: &str, right: &str) -> bool {
    compare_utc_timestamps(left, right) == Some(std::cmp::Ordering::Less)
}

/// One file which must be present, byte-for-byte, in a SC-20671 benchmark snapshot.
/// Binding every published safetensors payload and the parsed config is sufficient to reject a
/// caller-provided model substitution without inventing a cache path convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedSnapshotFile {
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// Source-owned identity for one of the four dense-baseline snapshots.  These are commit IDs, not
/// mutable Hugging Face branches or aliases.  Snapshot directories intentionally remain CLI
/// inputs: the operator controls storage, while this contract controls what may be loaded from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkModelSpec {
    pub family: &'static str,
    pub role: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub architecture: &'static str,
    pub model_type: &'static str,
    pub native_context_tokens: u64,
    pub quantized: bool,
    pub required_files: &'static [PinnedSnapshotFile],
}

const LLAMA_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 1122,
        sha256: "c546925585e48f43890d9dc5150df4fec73dd3780d92961c5ace451934cc4cd6",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 1_807_496_278,
        sha256: "d75e1ee0ea653cc5b76191ec934c7c0d568e94d4e47846619f1f4bc715b7b265",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 45_720,
        sha256: "2ef31fa0b9dcda01f87835851d5e1d5a39ab6258ae618af6ea864c3747e556e4",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 54_558,
        sha256: "022d5ae3df4737998ab97d8f31ac2bcb4c06dd8ebe5a8aba2b4aceef1e5ea7d3",
    },
];
const LLAMA_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 969,
        sha256: "7e4149635018dd8d82f9b0873d800459faca261b1ac969d458c93f51133643a8",
    },
    PinnedSnapshotFile {
        path: "model-00001-of-00002.safetensors",
        bytes: 5_368_478_882,
        sha256: "b475af8535933afaf9393b30c6bcb4bfede59586b16ecdd010e7da8a83dfc95b",
    },
    PinnedSnapshotFile {
        path: "model-00002-of-00002.safetensors",
        bytes: 1_057_050_089,
        sha256: "9b2903fa2b3728423f6442c0dc667db645906fb113c9d0cd794faeb820cc82d3",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 21_946,
        sha256: "de5995e586fc98a942b576a6752034a6d0a741d2dece5f61e697d99c8bb99b3d",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 54_528,
        sha256: "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
    },
];
const QWEN_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 937,
        sha256: "507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 968_080_210,
        sha256: "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 49_731,
        sha256: "1e3058d4ba4b04e4de35b74467725cbef90ff022198404218e48f21adc9cfa15",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];
const QWEN_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 784,
        sha256: "fa2aca3f3437d838672845487b8dd013b6a1f022daafb457bc9cb08564663ee4",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 3_441_185_437,
        sha256: "790a82e1fcbac0fa316e4c21cd025c0c39aa03e882e33e2b1dc39407c9629e4e",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 22_148,
        sha256: "d6a4b92a4ede4c18d5bbc8e5814e7a4b899a514262e74b92cedba90f0c3e836d",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];

pub const LLAMA_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama",
    role: "candidate",
    repository: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    revision: "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    quantized: true,
    required_files: LLAMA_4BIT_FILES,
};
pub const LLAMA_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama",
    role: "bf16-reference",
    repository: "mlx-community/Llama-3.2-3B-Instruct-bf16",
    revision: "6d88ba43024fef71b10e52e101c7cd4598322601",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    quantized: false,
    required_files: LLAMA_BF16_FILES,
};
pub const QWEN_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen",
    role: "candidate",
    repository: "mlx-community/Qwen3-1.7B-4bit",
    revision: "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    quantized: true,
    required_files: QWEN_4BIT_FILES,
};
pub const QWEN_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen",
    role: "bf16-reference",
    repository: "mlx-community/Qwen3-1.7B-bf16",
    revision: "9cd6692855d3e06772228e9a962b2606359b2d24",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    quantized: false,
    required_files: QWEN_BF16_FILES,
};

pub fn benchmark_model(
    family: &str,
    reference: bool,
) -> Result<&'static BenchmarkModelSpec, String> {
    match (family, reference) {
        ("llama", false) => Ok(&LLAMA_CANDIDATE),
        ("llama", true) => Ok(&LLAMA_REFERENCE),
        ("qwen", false) => Ok(&QWEN_CANDIDATE),
        ("qwen", true) => Ok(&QWEN_REFERENCE),
        _ => Err(format!(
            "SC-20671 has no immutable benchmark model for family {family:?}"
        )),
    }
}

pub fn context_band_target(context_window: u64, context_band: &str) -> Result<u64, String> {
    if context_window < 1_024 {
        return Err("SC-20671 requires a context window of at least 1024 tokens".into());
    }
    let medium = (context_window / 16).clamp(128, 1_024);
    let memory_material = context_window / 4;
    let fit_by_ratio = context_window
        .checked_mul(FIT_BOUNDARY_MIN_CONTEXT_BPS)
        .and_then(|tokens| tokens.checked_add(9_999))
        .map(|tokens| tokens / 10_000)
        .ok_or("fit-boundary target overflows u64")?;
    let fit_boundary = context_window.saturating_sub(512).max(fit_by_ratio);
    if !(32 < medium && medium < memory_material && memory_material < fit_boundary) {
        return Err(format!(
            "loaded context window {context_window} cannot represent four distinct bands"
        ));
    }
    match context_band {
        "short" => Ok(32),
        "medium" => Ok(medium),
        "memory-material" => Ok(memory_material),
        "fit-boundary" => Ok(fit_boundary),
        _ => Err(format!("unknown context band {context_band}")),
    }
}

/// Live tokens of a compressed row's forced continuation: the kernel prompt plus its continuation,
/// which a fit-boundary row shortens to the native window.
fn forced_continuation_live_tokens(kernel_prompt_tokens: u64, native_context_tokens: u64) -> u64 {
    kernel_prompt_tokens
        + FORCED_CONTINUATION_TOKENS.min(native_context_tokens.saturating_sub(kernel_prompt_tokens))
}

/// Compute a conservative total-live-token bound using only the pinned tokenizer and source
/// fixture construction. Product chat rendering is separately capped by the native window; the
/// prefix/batch paths bypass that renderer, so their exact raw prompt lengths are checked here.
/// A lifecycle can hold one prefix cache alongside one full native-window request, while a batch
/// can hold two prefix-length lanes. This is checked before any weights or Metal model is loaded.
fn preflight_total_live_tokens(
    snapshot: &Path,
    spec: &BenchmarkModelSpec,
    coordinate: &Coordinate,
    prompt: &str,
    compressed: bool,
) -> Result<(u64, u64), String> {
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_err(|e| format!("load pinned tokenizer for safety preflight: {e}"))?;
    let target = context_band_target(spec.native_context_tokens, coordinate.context_band)?;
    let header = format!("SC20671-CONTEXT-BAND-{}", coordinate.context_band);
    let token_count = |text: &str| -> Result<u64, String> {
        u64::try_from(
            tokenizer
                .encode(text, false)
                .map_err(|e| e.to_string())?
                .len(),
        )
        .map_err(|_| "preflight token count overflows u64".into())
    };
    let mut lower = 0_usize;
    let mut upper = usize::try_from(target).map_err(|_| "target overflows usize")?;
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        let candidate = format!("{header}{}", " context".repeat(midpoint));
        if token_count(&candidate)? <= target {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    let payload = format!("{header}{}", " context".repeat(lower));
    // The multi-turn fixture's payload: the same search, capped (fit-boundary) by
    // `multi_turn_payload_target`.
    let multi_turn_target = multi_turn_payload_target(spec.native_context_tokens, target)?;
    let (mut lower, mut upper) = (0_usize, lower);
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        let candidate = format!("{header}{}", " context".repeat(midpoint));
        if token_count(&candidate)? <= multi_turn_target {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    let multi_turn_prompt = format!(
        "{prompt}\n{header}{}\nRepeat the stable baseline fact.",
        " context".repeat(lower)
    );
    let needle = "SC20671-NUMERIC-NEEDLE-9b7a2e";
    let prompts = [
        format!("{prompt}\n{payload}\nReturn a concise deterministic answer."),
        format!("{prompt}\n{payload}\nCall record_baseline_fact with fact exactly `SC20671 structured fixture`."),
        needle_fixture_prompt(prompt, &payload, needle),
        format!("{prompt}\n{payload}\nRepeat the stable baseline fact."),
    ];
    let prompt_tokens = prompts
        .iter()
        .map(|text| token_count(text))
        .collect::<Result<Vec<_>, _>>()?;
    let max_prefix = prompt_tokens
        .iter()
        .copied()
        .max()
        .ok_or("no fixture prompts")?;
    // Each candidate repeat's steady decode holds the kernel fixture's raw context plus the fixed
    // decode length in its own cache, alone (every fixture cache has been released).
    let steady_decode = prompt_tokens[0]
        .checked_add(STEADY_DECODE_TOKENS)
        .ok_or("steady-decode live context overflows u64")?;
    let direct_request = max_prefix
        .checked_add(2)
        .ok_or("direct request token count overflows")?;
    if max_prefix > spec.native_context_tokens.saturating_sub(256) {
        return Err(format!(
            "{} fixture prompt exceeds the producer's native-context reserve before model load",
            coordinate_slug(coordinate)
        ));
    }
    // Direct prefix and batch paths bypass the chat renderer but generate at most one/two tokens.
    if direct_request > spec.native_context_tokens {
        return Err(format!(
            "{} prefix/batch prompt exceeds native context before model load",
            coordinate_slug(coordinate)
        ));
    }
    let rendered = prompts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let tools = if index == 1 {
                vec![structured_fixture_tool()]
            } else {
                Vec::new()
            };
            crate::provider::campaign_preflight_request_tokens(
                snapshot,
                &fixture_request(text.clone(), tools),
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no rendered requests")?;
    // The multi-turn fixture (contract v4) holds turn 1's stored prompt and answer beside turn
    // 2's request cache: turn 2 renders turn 1's conversation, an answer of at most
    // `FIXTURE_MAX_NEW_TOKENS` tokens (bounded here by a placeholder answer of that many words plus
    // a re-tokenization margin), and the follow-up, then decodes its natural answer or, on a
    // compressed row, its forced continuation.
    let turn1_rendered = crate::provider::campaign_preflight_request_tokens(
        snapshot,
        &fixture_request(multi_turn_prompt.clone(), Vec::new()),
    )
    .map_err(|e| e.to_string())?;
    let mut turn2_request = fixture_request(multi_turn_prompt, Vec::new());
    turn2_request.messages.push(Message::text(
        Role::Assistant,
        " context".repeat(FIXTURE_MAX_NEW_TOKENS as usize),
    ));
    turn2_request
        .messages
        .push(Message::text(Role::User, MULTI_TURN_FIXTURE_FOLLOW_UP));
    let turn2_rendered =
        crate::provider::campaign_preflight_request_tokens(snapshot, &turn2_request)
            .map_err(|e| e.to_string())?
            .checked_add(MULTI_TURN_ANSWER_RETOKENIZATION_MARGIN)
            .ok_or("multi-turn live context overflows u64")?;
    // Contract v4: turn 2 must leave the full turn-2 forced continuation, on every row (the
    // fixture is the same for dense and compressed rows), or the row is refused before load.
    if turn2_rendered + FORCED_CONTINUATION_TOKENS > spec.native_context_tokens {
        return Err(format!(
            "{} multi-turn fixture turn 2 ({turn2_rendered} tokens) leaves fewer than the \
             {FORCED_CONTINUATION_TOKENS}-token forced continuation in the {}-token window",
            coordinate_slug(coordinate),
            spec.native_context_tokens
        ));
    }
    let turn2_decode = if compressed {
        FORCED_CONTINUATION_TOKENS
    } else {
        u64::from(FIXTURE_MAX_NEW_TOKENS)
    };
    let multi_turn = [
        turn1_rendered,
        u64::from(FIXTURE_MAX_NEW_TOKENS),
        turn2_rendered,
        turn2_decode,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
    .ok_or("multi-turn live context overflows u64")?;
    let cancellation = crate::provider::campaign_preflight_cancellation_tokens(snapshot)
        .map_err(|e| e.to_string())?;
    if rendered > spec.native_context_tokens || cancellation > spec.native_context_tokens {
        return Err(format!(
            "{} rendered request exceeds native context before model load",
            coordinate_slug(coordinate)
        ));
    }
    let max_request = rendered.max(direct_request).max(cancellation);
    let prefix_plus_request = max_prefix
        .checked_add(1)
        .and_then(|prefix| prefix.checked_add(rendered.max(cancellation)))
        .ok_or("total live context overflows u64")?;
    let batch = max_prefix
        .checked_add(2)
        .and_then(|tokens| tokens.checked_mul(2))
        .ok_or("batch live context overflows u64")?;
    let total = if coordinate.request_mode == "supported-batch" {
        if coordinate.prefill_mode == "chunked" {
            prefix_plus_request.max(batch)
        } else {
            max_request.max(batch)
        }
    } else if coordinate.prefill_mode == "chunked" {
        prefix_plus_request
    } else {
        max_request
    };
    // A compressed row's forced continuation holds the kernel context plus its continuation (at
    // most the native window) in its own request-scoped cache.
    let forced_continuation = if compressed {
        forced_continuation_live_tokens(prompt_tokens[0], spec.native_context_tokens)
    } else {
        0
    };
    Ok((
        total
            .max(steady_decode)
            .max(forced_continuation)
            .max(multi_turn),
        max_request.max(turn2_rendered),
    ))
}

/// The dtype every campaign role's dense K/V is cached in: the causal loader's compute dtype. The
/// loader holds stored quantized scales/biases in the compute dtype (sc-20671), so a stored F16/F32
/// scale no longer promotes the activations — and the cache — to F32.
pub const DENSE_KV_COMPUTE_DTYPE: mlx_rs::Dtype = crate::models::CausalLm::COMPUTE_DTYPE;

/// Scalar width of [`DENSE_KV_COMPUTE_DTYPE`]. [`ProductObserver`] and
/// [`validate_receipt_semantics`] refuse any other observed width rather than record a silently
/// widened dense baseline.
pub const DENSE_KV_COMPUTE_ELEMENT_BYTES: u64 =
    crate::primitives::dtype_bytes(DENSE_KV_COMPUTE_DTYPE);

/// Validate the pinned decoder projection inventory without loading MLX arrays and return the
/// dense K/V scalar width it produces. Every supported stored scale dtype (BF16/F16/F32) is cast to
/// the BF16 compute dtype at load, and dense projections are cast to BF16 too, so the width is the
/// compute dtype's whatever the snapshot stores — never the widest stored scale (the pre-sc-20671
/// F32-promoted path). Both parent preflight and worker validate the exact pinned snapshot
/// inventory before this call.
fn pinned_dense_kv_element_bytes(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    layers: u64,
) -> Result<u64, String> {
    let mut dtypes = std::collections::HashMap::<String, String>::new();
    for required in spec
        .required_files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
    {
        let path = snapshot.join(required.path);
        let mut file = File::open(&path)
            .map_err(|e| format!("read pinned weight header {}: {e}", path.display()))?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("read pinned weight header length {}: {e}", path.display()))?;
        let len = u64::from_le_bytes(prefix);
        if len > 64 * 1024 * 1024 || len > required.bytes.saturating_sub(8) {
            return Err("pinned safetensors header length is invalid".into());
        }
        let mut bytes = vec![0_u8; len as usize];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("read pinned weight header {}: {e}", path.display()))?;
        let header: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse pinned weight header: {e}"))?;
        for (key, tensor) in header
            .as_object()
            .ok_or("pinned safetensors header is not an object")?
        {
            if key == "__metadata__" {
                continue;
            }
            let dtype = tensor
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("pinned tensor {key} lacks dtype"))?;
            if dtypes.insert(key.clone(), dtype.to_owned()).is_some() {
                return Err(format!("duplicate pinned tensor {key}"));
            }
        }
    }
    for layer in 0..layers {
        for projection in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let stem = format!("model.layers.{layer}.{projection}");
            let weight = dtypes
                .get(&format!("{stem}.weight"))
                .ok_or_else(|| format!("pinned decoder lacks {stem}.weight"))?;
            match dtypes.get(&format!("{stem}.scales")) {
                Some(scales) => {
                    let biases = dtypes
                        .get(&format!("{stem}.biases"))
                        .ok_or_else(|| format!("pinned decoder lacks {stem}.biases"))?;
                    if !spec.quantized || weight != "U32" || scales != biases {
                        return Err(format!("unsupported pinned quantized projection {stem}"));
                    }
                    // Held in the BF16 compute dtype at load whatever the stored width.
                    if !matches!(scales.as_str(), "BF16" | "F16" | "F32") {
                        return Err(format!("unsupported pinned scale dtype for {stem}"));
                    }
                }
                None => {
                    if !matches!(weight.as_str(), "BF16" | "F16" | "F32")
                        || dtypes.contains_key(&format!("{stem}.biases"))
                    {
                        return Err(format!("unsupported pinned dense projection {stem}"));
                    }
                }
            }
        }
    }
    Ok(DENSE_KV_COMPUTE_ELEMENT_BYTES)
}

/// The fail-closed reason for an observed dense K/V width that is not the compute dtype's.
fn dense_kv_width_refusal(observed: u64) -> String {
    let observed_dtype = match observed {
        1 => "an 8-bit",
        2 => "a 16-bit float",
        4 => "Float32",
        8 => "Float64",
        _ => "an unknown",
    };
    format!(
        "dense KV observed as {observed_dtype} dtype ({observed} bytes/element), expected the \
         {DENSE_KV_COMPUTE_DTYPE:?} compute dtype ({DENSE_KV_COMPUTE_ELEMENT_BYTES} \
         bytes/element); the loader promoted the cache (sc-20671), so this row is refused rather \
         than recorded as a widened dense baseline"
    )
}

/// A deliberately conservative planning floor, not a proven process peak. The two-times
/// checkpoint load reservation matches this model type's load admission; the pinned projection
/// metadata supplies role-specific dense KV width. The live KV is the total-live bound's (a stored
/// prefix alongside one full request, which is all the prefix path holds since sc-20671). The
/// request's prefill activations — one decoder layer's projections, MLP tensors and residuals
/// across the whole request, priced exactly as the product's tiled request admission prices them
/// ([`core_llm::tiled_prefill_activation_bytes`]) — scale with the request like the KV does and
/// were missing before sc-20671 (~12 GiB for the 3B bf16 reference at the fit boundary). The
/// actual lazy graph may retain more, especially at long context, so the floor is only a
/// pre-spawn refusal (a row whose floor already exceeds the child cap cannot fit) and the
/// estimate recorded in the receipt. Admission itself is runtime-guarded: see
/// [`runtime_guarded_admission`].
pub(crate) fn static_role_footprint_budget(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
) -> Result<u64, String> {
    let payload = spec
        .required_files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .try_fold(0_u64, |sum, file| sum.checked_add(file.bytes))
        .ok_or("pinned checkpoint payload overflows")?;
    if payload == 0 {
        return Err("pinned model has no checkpoint payload".into());
    }
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(snapshot.join("config.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let positive = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("pinned model lacks positive {key} for footprint preflight"))
    };
    if config
        .get("torch_dtype")
        .and_then(serde_json::Value::as_str)
        != Some("bfloat16")
    {
        return Err("pinned model loader input dtype is not BF16".into());
    }
    let layers = positive("num_hidden_layers")?;
    let kv_heads = positive("num_key_value_heads")?;
    let head_dim = positive("head_dim")?;
    let query_heads = positive("num_attention_heads")?;
    let hidden_size = positive("hidden_size")?;
    let intermediate_size = positive("intermediate_size")?;
    let vocab_size = positive("vocab_size")?;
    let element_bytes = pinned_dense_kv_element_bytes(spec, snapshot, layers)?;
    let kv = dense_kv_bytes(
        1,
        layers,
        kv_heads,
        total_live_tokens,
        head_dim,
        element_bytes,
    )?;
    let tile = [3_u64, 8, request_tokens, query_heads, 4]
        .into_iter()
        .try_fold(1_u64, |value, factor| value.checked_mul(factor))
        .ok_or("fused prefill tile footprint overflows")?;
    let activations = core_llm::tiled_prefill_activation_bytes(
        request_tokens,
        core_llm::LlmMemoryGeometry {
            query_heads,
            kv_heads,
            head_dim,
            layers,
            element_bytes,
            score_element_bytes: 4,
            hidden_size,
            intermediate_size,
            vocab_size,
            recurrent_bytes: 0,
        },
    )
    .ok_or("prefill activation footprint overflows")?;
    payload
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(kv))
        .and_then(|bytes| bytes.checked_add(tile))
        .and_then(|bytes| bytes.checked_add(activations))
        .ok_or("static model-load plus KV/working budget overflows".into())
}

fn static_row_footprint_budget(
    coordinate: &Coordinate,
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
    policy: &CampaignSafetyPolicy,
) -> Result<u64, String> {
    let candidate = static_role_footprint_budget(
        benchmark_model(coordinate.family, false)?,
        candidate_snapshot,
        total_live_tokens,
        request_tokens,
    )?;
    let reference = static_role_footprint_budget(
        benchmark_model(coordinate.family, true)?,
        reference_snapshot,
        total_live_tokens,
        request_tokens,
    )?;
    let required = candidate.max(reference);
    if required > policy.child_footprint_cap_bytes {
        return Err(format!(
            "{} needs at least {required} bytes of source-based model-load plus KV/working budget; child footprint cap {} refuses before spawn",
            coordinate_slug(coordinate), policy.child_footprint_cap_bytes,
        ));
    }
    Ok(required)
}

fn static_row_requirements(
    coordinate: &Coordinate,
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    prompt: &str,
    policy: &CampaignSafetyPolicy,
    compressed: bool,
) -> Result<(u64, u64, u64), String> {
    let candidate = benchmark_model(coordinate.family, false)?;
    let reference = benchmark_model(coordinate.family, true)?;
    let candidate_tokens = preflight_total_live_tokens(
        candidate_snapshot,
        candidate,
        coordinate,
        prompt,
        compressed,
    )?;
    let reference_tokens = preflight_total_live_tokens(
        reference_snapshot,
        reference,
        coordinate,
        prompt,
        compressed,
    )?;
    let total = candidate_tokens.0.max(reference_tokens.0);
    let max_request = candidate_tokens.1.max(reference_tokens.1);
    if max_request > policy.max_request_tokens {
        return Err(format!(
            "{} requires actual request ceiling {max_request}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    if total > policy.max_context_tokens {
        return Err(format!(
            "{} requires conservative total-live ceiling {total}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    let known_footprint_budget = static_row_footprint_budget(
        coordinate,
        candidate_snapshot,
        reference_snapshot,
        total,
        max_request,
        policy,
    )?;
    Ok((total, max_request, known_footprint_budget))
}

/// The live tokens one noise-floor worker holds at its peak: the live-token bound of a compressed
/// row's same-weights dense reference arm, which runs exactly nf's passes (the kernel forced
/// continuation to 1024 tokens or the window, and the multi-turn fixture whose stored turn 1 sits
/// beside turn 2's request cache through a 1024-token forced turn-2 continuation). That bound
/// already counts the stored prefix beside the request, so nothing is stacked on it.
fn noise_floor_live_tokens(
    snapshot: &Path,
    spec: &BenchmarkModelSpec,
    coordinate: &Coordinate,
    prompt: &str,
) -> Result<(u64, u64), String> {
    preflight_total_live_tokens(snapshot, spec, coordinate, prompt, true)
}

/// The static footprint of one noise-floor worker: one dense candidate session over
/// [`noise_floor_live_tokens`]: model load, that live K/V, the prefill tile and activations.
fn noise_floor_footprint_budget(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
) -> Result<u64, String> {
    static_role_footprint_budget(spec, snapshot, total_live_tokens, request_tokens)
}

/// The requirements of one noise-floor row: what its worker actually holds, one dense session of
/// the candidate at a time (never the bf16 reference, which it does not load) over
/// [`noise_floor_live_tokens`]. The chunked-prefill control's cache is sized once
/// ([`crate::provider`]), so that one-session budget prices every pass of the row.
fn noise_floor_row_requirements(
    coordinate: &Coordinate,
    snapshot: &Path,
    prompt: &str,
    policy: &CampaignSafetyPolicy,
) -> Result<(u64, u64, u64), String> {
    let spec = benchmark_model(coordinate.family, false)?;
    let (total, max_request) = noise_floor_live_tokens(snapshot, spec, coordinate, prompt)?;
    if max_request > policy.max_request_tokens || total > policy.max_context_tokens {
        return Err(format!(
            "{} requires request ceiling {max_request} and total-live ceiling {total}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    let budget = noise_floor_footprint_budget(spec, snapshot, total, max_request)?;
    if budget > policy.child_footprint_cap_bytes {
        return Err(format!(
            "{} needs at least {budget} bytes of one dense session; child footprint cap {} refuses before spawn",
            coordinate_slug(coordinate),
            policy.child_footprint_cap_bytes,
        ));
    }
    Ok((total, max_request, budget))
}

/// The runtime admission a row ran under. It is recorded in every receipt so the stated child cap,
/// the static planning estimate, and the [`campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE`]
/// decision (rule, estimate source and bytes, host measurement) travel with the evidence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptAdmission {
    pub mode: String,
    pub rule: String,
    pub child_footprint_cap_bytes: u64,
    pub host_free_reserve_bytes: u64,
    pub static_footprint_floor_bytes: u64,
    /// Where [`Self::estimate_bytes`] came from (see [`campaign_supervisor::AdmissionEstimate`]).
    pub estimate_source: String,
    /// The unit's estimated peak: admission required host available memory of at least this plus
    /// the reserve. Never below the static floor, never above the cap.
    pub estimate_bytes: u64,
    /// Every vm_stat component of the pre-spawn host measurement the row was admitted (or
    /// refused) on; `availableBytes` is the measure the rule compared. Absent only from a
    /// parent-side static admission computed before any measurement; a receipt requires it (see
    /// [`Self::validate_admitted`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host_memory_components: Option<campaign_supervisor::HostMemory>,
}

impl ReceiptAdmission {
    /// The recorded cap and reserve must be the captured campaign policy's, not caller values.
    pub fn validate_against(&self, policy: &CampaignSafetyPolicy) -> Result<(), String> {
        self.validate_admitted()?;
        if self.child_footprint_cap_bytes != policy.child_footprint_cap_bytes
            || self.host_free_reserve_bytes != policy.host_free_reserve_bytes
        {
            return Err("row admission cap/reserve differs from the captured safety policy".into());
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.mode != RUNTIME_GUARDED_ADMISSION
            || self.rule != campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE
            || self.child_footprint_cap_bytes == 0
            || self.host_free_reserve_bytes == 0
            || self
                .host_free_reserve_bytes
                .checked_add(self.child_footprint_cap_bytes)
                .is_none()
            || self.static_footprint_floor_bytes == 0
            || self.static_footprint_floor_bytes > self.child_footprint_cap_bytes
            || self.estimate_bytes < self.static_footprint_floor_bytes
            || self
                .estimate()
                .validate(self.child_footprint_cap_bytes)
                .is_err()
        {
            return Err(format!(
                "row admission is not a runtime-guarded cap with its {} decision",
                campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE
            ));
        }
        if let Some(host) = &self.host_memory_components {
            host.validate()?;
        }
        Ok(())
    }

    /// A receipt's admission: the row ran, so its recorded host measurement must exist,
    /// recompute, and cover estimate plus reserve.
    pub fn validate_admitted(&self) -> Result<(), String> {
        self.validate()?;
        self.host_memory_components
            .as_ref()
            .ok_or("row admission lacks its host memory components")?
            .validate_admits(self.host_free_reserve_bytes, self.estimate_bytes)
    }

    /// The estimate the supervisor admits the row on.
    pub fn estimate(&self) -> campaign_supervisor::AdmissionEstimate {
        campaign_supervisor::AdmissionEstimate {
            source: self.estimate_source.clone(),
            bytes: self.estimate_bytes,
        }
    }

    /// The same admission with the supervisor's pre-spawn measurement (for unaccepted records).
    pub fn with_host_memory(&self, host: Option<campaign_supervisor::HostMemory>) -> Self {
        Self {
            host_memory_components: host,
            ..self.clone()
        }
    }
}

/// Admit a row by its runtime guards rather than by a static whole-process MLX peak proof, which is
/// unobtainable for lazy long-context graphs. The row is admitted only when the mandatory policy
/// configures every supervisor guard (deadline, sampling, termination grace, host RAM reserve,
/// child `phys_footprint` watchdog cap) and the conservative static floor still fits under the
/// cap. The floor is the row's admission estimate (`estimate_source` names it): the supervisor
/// refuses before spawn unless host available RAM ([`campaign_supervisor::HostMemory`]) covers
/// estimate plus reserve, and terminates the child if the cap or reserve watchdog trips; neither
/// outcome is ever an accepted row. No SC-20671/76/77 unit has an already-completed identical
/// unit to raise the estimate from (each runs once per resume directory, and resume skips accepted
/// units), so no measured peak enters it.
pub(crate) fn runtime_guarded_admission(
    policy: &CampaignSafetyPolicy,
    static_footprint_floor_bytes: u64,
    estimate_source: &str,
) -> Result<ReceiptAdmission, String> {
    policy.validate()?;
    let estimate = campaign_supervisor::AdmissionEstimate::resolve(
        Some((estimate_source, static_footprint_floor_bytes)),
        None,
        policy.child_footprint_cap_bytes,
    )?;
    let admission = ReceiptAdmission {
        mode: RUNTIME_GUARDED_ADMISSION.into(),
        rule: campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE.into(),
        child_footprint_cap_bytes: policy.child_footprint_cap_bytes,
        host_free_reserve_bytes: policy.host_free_reserve_bytes,
        static_footprint_floor_bytes,
        estimate_source: estimate.source,
        estimate_bytes: estimate.bytes,
        host_memory_components: None,
    };
    admission.validate()?;
    Ok(admission)
}

/// Worker side: the runtime-guarded admission plus the decision its supervisor admitted it on
/// (handed over in [`campaign_supervisor::HOST_MEMORY_ADMISSION_ENV`]), so the worker's own
/// receipt records every component of the decision. The supervisor's estimate must be exactly the
/// one the worker derives from the same sealed inputs.
pub(crate) fn supervised_admission(
    policy: &CampaignSafetyPolicy,
    static_footprint_floor_bytes: u64,
    estimate_source: &str,
) -> Result<ReceiptAdmission, String> {
    let admission =
        runtime_guarded_admission(policy, static_footprint_floor_bytes, estimate_source)?;
    let decision = campaign_supervisor::admitted_host(
        policy.host_free_reserve_bytes,
        policy.child_footprint_cap_bytes,
    )?;
    if decision.estimate() != admission.estimate() {
        return Err(format!(
            "supervisor admitted the worker on estimate {} bytes ({}), not the worker's own {} bytes ({})",
            decision.estimate_bytes,
            decision.estimate_source,
            admission.estimate_bytes,
            admission.estimate_source
        ));
    }
    let admission = admission.with_host_memory(Some(decision.host_memory));
    admission.validate_admitted()?;
    Ok(admission)
}

/// First attempt prefix whose stdout, stderr, and unaccepted record are all absent. The supervisor
/// removes its logs when a spawn fails, so the record alone must also reserve a prefix.
pub(crate) fn unused_attempt_prefix(logs: &Path, slug: &str) -> Option<String> {
    (0_u64..)
        .map(|attempt| format!("{slug}.attempt-{attempt}"))
        .find(|prefix| {
            ["stdout.log", "stderr.log", "unaccepted.json"]
                .iter()
                .all(|suffix| !logs.join(format!("{prefix}.{suffix}")).exists())
        })
}

/// Why a row stopped: reason, detail, the owned child PID (after spawn), and the live host sample
/// that tripped the host-reserve watchdog, when one did.
pub(crate) type UnacceptedStop<'a> = (
    &'a str,
    &'a str,
    Option<u32>,
    Option<&'a campaign_supervisor::HostMemory>,
);

/// Record an unaccepted row and return the row's own failure. A record-write failure is appended
/// to, never substituted for, the original refusal/abort/exit reason.
pub(crate) fn unaccepted_row_error(
    path: &Path,
    kind: &str,
    slug: &str,
    admission: &ReceiptAdmission,
    stop: UnacceptedStop<'_>,
    failure: String,
) -> String {
    match write_unaccepted_row_record(path, kind, slug, admission, stop) {
        Ok(()) => format!("{failure}; not accepted ({})", path.display()),
        Err(error) => format!(
            "{failure}; not accepted, and its record {} could not be written: {error}",
            path.display()
        ),
    }
}

/// Persist a refused (pre-spawn), aborted (post-spawn guard), or failed row as a sealed, explicitly
/// unaccepted log record. It never becomes a receipt and the resume validator never reads it.
pub(crate) fn write_unaccepted_row_record(
    path: &Path,
    kind: &str,
    slug: &str,
    admission: &ReceiptAdmission,
    stop: UnacceptedStop<'_>,
) -> Result<(), String> {
    let (reason, detail, pid, watchdog_host_memory) = stop;
    let mut record = serde_json::json!({
        "schemaVersion": 1,
        "kind": kind,
        "coordinate": slug,
        "accepted": false,
        "outcome": match (reason, pid) {
            ("ChildExit", _) => "failed",
            (_, None) => "refused",
            (_, Some(_)) => "aborted",
        },
        "reason": reason,
        "detail": detail,
        "pid": pid,
        "admission": admission,
    });
    if let Some(sample) = watchdog_host_memory {
        record["watchdogHostMemory"] = serde_json::to_value(sample).map_err(|e| e.to_string())?;
    }
    let bytes = seal_json(&record)?.0;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("write unaccepted row record: {e}"))?;
    std::io::Write::write_all(&mut file, &bytes).map_err(|e| e.to_string())
}

fn valid_locked_mlx_fields(version: &str, source: &str, revision: &str) -> bool {
    !version.is_empty()
        && revision.len() == 40
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && source == format!("git+{PMETAL_MLX_REPOSITORY}?rev={revision}#{revision}")
}

fn valid_locked_mlx_identity(provenance: &ReceiptProvenance) -> bool {
    valid_locked_mlx_fields(
        &provenance.mlx_version,
        &provenance.mlx_source,
        &provenance.mlx_revision,
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptProvenance {
    pub scene_works_repository: String,
    pub inference_repository: String,
    pub scene_works_revision: String,
    pub inference_revision: String,
    pub mlx_version: String,
    pub mlx_source: String,
    pub mlx_revision: String,
    pub dependency_lock_sha256: String,
    pub os: String,
    pub xcode: String,
    pub hardware: String,
    pub model_id: String,
    pub model_file_sha256: String,
    pub model_file_bytes: u64,
    pub reference_model_id: String,
    pub reference_model_sha256: String,
    pub reference_model_bytes: u64,
    /// The row-start energy mode (one of [`POWER_MODES`]).
    pub power_mode: String,
    /// The row-start thermal state: never throttled, because a throttled start refuses the row.
    pub thermal_state: String,
    /// Power mode and thermal state observed at [`HOST_STATE_BOUNDARIES`], in order.
    pub host_states: Vec<ReceiptHostState>,
    /// Recorded, never refused: after row start the thermal state changed or throttled (row end
    /// or any timing sample).
    pub thermal_changed_during_row: bool,
    /// Recorded, never refused: after row start the power mode changed.
    pub power_mode_changed_during_row: bool,
    pub command_template: String,
    pub command: String,
    pub campaign_session_id: String,
    pub campaign_cache_state_version: u64,
    pub coordinate_operation_sha256: String,
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
    pub context_window_tokens: u64,
    pub context_target_tokens: u64,
    pub context_payload_tokens: u64,
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

/// Compressed KV representations the SC-20671 harness can run as a `compressed` row. The
/// producer, receipt, and validators are method-agnostic: a further candidate (for example the
/// SC-20677 RVQ/RaBitQ caches) is one variant here plus its cache selection and kernel parity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressedKvMethod {
    /// SC-20675 packed 2-bit group-affine cache read by the SC-20676 fused Metal kernel. Its
    /// identifier stays `group-affine` so earlier receipts and resume identities keep binding.
    GroupAffine,
    /// The same cache and reader with 4-bit (KIVI-style) codes, group 32.
    GroupAffine4,
    /// The same cache and reader with 8-bit codes, group 32.
    GroupAffine8,
}

impl CompressedKvMethod {
    pub const ALL: [Self; 3] = [Self::GroupAffine, Self::GroupAffine4, Self::GroupAffine8];

    /// Stable receipt/CLI identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::GroupAffine => "group-affine",
            Self::GroupAffine4 => "group-affine-4",
            Self::GroupAffine8 => "group-affine-8",
        }
    }

    /// Code width of this method's packed cache.
    pub fn code_bits(self) -> crate::primitives::PackedCodeBits {
        match self {
            Self::GroupAffine => crate::primitives::PackedCodeBits::Two,
            Self::GroupAffine4 => crate::primitives::PackedCodeBits::Four,
            Self::GroupAffine8 => crate::primitives::PackedCodeBits::Eight,
        }
    }

    /// Representation identity this method's cache and reader carry (recorded on receipts).
    pub fn representation_identity(self) -> &'static str {
        crate::primitives::packed_metal_identity(self.code_bits())
    }

    pub fn parse(value: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|method| method.id() == value)
            .ok_or_else(|| {
                format!(
                    "unknown compressed KV method {value:?}; expected one of {:?}",
                    Self::ALL.map(Self::id)
                )
            })
    }

    /// GPU-family tuning profile of this method's fused reader, recorded on every compressed
    /// receipt. No device-family detector exists in this crate; none is needed because MLX's
    /// Metal backend runs only on Apple silicon, whose oldest Mac GPU (M1) is Apple family 7, so
    /// any device this arm can dispatch on qualifies for the recent-family profile.
    pub fn kernel_gpu_family(self) -> crate::primitives::PackedMetalGpuFamily {
        match self {
            Self::GroupAffine | Self::GroupAffine4 | Self::GroupAffine8 => {
                crate::primitives::PackedMetalGpuFamily::Apple7OrNewer
            }
        }
    }

    /// Build this method's retained fused reader once per campaign session.
    pub fn arm(self) -> Result<CompressedKvArm, String> {
        let reader = match self {
            Self::GroupAffine | Self::GroupAffine4 | Self::GroupAffine8 => {
                let kernel = crate::primitives::PackedMetalKernel::for_identity_family_and_bits(
                    self.representation_identity(),
                    self.kernel_gpu_family(),
                    self.code_bits(),
                )
                .map_err(|e| e.to_string())?;
                // The single-threaded campaign worker owns this non-Send Metal object through the
                // cache handle's `Arc`, exactly as the SC-20676 evidence worker does.
                #[allow(clippy::arc_with_non_send_sync)]
                let kernel = std::sync::Arc::new(kernel);
                crate::primitives::CompiledKernelHandle::new(kernel)
            }
        };
        Ok(CompressedKvArm {
            method: self,
            reader,
        })
    }
}

/// The compressed arm of one campaign session: its method and retained fused reader.
pub struct CompressedKvArm {
    method: CompressedKvMethod,
    reader: crate::primitives::CompiledKernelHandle,
}

impl CompressedKvArm {
    /// Test-only arm bound to an arbitrary retained reader (e.g. one the cache refuses to bind).
    #[cfg(test)]
    pub(crate) fn with_reader(
        method: CompressedKvMethod,
        reader: crate::primitives::CompiledKernelHandle,
    ) -> Self {
        Self { method, reader }
    }

    pub fn method(&self) -> CompressedKvMethod {
        self.method
    }

    /// The decoder cache for one compressed request, chosen before any K/V mutation. A refusal is
    /// returned as a dense route with its reason, which the caller must record.
    pub(crate) fn select_cache(
        &self,
        model: &crate::models::CausalLm,
    ) -> crate::primitives::DecoderCacheSelection {
        match self.method {
            CompressedKvMethod::GroupAffine
            | CompressedKvMethod::GroupAffine4
            | CompressedKvMethod::GroupAffine8 => {
                model.select_cache_with_packed_reader(self.reader.clone(), 1, 1, false)
            }
        }
    }

    /// Kernel parity of this method's fused reader against an independent host-fp32
    /// dequantize-then-attend reference over the exact stored codes.
    pub fn kernel_parity_errors(&self) -> Result<Vec<f64>, String> {
        match self.method {
            CompressedKvMethod::GroupAffine
            | CompressedKvMethod::GroupAffine4
            | CompressedKvMethod::GroupAffine8 => {
                crate::primitives::group_affine_kernel_fp32_parity_errors(&self.reader)
            }
        }
    }
}

/// A campaign-only loaded product session. The provider and prefix cache remain resident across
/// warmup and measured repeats; ordinary serving continues to use the provider directly.
pub struct CampaignSession {
    provider: crate::provider::LlamaProvider,
    inventory: SnapshotInventory,
    family: &'static str,
    load_elapsed_ms: f64,
    load_start_sample: MemorySample,
    weights_loaded_sample: MemorySample,
    model_weights_bytes: u64,
    session_id: String,
    cache_state_version: Cell<u64>,
    /// `Some` for a compressed row's measured arm: every observed decode runs on this method's
    /// compressed cache. `None` for dense rows and the same-weights dense-KV reference arm.
    compressed: Option<CompressedKvArm>,
}

impl CampaignSession {
    /// Load the compressed arm of a compressed row: the product session with `method`'s retained
    /// fused reader. The reader is built after the weights-loaded sample so its (small) retained
    /// state is not attributed to model weights.
    pub fn load_compressed(
        snapshot: impl AsRef<Path>,
        method: CompressedKvMethod,
    ) -> core_llm::Result<Self> {
        let mut session = Self::load(snapshot)?;
        session.compressed = Some(method.arm().map_err(core_llm::Error::Load)?);
        Ok(session)
    }

    pub fn compressed(&self) -> Option<&CompressedKvArm> {
        self.compressed.as_ref()
    }

    /// Record, on the compressed arm only, that a product operation runs on the explicit dense
    /// path because the compressed representation has no route for it.
    fn dense_fallback(&self, observer: &mut dyn Observer, operation: &str, reason: &str) {
        if self.compressed.is_some() {
            observer.dense_fallback(operation, reason);
        }
    }

    pub fn load(snapshot: impl AsRef<Path>) -> core_llm::Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT_SESSION_GENERATION: AtomicU64 = AtomicU64::new(1);
        mlx_rs::memory::clear_cache();
        mlx_rs::memory::reset_peak_memory();
        let load_start_sample = sample_memory(std::process::id())
            .map_err(|e| core_llm::Error::Load(format!("campaign load-start sample: {e}")))?;
        let load_started = std::time::Instant::now();
        let inventory = inventory_snapshot(snapshot.as_ref())
            .map_err(|e| core_llm::Error::Load(format!("snapshot inventory: {e}")))?;
        let provider = crate::provider::LlamaProvider::load_for_campaign(
            &core_llm::LoadSpec::dense(snapshot.as_ref().to_string_lossy().to_string()),
        )?;
        let weights_loaded_sample = sample_memory(std::process::id())
            .map_err(|e| core_llm::Error::Load(format!("campaign weights-loaded sample: {e}")))?;
        let model_weights_bytes =
            measured_model_weight_bytes(&load_start_sample, &weights_loaded_sample)
                .map_err(core_llm::Error::Load)?;
        let family = provider.campaign_family()?;
        let load_elapsed_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        if !load_elapsed_ms.is_finite() || load_elapsed_ms <= 0.0 {
            return Err(core_llm::Error::Load(
                "campaign session did not measure a positive snapshot load duration".into(),
            ));
        }
        let generation = NEXT_SESSION_GENERATION.fetch_add(1, Ordering::Relaxed);
        let session_id = seal_bytes(
            format!(
                "{}:{}:{}:{}",
                inventory.sha256,
                inventory.bytes,
                std::process::id(),
                generation,
            )
            .as_bytes(),
        );
        Ok(Self {
            provider,
            inventory,
            family,
            load_elapsed_ms,
            load_start_sample,
            weights_loaded_sample,
            model_weights_bytes,
            session_id,
            cache_state_version: Cell::new(0),
            compressed: None,
        })
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }
    pub fn inventory(&self) -> &SnapshotInventory {
        &self.inventory
    }
    fn model_weights_bytes(&self) -> u64 {
        self.model_weights_bytes
    }
    fn validate_coordinate_family(&self, coordinate: &Coordinate) -> core_llm::Result<()> {
        if self.family != coordinate.family {
            return Err(core_llm::Error::InvalidRequest(format!(
                "loaded {family} architecture cannot produce {coordinate_family} coordinate",
                family = self.family,
                coordinate_family = coordinate.family,
            )));
        }
        Ok(())
    }
    fn advance_cache_state(&self) -> u64 {
        let next = self.cache_state_version.get().saturating_add(1);
        self.cache_state_version.set(next);
        next
    }
    pub fn cache_state_version(&self) -> u64 {
        self.cache_state_version.get()
    }
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
    /// [`post_release_mlx_slack_bytes`] of the weights-loaded MLX active bytes.
    pub mlx_active_tolerance_bytes: u64,
    /// [`post_release_mlx_slack_bytes`] of the weights-loaded MLX cache bytes.
    pub mlx_cache_tolerance_bytes: u64,
    /// Post-release MLX active bytes above the weights-loaded boundary (0 when at or below it).
    pub mlx_active_residual_bytes: u64,
    /// Post-release MLX cache bytes above the weights-loaded boundary (0 when at or below it).
    pub mlx_cache_residual_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptPeakWindow {
    pub started_at: String,
    pub baseline_active_bytes: u64,
    pub reset_peak_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptMemory {
    pub model_weights_bytes: u64,
    pub persistent_kv_bytes: u64,
    pub transient_workspace_bytes: u64,
    pub dense_theoretical_kv_bytes: u64,
    pub prefill_peak_window: ReceiptPeakWindow,
    pub phase_samples: Vec<ReceiptPhase>,
    pub allocation_events: Vec<ReceiptAllocation>,
    pub reconciliation: ReceiptReconciliation,
    pub release: ReceiptRelease,
    pub admission: ReceiptAdmission,
    /// Dense KV bytes as a share of the prefill-peak process footprint, in basis points (floor).
    pub dense_kv_share_bps: u64,
    /// A memory-material row whose dense KV share is below
    /// [`MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS`]. The band is defined by geometry, so this is a
    /// recorded flag, never a refusal.
    pub below_memory_material_share: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptTimingSample {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    /// `steady_decode_timed_tokens * 1000 / steady_decode_ms` of this repeat's dedicated
    /// fixed-length steady decode ([`STEADY_DECODE_TOKENS`]).
    pub decode_tokens_per_second: f64,
    pub steady_decode_prompt_tokens: u64,
    pub steady_decode_generated_tokens: u64,
    pub steady_decode_timed_tokens: u64,
    pub steady_decode_ms: f64,
    pub steady_decode_forced_stop_tokens: u64,
    /// Host power/thermal state right after this repeat, so throughput drift is attributable.
    pub host_state: ReceiptHostState,
}

impl ReceiptTimingSample {
    fn steady_decode(&self) -> SteadyDecodeMeasurement {
        SteadyDecodeMeasurement {
            prompt_tokens: self.steady_decode_prompt_tokens,
            generated_tokens: self.steady_decode_generated_tokens,
            timed_tokens: self.steady_decode_timed_tokens,
            decode_ms: self.steady_decode_ms,
            forced_stop_tokens: self.steady_decode_forced_stop_tokens,
        }
    }

    /// The sample's throughput is exactly its recorded fixed-length steady decode, and that decode
    /// fit the loaded native context.
    fn validate_steady_decode(&self, context_window_tokens: u64) -> Result<(), String> {
        let steady = self.steady_decode();
        let derived = steady.tokens_per_second()?;
        if (self.decode_tokens_per_second - derived).abs() > derived * 1e-9
            || steady.prompt_tokens.saturating_add(steady.generated_tokens) > context_window_tokens
        {
            return Err(
                "timing sample throughput does not derive from its fixed-length steady decode"
                    .into(),
            );
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCompileProbeEvidence {
    pub index: u64,
    pub operation: String,
    pub source: String,
    pub matrix_coordinate: String,
    pub setup_ms: f64,
    pub dispatch_ms: f64,
    pub operation_evidence_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCompileAttribution {
    pub method: String,
    pub operation: String,
    pub source: String,
    pub probe_durations_ms: Vec<f64>,
    pub probe_evidence: Vec<ReceiptCompileProbeEvidence>,
    pub first_dispatch_ms: f64,
    pub steady_dispatch_ms: f64,
    /// `first - steady`; any finite sign under [`COMPILE_ATTRIBUTION_METHOD`].
    pub first_dispatch_excess_ms: f64,
    /// Steady-state dispatch durations the noise band is read from: the four post-first measured
    /// repeats of a cold row, or the five measured repeats that follow a warm row's warmups.
    /// Required, like the band and resolution below (`Option` only for construction).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_samples_ms: Option<Vec<f64>>,
    /// `max - min` of [`Self::noise_samples_ms`]: run-to-run spread of the same dispatch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_band_ms: Option<f64>,
    /// Whether the first-dispatch excess clears the noise band. At large geometries compute
    /// dominates and compile cost is below run-to-run noise, so this is recorded, never required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_cost_resolved: Option<bool>,
    /// The excess, present exactly when it is resolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_cost_ms: Option<f64>,
    /// [`COMPILE_COST_NOT_SLOWER`] or [`COMPILE_COST_WITHIN_NOISE`], present exactly when the
    /// excess is unresolved.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compile_cost_unresolved_reason: Option<String>,
}

/// Compile attribution that records the first-dispatch excess against a measured noise band.
pub const COMPILE_ATTRIBUTION_METHOD: &str = "first-dispatch-minus-steady-v2";
/// Unresolved: the first dispatch was not slower than steady state.
pub const COMPILE_COST_NOT_SLOWER: &str = "first-dispatch-not-slower-than-steady";
/// Unresolved: the positive excess does not exceed the steady noise band.
pub const COMPILE_COST_WITHIN_NOISE: &str = "excess-within-steady-noise-band";
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
    /// The resolved compile cost; `null` when it is below the steady noise band.
    pub cold_compile_ms: Option<f64>,
    pub warm_compile_ms: f64,
    pub compile_attribution: ReceiptCompileAttribution,
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
/// The contract v5 `statistics` block every receipt records verbatim.
pub fn contract_statistics() -> ReceiptQualityStatistics {
    ReceiptQualityStatistics {
        repeats: TIMING_REPEATS as u64,
        warmups: TIMING_WARMUPS as u64,
        quality_measured_once: QUALITY_MEASUREMENTS == 1,
        confidence_interval: "95% bootstrap".into(),
        outlier_policy: "report all samples; no silent deletion".into(),
        variance_policy: "repeats and warmups are timing measurements (coordinate prefill, time to first token, steady decode); all raw timing repeats retained; decode throughput coefficient of variation must stay within the frozen maximum".into(),
        max_coefficient_of_variation: 0.05,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptQualityStatistics {
    pub repeats: u64,
    pub warmups: u64,
    pub quality_measured_once: bool,
    pub confidence_interval: String,
    pub outlier_policy: String,
    pub variance_policy: String,
    #[serde(rename = "maxCoefficientOfVariation")]
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
    /// Always [`GREEDY_AGREEMENT_METHOD`].
    #[serde(rename = "greedyAgreementMethod")]
    pub greedy_agreement_method: String,
    /// Observation only: the first position at which the free-running candidate and reference
    /// kernel streams of the primary repeat differ (`null` when identical).
    #[serde(rename = "freeRunningFirstDivergence")]
    pub free_running_first_divergence: Option<u64>,
    /// Teacher-forced greedy agreement of each measured repeat; `greedyTokenAgreement` is their
    /// minimum.
    #[serde(rename = "greedyTokenAgreementByRepeat")]
    pub greedy_token_agreement_by_repeat: Vec<f64>,
    #[serde(rename = "structuredToolAgreement")]
    pub structured_tool_agreement: f64,
    #[serde(rename = "needleRetrieval")]
    pub needle_retrieval: f64,
    /// False when the same-weights dense-KV run itself missed the needle: the needle check then
    /// measures only agreement with that dense output and cannot detect KV-induced retrieval loss.
    #[serde(rename = "needleDiscriminating")]
    pub needle_discriminating: bool,
    /// False when the same-weights dense-KV run did not emit the valid structured tool call: tool
    /// agreement then cannot detect KV-induced tool-call loss.
    #[serde(rename = "toolDiscriminating")]
    pub tool_discriminating: bool,
    #[serde(rename = "multiTurnPromptCache")]
    pub multi_turn_prompt_cache: f64,
    /// Always [`MULTI_TURN_PROMPT_CACHE_METHOD`] (quality contract v4).
    #[serde(rename = "multiTurnPromptCacheMethod")]
    pub multi_turn_prompt_cache_method: String,
    /// Observation only: the earliest position, over the measured repeats, at which the
    /// free-running candidate and reference turn-2 streams differ (`null` when identical).
    #[serde(rename = "multiTurnFreeRunningFirstDivergence")]
    pub multi_turn_free_running_first_divergence: Option<u64>,
    /// Observation only: the fewest leading turn-2 positions on which those streams agree.
    #[serde(rename = "multiTurnMatchedPrefixTokens")]
    pub multi_turn_matched_prefix_tokens: u64,
    /// The primary repeat's per-turn prompt-cache records of both arms.
    #[serde(rename = "multiTurnCache")]
    pub multi_turn_cache: ReceiptMultiTurnCache,
    /// Compressed rows only: the turn-2 forced continuation behind `multiTurnPromptCache`.
    #[serde(
        rename = "multiTurnForcedContinuation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub multi_turn_forced_continuation: Option<ReceiptForcedContinuation>,
    /// Compressed rows only: both sessions' turn records of that forced continuation.
    #[serde(
        rename = "multiTurnForcedPass",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub multi_turn_forced_pass: Option<MultiTurnForcedPass>,
    pub statistics: ReceiptQualityStatistics,
    #[serde(rename = "fixtureEvidence")]
    pub fixture_evidence: std::collections::BTreeMap<String, ReceiptFixture>,
    /// Compressed rows only: the frozen-threshold outcome of the row's quality measurement. A failed gate
    /// is a measured result for the Go/No-Go decision, never a relabelled pass or a refused row.
    /// Dense rows are characterization and carry none.
    #[serde(
        rename = "qualityGate",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub quality_gate: Option<ReceiptQualityGate>,
    /// Compressed rows only: the forced-continuation measurement behind `greedyTokenAgreement`.
    #[serde(
        rename = "forcedContinuation",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub forced_continuation: Option<ReceiptForcedContinuation>,
}
/// Both arms' prompt-cache records of the multi-turn fixture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptMultiTurnCache {
    pub candidate: MultiTurnPromptCacheTurns,
    pub reference: MultiTurnPromptCacheTurns,
}

/// Compressed rows: both sessions' turn records of the row's turn-2 forced-continuation pass
/// (the dense-KV reference stream and the compressed teacher-forced pass). Their turn-2 prompts
/// are the same token ids, so the compressed arm is forced on the stream of its own prompt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultiTurnForcedPass {
    pub reference: MultiTurnPromptCacheTurns,
    pub candidate: MultiTurnPromptCacheTurns,
}

impl MultiTurnForcedPass {
    /// Both passes served turn 2 by a cache hit, over the same turn-2 prompt.
    pub fn validate(&self) -> Result<(), String> {
        self.reference.validate("forced-pass reference")?;
        self.candidate.validate("forced-pass candidate")?;
        if self.reference.turn2.prompt_sha256 != self.candidate.turn2.prompt_sha256 {
            return Err(format!(
                "the turn-2 forced continuation's reference and candidate turn-2 prompts differ ({} vs {})",
                self.reference.turn2.prompt_sha256, self.candidate.turn2.prompt_sha256
            ));
        }
        Ok(())
    }
}
/// One frozen-threshold miss of one measured repeat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptQualityGateFailure {
    pub metric: String,
    pub fixture: String,
    pub repeat: u64,
    pub value: f64,
    pub threshold: f64,
    pub comparison: String,
}
/// The measured quality-gate outcome of a compressed receipt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptQualityGate {
    pub passed: bool,
    pub failures: Vec<ReceiptQualityGateFailure>,
}

/// Length of a compressed row's forced continuation. Teacher-forced agreement over the short
/// natural fixture streams (~7-21 tokens) turned the frozen 0.999 minimum into "zero flips"; over
/// this many positions it resolves one flip.
pub const FORCED_CONTINUATION_TOKENS: u64 = 1024;
/// Only a fit-boundary row may shorten its continuation (to the context window left after the
/// kernel prompt), never below the steady-decode reserve.
pub const FORCED_CONTINUATION_MIN_TOKENS: u64 = STEADY_DECODE_TOKENS;
/// How many flip positions a receipt records (the first ones, in order).
pub const FORCED_CONTINUATION_RECORDED_FLIPS: usize = 32;
pub const FORCED_CONTINUATION_METHOD: &str =
    "dense-kv-same-weights-greedy-continuation-eos-ignored-teacher-forced";

/// A compressed row's greedy agreement (quality contract v3 `greedyTokenAgreement`): the
/// same-weights dense-KV session greedily continues the kernel fixture prompt through every stop
/// token for `tokens` positions, and the compressed session, teacher-forced on that stream in one
/// pass, is compared by argmax at every position. The stream is deterministic, so it is measured
/// once per row and shared by every repeat.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptForcedContinuation {
    pub method: String,
    pub tokens: u64,
    pub matches: u64,
    pub agreement: f64,
    pub flip_count: u64,
    /// The first [`FORCED_CONTINUATION_RECORDED_FLIPS`] disagreeing positions, ascending.
    pub first_flip_positions: Vec<u64>,
    pub reference_stream_sha256: String,
    pub candidate_choices_sha256: String,
}

pub(crate) fn token_stream_sha256(tokens: &[i32]) -> String {
    seal_bytes(&serde_json::to_vec(tokens).unwrap_or_default())
}

/// Per-position argmax agreement of a teacher-forced pass with the stream it was forced on.
pub fn forced_continuation_evidence(
    reference: &[i32],
    choices: &[i32],
) -> Result<ReceiptForcedContinuation, String> {
    forced_continuation_evidence_with(reference, choices, FORCED_CONTINUATION_METHOD)
}

/// [`forced_continuation_evidence`] of the multi-turn fixture's turn-2 continuation (contract v4).
pub fn multi_turn_forced_continuation_evidence(
    reference: &[i32],
    choices: &[i32],
) -> Result<ReceiptForcedContinuation, String> {
    forced_continuation_evidence_with(reference, choices, MULTI_TURN_FORCED_CONTINUATION_METHOD)
}

fn forced_continuation_evidence_with(
    reference: &[i32],
    choices: &[i32],
    method: &str,
) -> Result<ReceiptForcedContinuation, String> {
    if reference.is_empty() || reference.len() != choices.len() {
        return Err(format!(
            "forced continuation has {} reference positions but {} teacher-forced choices",
            reference.len(),
            choices.len()
        ));
    }
    let flips = reference
        .iter()
        .zip(choices)
        .enumerate()
        .filter(|(_, (reference, choice))| reference != choice)
        .map(|(position, _)| position as u64)
        .collect::<Vec<_>>();
    let tokens = reference.len() as u64;
    let matches = tokens - flips.len() as u64;
    Ok(ReceiptForcedContinuation {
        method: method.into(),
        tokens,
        matches,
        agreement: matches as f64 / tokens as f64,
        flip_count: flips.len() as u64,
        first_flip_positions: flips
            .into_iter()
            .take(FORCED_CONTINUATION_RECORDED_FLIPS)
            .collect(),
        reference_stream_sha256: token_stream_sha256(reference),
        candidate_choices_sha256: token_stream_sha256(choices),
    })
}

/// Internal consistency of a recorded forced continuation for a row of `context_band`.
fn validate_forced_continuation(
    continuation: &ReceiptForcedContinuation,
    context_band: &str,
) -> Result<(), String> {
    validate_forced_continuation_with(
        continuation,
        context_band == "fit-boundary",
        FORCED_CONTINUATION_METHOD,
    )
}

/// `shortened`: whether the continuation may stop short of [`FORCED_CONTINUATION_TOKENS`] (the
/// kernel continuation of a fit-boundary row). The multi-turn fixture is sized so its turn-2
/// continuation always has the full length.
fn validate_forced_continuation_with(
    continuation: &ReceiptForcedContinuation,
    shortened: bool,
    method: &str,
) -> Result<(), String> {
    let digest = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    let length_valid = continuation.tokens == FORCED_CONTINUATION_TOKENS
        || (shortened
            && (FORCED_CONTINUATION_MIN_TOKENS..FORCED_CONTINUATION_TOKENS)
                .contains(&continuation.tokens));
    let recorded = usize::try_from(continuation.flip_count)
        .unwrap_or(usize::MAX)
        .min(FORCED_CONTINUATION_RECORDED_FLIPS);
    if continuation.method != method
        || !length_valid
        || continuation.matches > continuation.tokens
        || continuation.flip_count != continuation.tokens - continuation.matches
        || continuation.agreement.to_bits()
            != (continuation.matches as f64 / continuation.tokens as f64).to_bits()
        || continuation.first_flip_positions.len() != recorded
        || continuation
            .first_flip_positions
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || continuation
            .first_flip_positions
            .last()
            .is_some_and(|last| *last >= continuation.tokens)
        || !digest(&continuation.reference_stream_sha256)
        || !digest(&continuation.candidate_choices_sha256)
    {
        return Err(format!(
            "forced continuation evidence is inconsistent: method={}, tokens={} (shortened allowed: {shortened}), matches={}, agreement={}, flipCount={}, recordedFlips={}",
            continuation.method,
            continuation.tokens,
            continuation.matches,
            continuation.agreement,
            continuation.flip_count,
            continuation.first_flip_positions.len()
        ));
    }
    Ok(())
}

/// Re-tokenization slack of turn 1's decoded answer when turn 2 renders it (preflight bound).
const MULTI_TURN_ANSWER_RETOKENIZATION_MARGIN: u64 = 32;
/// Tokens the multi-turn fixture keeps beside its payload, turn 1's answer, and the turn-2 forced
/// continuation: the prompt text and fixture framing, two turns of chat template, the follow-up,
/// and the answer's re-tokenization slack. Preflight and the provider still refuse a turn 2 that
/// leaves less than the full continuation.
pub const MULTI_TURN_PROMPT_RESERVE_TOKENS: u64 = 512;

/// The multi-turn fixture's payload target (contract v4): the band's own target, capped so turn 2
/// (turn 1's prompt, its at most 64-token fixture answer, the follow-up) plus the
/// [`FORCED_CONTINUATION_TOKENS`] turn-2 continuation fits the native window. Only the
/// fit-boundary band is capped; every row's coordinate operations keep the full band.
pub fn multi_turn_payload_target(native_context: u64, band_target: u64) -> Result<u64, String> {
    let cap = native_context
        .checked_sub(
            FORCED_CONTINUATION_TOKENS
                + u64::from(FIXTURE_MAX_NEW_TOKENS)
                + MULTI_TURN_PROMPT_RESERVE_TOKENS,
        )
        .filter(|cap| *cap >= 32)
        .ok_or_else(|| {
            format!("a {native_context}-token window cannot hold the multi-turn fixture")
        })?;
    Ok(band_target.min(cap))
}
/// Quality contract v4 multi-turn prompt-cache fixture: turn 2's user message. Turn 1 is the
/// frozen cache prompt; turn 2 is turn 1's conversation, its answer, and this follow-up.
pub const MULTI_TURN_FIXTURE_FOLLOW_UP: &str =
    "Now repeat that same stable baseline fact once more, in one short sentence.";
/// Contract v4 `multiTurnPromptCache`: teacher-forced per-position argmax agreement of turn 2,
/// served by a prompt-cache hit over turn 1, with the reference's turn-2 stream.
pub const MULTI_TURN_PROMPT_CACHE_METHOD: &str = "teacher-forced-turn-2-after-prompt-cache-hit";
/// The compressed rows' turn-2 forced continuation (the denominator of `multiTurnPromptCache`).
pub const MULTI_TURN_FORCED_CONTINUATION_METHOD: &str =
    "dense-kv-same-weights-turn-2-prompt-cache-hit-greedy-continuation-eos-ignored-teacher-forced";
/// Frozen compressed `multiTurnPromptCache` minimum (contract v4 `thresholds.multiTurnPromptCache`,
/// aligned with greedy agreement).
pub const COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN: f64 = 0.999;

/// One turn of the multi-turn prompt-cache fixture as the product prompt cache served it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct PromptCacheTurn {
    /// Rendered prompt tokens of the turn.
    pub prompt_tokens: u64,
    /// SHA-256 of the turn's rendered prompt token ids (prompt identity across arms and passes).
    pub prompt_sha256: String,
    /// Whether the turn's prompt-cache lookup hit.
    pub cache_hit: bool,
    /// Prompt tokens whose K/V the hit reused (zero on a miss).
    pub reused_prefix_tokens: u64,
}

/// Both turns of one multi-turn prompt-cache fixture run.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct MultiTurnPromptCacheTurns {
    pub turn1: PromptCacheTurn,
    pub turn2: PromptCacheTurn,
}

impl MultiTurnPromptCacheTurns {
    /// Turn 1 misses its fresh store; turn 2 is served by a hit that reuses part of its prompt
    /// (at most turn 1's stored prompt and answer). Anything else measured a cold turn 2.
    pub fn validate(&self, arm: &str) -> Result<(), String> {
        let (one, two) = (&self.turn1, &self.turn2);
        let digest = |value: &str| {
            value.len() == 64
                && value
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        };
        if !digest(&one.prompt_sha256)
            || !digest(&two.prompt_sha256)
            || one.prompt_sha256 == two.prompt_sha256
            || one.cache_hit
            || one.reused_prefix_tokens != 0
            || one.prompt_tokens == 0
            || !two.cache_hit
            || two.reused_prefix_tokens == 0
            || two.reused_prefix_tokens >= two.prompt_tokens
            || two.prompt_tokens <= one.prompt_tokens
            || two.reused_prefix_tokens > one.prompt_tokens + u64::from(FIXTURE_MAX_NEW_TOKENS)
        {
            return Err(format!(
                "{arm} multi-turn prompt cache did not serve turn 2 from turn 1's prefix: {self:?}"
            ));
        }
        Ok(())
    }
}

/// Leading positions on which two free-running streams agree.
fn matched_prefix_tokens(candidate: &[i32], reference: &[i32]) -> u64 {
    candidate
        .iter()
        .zip(reference)
        .take_while(|(candidate, reference)| candidate == reference)
        .count() as u64
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
    pub session_id: String,
    pub cache_state_version: u64,
}
/// One reasoned dense-fallback site of a compressed arm and how many times it executed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCompressionFallback {
    pub operation: String,
    pub reason: String,
    pub calls: u64,
}

/// One fused-reader kernel path of a compressed arm and the accepted calls that ran it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCompressionKernelPath {
    /// Kernel name (per-row split-KV, fp32 tiled, or NAX tiled).
    pub kernel: String,
    /// Selection token: why that kernel ran (`nax-selected`, `nax-unavailable`, `f32-query`, ...).
    pub selection: String,
    /// Human-readable reason for the selection.
    pub reason: String,
    /// Query dtype the reader was dispatched with (`float32`, `float16`, `bfloat16`).
    pub query_dtype: String,
    pub calls: u64,
}

/// Compressed-row evidence (SC-20676). Present exactly on `compressed` receipts; counts cover
/// every compressed-arm dispatch of the row (every timing run and the quality measurement).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptCompression {
    /// Campaign method parameter (`--kv-method`).
    pub method: String,
    /// Representation identity/version exported by the cache itself.
    pub representation_identity: String,
    pub representation_version: u64,
    pub bits: u64,
    pub quantization_group_size: u64,
    /// GPU-family tuning profile the fused reader was built with (`PackedMetalGpuFamily`).
    pub kernel_gpu_family: String,
    /// Every kernel path the fused reader actually ran, from its dispatch-time descriptor at the
    /// real query shape and dtype, sorted by (kernel, selection, query dtype); the calls sum to
    /// `fused_calls`, so a NAX and a non-NAX run can never produce the same receipt.
    pub kernel_paths: Vec<ReceiptCompressionKernelPath>,
    /// Peak live compressed storage of the coordinate operation's own cache, measured from its
    /// retained device arrays and its allocated host payload (host codes/metadata copy plus the
    /// staged key tail): `physicalKvBytes` is exactly their sum and is the whole physical
    /// representation a reduction claim is computed from. `memory.persistentKvBytes` is the MLX
    /// device share only and equals `deviceCodeBytes + deviceMetadataBytes`. All zero when the
    /// coordinate operation ran on an explicit dense fallback.
    pub device_code_bytes: u64,
    pub device_metadata_bytes: u64,
    pub host_payload_bytes: u64,
    pub physical_kv_bytes: u64,
    /// Live tokens of that measured storage; equals `geometry.kvLength` (zero on a dense fallback).
    pub storage_tokens: u64,
    /// Whether the coordinate operation (and only it: warmups, fixtures, prompt-cache reuse and
    /// cancellation are excluded) ran on the compressed representation or an explicit dense
    /// fallback. Only `compressed` rows are eligible for a persistent-KV reduction claim.
    pub persistent_kv_representation: String,
    /// Fused compressed-domain attention calls accepted by the cache (per layer and step).
    pub fused_calls: u64,
    /// Explicit dense fallbacks; always the sum of `fallbacks[].calls`.
    pub fallback_calls: u64,
    pub fallbacks: Vec<ReceiptCompressionFallback>,
    /// Packed-to-dense transitions that rebuilt the full history as dense K/V. E3 requires zero.
    pub full_cache_dequantizations: u64,
    pub failed_dispatches: u64,
    /// The KV path each measurement block of the row actually ran on (additive; absent on receipts
    /// produced before it). A supported-batch row's coordinate operation, which supplies memory,
    /// representation, prefill and first-token time, runs the explicit dense fallback, while its
    /// steady decode and quality run single-sequence on the compressed reader.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement_paths: Option<ReceiptMeasurementPaths>,
}

/// The KV path one receipt measurement block ran on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptMeasurementPath {
    /// `compressed` or `dense-fallback` (the persistent-KV representation tokens).
    pub kv_path: String,
    /// The product operation measured: the coordinate operation (`supported-batch`,
    /// `chunked-prefix-reuse`, `single-shot-generation`), `steady-decode`, or `quality`.
    pub operation: String,
    /// Sequences the measured dispatch decoded.
    pub sequences: u64,
}

/// Per-block KV paths of a compressed receipt ([`ReceiptCompression::measurement_paths`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptMeasurementPaths {
    /// `memory`, `compression` storage and `persistentKvRepresentation`: the coordinate operation.
    pub memory: ReceiptMeasurementPath,
    /// `timings.prefillMs`, `ttftMs` and `firstTokenMs`: the same coordinate operation.
    pub prefill_first_token: ReceiptMeasurementPath,
    /// `timings.decodeTokensPerSecond`: the fixed-length steady decode, always one sequence on
    /// the fused compressed reader (it fails closed otherwise).
    pub decode_timing: ReceiptMeasurementPath,
    /// `quality`: the single-request fixtures and the teacher-forced passes on the compressed
    /// reader (the forced passes fail closed off the fused reader).
    pub quality: ReceiptMeasurementPath,
}

/// The measurement paths a compressed row of `request_mode`/`prefill_mode` runs, given the
/// representation its coordinate operation recorded.
pub fn receipt_measurement_paths(
    request_mode: &str,
    prefill_mode: &str,
    persistent_kv_representation: &str,
) -> ReceiptMeasurementPaths {
    let (operation, sequences) = if request_mode == "supported-batch" {
        ("supported-batch", 2)
    } else if prefill_mode == "chunked" {
        ("chunked-prefix-reuse", 1)
    } else {
        ("single-shot-generation", 1)
    };
    let path = |kv_path: &str, operation: &str, sequences: u64| ReceiptMeasurementPath {
        kv_path: kv_path.into(),
        operation: operation.into(),
        sequences,
    };
    ReceiptMeasurementPaths {
        memory: path(persistent_kv_representation, operation, sequences),
        prefill_first_token: path(persistent_kv_representation, operation, sequences),
        decode_timing: path(COMPRESSED_PERSISTENT_KV, "steady-decode", 1),
        quality: path(COMPRESSED_PERSISTENT_KV, "quality", 1),
    }
}

pub const COMPRESSED_PERSISTENT_KV: &str = "compressed";
pub const DENSE_FALLBACK_PERSISTENT_KV: &str = "dense-fallback";

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
    /// Compressed rows only; absent (not serialized) on dense receipts so their bytes and seals are
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compression: Option<ReceiptCompression>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RawTiming {
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub ttft_ms: f64,
    pub first_token_ms: f64,
    pub decode_tokens_per_second: f64,
    pub steady_decode: SteadyDecodeMeasurement,
    /// The host state captured right after this repeat's steady decode (the worker fills it).
    pub host_state: Option<ReceiptHostState>,
}

/// Fixed decode length of every SC-20671 steady-decode sample. Steady throughput is a dedicated
/// measurement per repeat — the row's context prefilled into a fresh cache of the row's KV
/// representation, then exactly this many greedy tokens decoded through any stop token — never
/// the coordinate's own (EOS- or budget-terminated, footprint-sampled) generation. 256 is the
/// longest fixed length every frozen context band admits: the fit-boundary fixture prompt is
/// refused above `native context - 256`, so `prompt + 256` always fits the native window. The
/// 255 timed tokens span roughly 1-2 s at short context on the pinned 1.7B-3B 4-bit candidates
/// (about 130-250 tok/s) and several seconds at the longer bands.
pub const STEADY_DECODE_TOKENS: u64 = 256;

/// One steady-decode sample, measured by the product decoder at a synchronized per-token boundary
/// (the crate-private `decode::forced_greedy_decode`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SteadyDecodeMeasurement {
    /// Raw tokens of the row context prefilled before decoding.
    pub prompt_tokens: u64,
    /// Every decoded token, including the untimed first one; always [`STEADY_DECODE_TOKENS`].
    pub generated_tokens: u64,
    /// Tokens inside the timed window: every decoded token after the first.
    pub timed_tokens: u64,
    /// Milliseconds from the first token's GPU completion to the last token's.
    pub decode_ms: f64,
    /// Decoded stop tokens that were forced through instead of ending the sample.
    pub forced_stop_tokens: u64,
}

impl SteadyDecodeMeasurement {
    /// The frozen fixed-length shape and its derived throughput.
    pub fn tokens_per_second(&self) -> Result<f64, String> {
        if self.generated_tokens != STEADY_DECODE_TOKENS
            || self.timed_tokens + 1 != self.generated_tokens
            || self.prompt_tokens == 0
            || self.forced_stop_tokens > self.generated_tokens
            || !self.decode_ms.is_finite()
            || self.decode_ms <= 0.0
        {
            return Err(format!(
                "steady decode is not a {STEADY_DECODE_TOKENS}-token fixed-length sample in positive time: {self:?}"
            ));
        }
        let throughput = self.timed_tokens as f64 * 1_000.0 / self.decode_ms;
        if !throughput.is_finite() || throughput <= 0.0 {
            return Err("invalid steady decode throughput".into());
        }
        Ok(throughput)
    }
}

pub struct ProductTimingMeasurements {
    pub samples: Vec<RawTiming>,
    pub compile_attribution: ReceiptCompileAttribution,
}

pub struct ReceiptBuilder {
    pub template: Receipt,
    pub phases: Vec<ReceiptPhase>,
    pub allocations: Vec<ReceiptAllocation>,
    pub timings: Vec<RawTiming>,
    pub compile_attribution: ReceiptCompileAttribution,
    pub quality: QualityObservation,
}

impl ReceiptBuilder {
    pub fn finish(mut self) -> Result<Receipt, String> {
        self.template.schema_version = RECEIPT_SCHEMA_VERSION;
        self.template.harness_version = RECEIPT_HARNESS_VERSION.into();
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
            || !digest(&self.template.provenance.reference_model_sha256)
            || !revision(&self.template.provenance.scene_works_revision)
            || !revision(&self.template.provenance.inference_revision)
            || !valid_locked_mlx_identity(&self.template.provenance)
            || self.template.provenance.reference_model_bytes == 0
            || self.template.provenance.reference_model_id.is_empty()
        {
            return Err("malformed receipt identity".into());
        }
        if self.template.provenance.scene_works_repository != SCENEWORKS_REPOSITORY
            || self.template.provenance.inference_repository != INFERENCE_REPOSITORY
        {
            return Err("receipt repository identity is not paired SceneWorks".into());
        }
        if !THERMAL_STATES.contains(&self.template.provenance.thermal_state.as_str())
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
            || self.template.geometry.context_window_tokens == 0
            || self.template.geometry.context_target_tokens == 0
            || self.template.geometry.context_payload_tokens == 0
            || !self
                .template
                .geometry
                .query_heads
                .is_multiple_of(self.template.geometry.kv_heads)
            || self.template.geometry.capacity < self.template.geometry.kv_length
        {
            return Err("invalid geometry".into());
        }
        if self.template.geometry.context_target_tokens
            != context_band_target(
                self.template.geometry.context_window_tokens,
                &self.template.matrix.context_band,
            )?
            || self.template.geometry.context_payload_tokens
                > self.template.geometry.context_target_tokens
            || self.template.geometry.context_payload_tokens
                < self.template.geometry.context_target_tokens / 2
        {
            return Err("context band token measurement is outside producer bounds".into());
        }
        if (self.template.matrix.request_mode == "single" && self.template.geometry.batch != 1)
            || (self.template.matrix.request_mode == "supported-batch"
                && self.template.geometry.batch <= 1)
        {
            return Err("batch disagrees with matrix".into());
        }
        if self.phases.len() != 8
            || self.allocations.is_empty()
            || self.timings.len() != TIMING_REPEATS
        {
            return Err("receipt evidence is incomplete".into());
        }
        for (phase, expected) in self.phases.iter().zip(REQUIRED_PHASES) {
            if phase.phase != expected
                || phase.pid == 0
                || !valid_phys_footprint_source(&phase.source)
                || phase.mlx.source != "mlx_rs::memory"
            {
                return Err("invalid phase evidence".into());
            }
        }
        if self
            .phases
            .windows(2)
            .any(|w| !utc_timestamp_before(&w[0].timestamp, &w[1].timestamp))
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
            ]
            .into_iter()
            .all(positive)
        }) {
            return Err("timing samples must be finite and positive".into());
        }
        validate_compile_attribution(&self.compile_attribution, &self.template.matrix)?;
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
        self.template.memory.release = release_evidence(&self.phases[1], &self.phases[7]);
        (
            self.template.memory.dense_kv_share_bps,
            self.template.memory.below_memory_material_share,
        ) = dense_kv_share(
            dense,
            self.phases[2].phys_footprint_bytes,
            &self.template.matrix.context_band,
        )?;
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
            cold_compile_ms: cold_compile_alias(&self.compile_attribution),
            warm_compile_ms: self.compile_attribution.steady_dispatch_ms,
            compile_attribution: self.compile_attribution,
            samples: self
                .timings
                .into_iter()
                .map(|t| {
                    Ok(ReceiptTimingSample {
                        load_ms: t.load_ms,
                        prefill_ms: t.prefill_ms,
                        ttft_ms: t.ttft_ms,
                        first_token_ms: t.first_token_ms,
                        decode_tokens_per_second: t.decode_tokens_per_second,
                        steady_decode_prompt_tokens: t.steady_decode.prompt_tokens,
                        steady_decode_generated_tokens: t.steady_decode.generated_tokens,
                        steady_decode_timed_tokens: t.steady_decode.timed_tokens,
                        steady_decode_ms: t.steady_decode.decode_ms,
                        steady_decode_forced_stop_tokens: t.steady_decode.forced_stop_tokens,
                        host_state: t
                            .host_state
                            .ok_or("timing sample has no recorded host state")?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
            summary,
        };
        self.template.quality.parity_max_error = metrics.parity_max_error;
        self.template.quality.perplexity_delta = metrics.perplexity_delta;
        self.template.quality.greedy_token_agreement = metrics.greedy_token_agreement;
        self.template.quality.greedy_agreement_method = GREEDY_AGREEMENT_METHOD.into();
        self.template.quality.free_running_first_divergence =
            self.quality.free_running_first_divergence;
        self.template.quality.greedy_token_agreement_by_repeat =
            self.quality.greedy_agreement_by_repeat.clone();
        self.template.quality.structured_tool_agreement = metrics.structured_tool_agreement;
        self.template.quality.needle_retrieval = metrics.needle_retrieval;
        self.template.quality.needle_discriminating = self.quality.needle_discriminating;
        self.template.quality.tool_discriminating = self.quality.tool_discriminating;
        self.template.quality.multi_turn_prompt_cache = metrics.multi_turn_prompt_cache;
        self.template.quality.multi_turn_prompt_cache_method =
            MULTI_TURN_PROMPT_CACHE_METHOD.into();
        self.template
            .quality
            .multi_turn_free_running_first_divergence =
            self.quality.cache_free_running_first_divergence;
        self.template.quality.multi_turn_matched_prefix_tokens =
            self.quality.cache_matched_prefix_tokens;
        self.template.quality.multi_turn_cache = ReceiptMultiTurnCache {
            candidate: self.quality.cache_candidate_turns.clone(),
            reference: self.quality.cache_reference_turns.clone(),
        };
        // A compressed row's quality measurement is gated; the outcome, pass or fail, is
        // recorded with the row (dense rows are characterization and carry no gate).
        (
            self.template.quality.quality_gate,
            self.template.quality.forced_continuation,
            self.template.quality.multi_turn_forced_continuation,
            self.template.quality.multi_turn_forced_pass,
        ) = if self.template.mode == "compressed" {
            if self.quality.repeat_metrics.len() != QUALITY_MEASUREMENTS {
                return Err(
                    "a compressed quality gate evaluates exactly the row's one quality measurement"
                        .into(),
                );
            }
            (
                Some(quality_gate_from_repeats(
                    &self.quality.repeat_metrics,
                    self.quality.needle_discriminating,
                )),
                self.quality.forced_continuation.clone(),
                self.quality.multi_turn_forced_continuation.clone(),
                self.quality.multi_turn_forced_pass.clone(),
            )
        } else {
            (None, None, None, None)
        };
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

/// Canonical semantic-seal bytes shared with the JavaScript consumer. JSON serializers do not
/// agree on the shortest decimal spelling of every IEEE-754 value, so hashing their textual
/// number rendering is not portable. Normalize every finite JSON number to its exact f64 bit
/// pattern before applying the ordinary recursively sorted JSON representation.
fn canonical_semantic_seal_bytes(value: &serde_json::Value) -> Result<Vec<u8>, String> {
    fn normalize(value: &serde_json::Value) -> Result<serde_json::Value, String> {
        match value {
            serde_json::Value::Array(values) => values
                .iter()
                .map(normalize)
                .collect::<Result<Vec<_>, _>>()
                .map(serde_json::Value::Array),
            serde_json::Value::Object(values) => values
                .iter()
                .map(|(key, value)| Ok((key.clone(), normalize(value)?)))
                .collect::<Result<serde_json::Map<_, _>, String>>()
                .map(serde_json::Value::Object),
            serde_json::Value::Number(number) => {
                let value = number
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or("semantic seal contains a non-finite or unrepresentable number")?;
                Ok(serde_json::Value::String(format!(
                    "f64:{:016x}",
                    value.to_bits()
                )))
            }
            _ => Ok(value.clone()),
        }
    }
    canonical_json_bytes(&normalize(value)?).map_err(|error| error.to_string())
}

/// Seal the same semantic core used by the SC-20671 artifact publisher.  Consumers must never
/// treat a sidecar or a syntactically valid receipt as proof that `receiptSha256` binds it.
pub fn receipt_semantic_seal(receipt: &Receipt) -> Result<String, String> {
    // Normalize through the exact serialized representation before hashing. serde_json can retain
    // a slightly different internal Number for a producer-owned f64 than it constructs when the
    // published bytes are parsed again (for example 14.162040999999993). Hashing the pre-serialize
    // Value would therefore create a receipt which fails its own byte-level consumer check.
    let bytes = receipt.bytes().map_err(|e| e.to_string())?;
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    value
        .as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    Ok(seal_bytes(&canonical_semantic_seal_bytes(&value)?))
}

/// Validate a receipt together with the semantic-core seal embedded by the producer.
pub fn validate_sealed_receipt(receipt: &Receipt) -> Result<(), String> {
    validate_receipt_semantics(receipt)?;
    if receipt.receipt_sha256 != receipt_semantic_seal(receipt)? {
        return Err("receipt semantic-core seal does not match receiptSha256".into());
    }
    Ok(())
}

/// Compressed-row evidence rules (SC-20676): fused execution happened, every dense fallback is
/// reasoned and counted, physical bytes are the sum of their measured components, and no dense
/// full-cache reconstruction survived.
fn validate_receipt_compression(
    receipt: &Receipt,
    compression: &ReceiptCompression,
) -> Result<(), String> {
    if compression.method.is_empty()
        || !compression
            .method
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        || compression.representation_identity.trim().is_empty()
        || compression.representation_version == 0
        || compression.bits == 0
        || compression.quantization_group_size == 0
        || ![
            crate::primitives::PackedMetalGpuFamily::ConservativeUnknownApple,
            crate::primitives::PackedMetalGpuFamily::Apple7OrNewer,
        ]
        .iter()
        .any(|family| family.as_str() == compression.kernel_gpu_family)
    {
        return Err("compressed representation identity is incomplete".into());
    }
    // Each method names exactly one representation: its code width and reader identity.
    let method = CompressedKvMethod::parse(&compression.method)?;
    if compression.bits != u64::from(method.code_bits().bits())
        || compression.representation_identity != method.representation_identity()
    {
        return Err(format!(
            "compressed method {} is {}-bit {} but the receipt records {}-bit {}",
            method.id(),
            method.code_bits().bits(),
            method.representation_identity(),
            compression.bits,
            compression.representation_identity
        ));
    }
    let device_bytes = compression
        .device_code_bytes
        .checked_add(compression.device_metadata_bytes);
    if device_bytes.and_then(|bytes| bytes.checked_add(compression.host_payload_bytes))
        != Some(compression.physical_kv_bytes)
    {
        return Err("compressed physical KV bytes do not reconcile with measured storage".into());
    }
    if compression.fused_calls == 0 {
        return Err("compressed row never executed the fused compressed-domain reader".into());
    }
    let mut path_calls = 0_u64;
    for (index, path) in compression.kernel_paths.iter().enumerate() {
        // The NAX kernel appears exactly with the NAX selection (and 16-bit queries), and every
        // other kernel only with a selection its GPU family and query dtype allow.
        if !crate::primitives::packed_kernel_path_valid(
            &compression.kernel_gpu_family,
            &path.kernel,
            &path.selection,
            &path.query_dtype,
        ) || path.reason.trim().is_empty()
            || path.calls == 0
            || compression.kernel_paths[..index].iter().any(|prior| {
                (&prior.kernel, &prior.selection, &prior.query_dtype)
                    >= (&path.kernel, &path.selection, &path.query_dtype)
            })
        {
            return Err(
                "compressed kernel path disagrees with its selection or is unordered".into(),
            );
        }
        path_calls = path_calls
            .checked_add(path.calls)
            .ok_or("compressed kernel path count overflows u64")?;
    }
    if path_calls != compression.fused_calls {
        return Err("compressed fused calls are not all attributed to a kernel path".into());
    }
    let mut calls = 0_u64;
    for (index, fallback) in compression.fallbacks.iter().enumerate() {
        if fallback.operation.trim().is_empty()
            || fallback.reason.trim().is_empty()
            || fallback.calls == 0
            || compression.fallbacks[..index].iter().any(|prior| {
                (prior.operation.as_str(), prior.reason.as_str())
                    >= (fallback.operation.as_str(), fallback.reason.as_str())
            })
        {
            return Err("compressed dense fallback is unreasoned, empty, or unordered".into());
        }
        calls = calls
            .checked_add(fallback.calls)
            .ok_or("compressed fallback count overflows u64")?;
    }
    if calls != compression.fallback_calls
        || (compression.failed_dispatches != 0 && compression.fallback_calls == 0)
        || (compression.fallback_calls != 0 && !receipt.lifecycle.dense_fallback)
    {
        return Err("compressed fallback calls are not fully reasoned".into());
    }
    match compression.persistent_kv_representation.as_str() {
        // The coordinate's own compressed storage: the MLX-resident persistent KV is exactly its
        // device arrays, it describes the receipt's KV length, and the whole physical
        // representation (device + host copy + staged tail) is below the dense geometry.
        COMPRESSED_PERSISTENT_KV => {
            coordinate_storage_reconciles(
                compression.device_code_bytes,
                device_bytes,
                compression.storage_tokens,
                receipt.memory.persistent_kv_bytes,
                receipt.geometry.kv_length,
            )?;
            if compression.physical_kv_bytes >= receipt.memory.dense_theoretical_kv_bytes {
                return Err(
                    "compressed persistent KV representation disagrees with its evidence".into(),
                );
            }
        }
        // A dense coordinate claims no compressed storage, so it can never yield a reduction.
        DENSE_FALLBACK_PERSISTENT_KV
            if compression.fallback_calls != 0
                && compression.physical_kv_bytes == 0
                && compression.storage_tokens == 0 => {}
        _ => {
            return Err(
                "compressed persistent KV representation disagrees with its evidence".into(),
            )
        }
    }
    if compression.full_cache_dequantizations != 0 {
        return Err("compressed row reconstructed a dense full cache".into());
    }
    if compression.measurement_paths.as_ref().is_some_and(|paths| {
        *paths
            != receipt_measurement_paths(
                &receipt.matrix.request_mode,
                &receipt.matrix.prefill_mode,
                &compression.persistent_kv_representation,
            )
    }) {
        return Err(
            "compression.measurementPaths does not name the KV path each measurement ran on".into(),
        );
    }
    Ok(())
}

/// A compressed coordinate's measured storage must be its persistent KV exactly: the MLX-resident
/// device share equals `memory.persistentKvBytes` and the storage describes `geometry.kvLength`.
pub(crate) fn coordinate_storage_reconciles(
    device_code_bytes: u64,
    device_bytes: Option<u64>,
    storage_tokens: u64,
    persistent_kv_bytes: u64,
    kv_length: u64,
) -> Result<(), String> {
    if device_code_bytes == 0
        || device_bytes != Some(persistent_kv_bytes)
        || storage_tokens != kv_length
    {
        return Err(format!(
            "compressed persistent KV does not reconcile with the coordinate's measured storage: device bytes {device_bytes:?} (codes {device_code_bytes}) vs persistentKvBytes {persistent_kv_bytes}; storageTokens {storage_tokens} vs kvLength {kv_length}"
        ));
    }
    Ok(())
}

/// The row's recorded power mode and thermal state: one observation per boundary, in order, each
/// nominal and in the row's single power mode, bracketing every measured phase sample.
fn validate_host_states(receipt: &Receipt) -> Result<(), String> {
    let p = &receipt.provenance;
    let invalid = |detail: &str| Err(format!("host power/thermal provenance {detail}"));
    let [start, end] = p.host_states.as_slice() else {
        return invalid("does not record exactly the row-start and row-end states");
    };
    start.validate(HOST_STATE_BOUNDARIES[0])?;
    end.validate(HOST_STATE_BOUNDARIES[1])?;
    refuse_throttled_row_start(start)
        .map_err(|error| format!("host power/thermal provenance: {error}"))?;
    if p.power_mode != start.power_mode || p.thermal_state != start.thermal_state {
        return invalid("is not the row-start state");
    }
    let samples = receipt
        .timings
        .samples
        .iter()
        .map(|sample| &sample.host_state)
        .collect::<Vec<_>>();
    for state in &samples {
        state.validate(TIMING_SAMPLE_HOST_BOUNDARY)?;
    }
    let order = std::iter::once(start)
        .chain(samples.iter().copied())
        .chain(std::iter::once(end))
        .collect::<Vec<_>>();
    if order
        .windows(2)
        .any(|pair| !utc_timestamp_before(&pair[0].captured_at, &pair[1].captured_at))
    {
        return invalid("is not ordered row start, timing samples, row end");
    }
    if (
        p.thermal_changed_during_row,
        p.power_mode_changed_during_row,
    ) != host_state_changes(start, samples.iter().copied().chain(std::iter::once(end)))
    {
        return invalid("change flags do not recompute");
    }
    let (Some(first), Some(last)) = (
        receipt.memory.phase_samples.first(),
        receipt.memory.phase_samples.last(),
    ) else {
        return invalid("has no phase samples to bracket");
    };
    if !utc_timestamp_before(&start.captured_at, &first.timestamp)
        || !utc_timestamp_before(&last.timestamp, &end.captured_at)
    {
        return invalid("does not bracket the row's measured phases");
    }
    Ok(())
}

/// A compressed receipt's measured quality gate must be exactly the frozen-threshold evaluation
/// of the values the receipt records: every miss named with its value, threshold, repeat, and
/// fixture, and `passed` only when nothing missed. Dense receipts are characterization and carry
/// neither a gate nor a forced continuation. Values of repeats 1-4 other than greedy agreement
/// live in the sealed fixture artifacts and are bound by the artifact-bundle validator.
fn validate_quality_gate(receipt: &Receipt) -> Result<(), String> {
    let quality = &receipt.quality;
    if receipt.mode != "compressed" {
        if quality.quality_gate.is_some()
            || quality.forced_continuation.is_some()
            || quality.multi_turn_forced_continuation.is_some()
            || quality.multi_turn_forced_pass.is_some()
        {
            return Err(
                "a quality gate and forced continuation are recorded only on compressed receipts"
                    .into(),
            );
        }
        return Ok(());
    }
    let multi_turn = quality
        .multi_turn_forced_continuation
        .as_ref()
        .ok_or("compressed receipt has no turn-2 forced continuation for multiTurnPromptCache")?;
    validate_forced_continuation_with(multi_turn, false, MULTI_TURN_FORCED_CONTINUATION_METHOD)?;
    let pass = quality
        .multi_turn_forced_pass
        .as_ref()
        .ok_or("compressed receipt has no turn records for its turn-2 forced continuation")?;
    pass.validate()?;
    if pass.reference.turn2.prompt_sha256 != quality.multi_turn_cache.reference.turn2.prompt_sha256
    {
        return Err(
            "the turn-2 forced continuation's prompt is not the multi-turn fixture's turn-2 prompt"
                .into(),
        );
    }
    if quality.multi_turn_prompt_cache.to_bits() != multi_turn.agreement.to_bits() {
        return Err(format!(
            "multiTurnPromptCache {} is not the row's turn-2 forced-continuation agreement {}",
            quality.multi_turn_prompt_cache, multi_turn.agreement
        ));
    }
    let continuation = quality
        .forced_continuation
        .as_ref()
        .ok_or("compressed receipt has no forced-continuation greedy agreement")?;
    validate_forced_continuation(continuation, &receipt.matrix.context_band)?;
    if quality
        .greedy_token_agreement_by_repeat
        .iter()
        .chain(std::iter::once(&quality.greedy_token_agreement))
        .any(|value| value.to_bits() != continuation.agreement.to_bits())
    {
        return Err(format!(
            "greedyTokenAgreement {} (by repeat {:?}) is not the row's forced-continuation agreement {}",
            quality.greedy_token_agreement,
            quality.greedy_token_agreement_by_repeat,
            continuation.agreement
        ));
    }
    let gate = quality
        .quality_gate
        .as_ref()
        .ok_or("compressed receipt has no measured quality-gate record")?;
    let mut previous: Option<(u64, usize)> = None;
    for failure in &gate.failures {
        let index = QUALITY_GATE_METRICS
            .iter()
            .position(|spec| spec.metric == failure.metric)
            .ok_or_else(|| format!("quality gate names unknown metric {}", failure.metric))?;
        let spec = &QUALITY_GATE_METRICS[index];
        if failure.fixture != spec.fixture
            || failure.threshold.to_bits() != spec.threshold.to_bits()
            || failure.comparison != spec.comparison.as_str()
            || failure.repeat >= QUALITY_MEASUREMENTS as u64
            || (failure.metric == "needleRetrieval" && !quality.needle_discriminating)
            || !failure.value.is_finite()
            || !spec.comparison.misses(failure.value, spec.threshold)
            || previous.is_some_and(|previous| previous >= (failure.repeat, index))
        {
            return Err(format!(
                "quality gate failure is not an ordered frozen-threshold miss: metric={} value={} threshold={} comparison={} repeat={} fixture={}",
                failure.metric,
                failure.value,
                failure.threshold,
                failure.comparison,
                failure.repeat,
                failure.fixture
            ));
        }
        previous = Some((failure.repeat, index));
    }
    if gate.passed != gate.failures.is_empty() {
        return Err(format!(
            "quality gate claims passed={} with {} recorded failure(s): {}",
            gate.passed,
            gate.failures.len(),
            quality_gate_summary(&ReceiptQualityGate {
                passed: false,
                failures: gate.failures.clone(),
            })
        ));
    }
    // Every value the receipt itself records must appear in the gate exactly when it misses.
    let recorded = |metric: &str, repeat: u64| {
        gate.failures
            .iter()
            .find(|failure| failure.metric == metric && failure.repeat == repeat)
            .map(|failure| failure.value.to_bits())
    };
    let visible = quality
        .greedy_token_agreement_by_repeat
        .iter()
        .enumerate()
        .map(|(repeat, value)| ("greedyTokenAgreement", repeat as u64, *value))
        .chain([
            ("perplexityDelta", 0, quality.perplexity_delta),
            (
                "structuredToolAgreement",
                0,
                quality.structured_tool_agreement,
            ),
            ("needleRetrieval", 0, quality.needle_retrieval),
            ("multiTurnPromptCache", 0, quality.multi_turn_prompt_cache),
        ]);
    for (metric, repeat, value) in visible {
        let spec = QUALITY_GATE_METRICS
            .iter()
            .find(|spec| spec.metric == metric)
            .ok_or("quality gate metric table is incomplete")?;
        // Contract v5: a non-discriminating needle is an observation and never a gate failure.
        let gated = quality.needle_discriminating || metric != "needleRetrieval";
        let expected =
            (gated && spec.comparison.misses(value, spec.threshold)).then_some(value.to_bits());
        if recorded(metric, repeat) != expected {
            return Err(format!(
                "quality gate does not record {metric} repeat {repeat} = {value} against the frozen {} {} (fixture {}): passed={}",
                spec.comparison.as_str(),
                spec.threshold,
                spec.fixture,
                gate.passed
            ));
        }
    }
    Ok(())
}

pub fn validate_receipt_semantics(receipt: &Receipt) -> Result<(), String> {
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
    if receipt.schema_version != RECEIPT_SCHEMA_VERSION
        || receipt.harness_version != RECEIPT_HARNESS_VERSION
        || receipt.status != "complete"
        || receipt.contract_hash != QUALITY_CONTRACT_HASH
    {
        return Err("receipt constants mismatch".into());
    }
    if !["dense", "compressed"].contains(&receipt.mode.as_str())
        || receipt.run_id.is_empty()
        || !valid_utc_timestamp(&receipt.captured_at)
    {
        return Err("receipt timestamp/run id is malformed".into());
    }
    let p = &receipt.provenance;
    if p.scene_works_repository != SCENEWORKS_REPOSITORY
        || p.inference_repository != INFERENCE_REPOSITORY
    {
        return Err("provenance repository identity is not paired SceneWorks".into());
    }
    if !revision(&p.scene_works_revision)
        || !revision(&p.inference_revision)
        || !lowercase_hex(&p.dependency_lock_sha256, 64)
        || !lowercase_hex(&p.model_file_sha256, 64)
        || !lowercase_hex(&p.reference_model_sha256, 64)
        || !valid_locked_mlx_identity(p)
        || [
            p.mlx_version.as_str(),
            p.mlx_source.as_str(),
            p.mlx_revision.as_str(),
            p.os.as_str(),
            p.xcode.as_str(),
            p.hardware.as_str(),
            p.model_id.as_str(),
            p.reference_model_id.as_str(),
            p.power_mode.as_str(),
            p.thermal_state.as_str(),
            p.command_template.as_str(),
            p.command.as_str(),
            p.campaign_session_id.as_str(),
        ]
        .iter()
        .any(|v| v.is_empty())
        || p.model_file_bytes == 0
        || p.reference_model_bytes == 0
        || !lowercase_hex(&p.campaign_session_id, 64)
        || !lowercase_hex(&p.coordinate_operation_sha256, 64)
        || p.campaign_cache_state_version == 0
        || !THERMAL_STATES.contains(&p.thermal_state.as_str())
        || !p.command_template.contains("{mode}")
        || p.command_template.replace("{mode}", &receipt.mode) != p.command
    {
        return Err("provenance is incomplete".into());
    }
    validate_host_states(receipt)?;
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
        g.context_window_tokens,
        g.context_target_tokens,
        g.context_payload_tokens,
    ]
    .into_iter()
    .any(|v| v == 0)
    {
        return Err("geometry contains zero".into());
    }
    if g.element_bytes != DENSE_KV_COMPUTE_ELEMENT_BYTES {
        return Err(dense_kv_width_refusal(g.element_bytes));
    }
    if !["llama", "qwen"].contains(&receipt.matrix.family.as_str())
        || !CONTEXT_BANDS.contains(&receipt.matrix.context_band.as_str())
        || !["single", "supported-batch"].contains(&receipt.matrix.request_mode.as_str())
        || !["chunked", "single-shot"].contains(&receipt.matrix.prefill_mode.as_str())
        || !["cold", "warm"].contains(&receipt.matrix.process_temperature.as_str())
        || !g.query_heads.is_multiple_of(g.kv_heads)
        || g.capacity < g.kv_length
        || (receipt.matrix.request_mode == "single" && g.batch != 1)
        || (receipt.matrix.request_mode == "supported-batch" && g.batch <= 1)
    {
        return Err("matrix coordinate or geometry relationship is invalid".into());
    }
    if receipt.geometry.context_target_tokens
        != context_band_target(
            receipt.geometry.context_window_tokens,
            &receipt.matrix.context_band,
        )?
        || receipt.geometry.context_payload_tokens > receipt.geometry.context_target_tokens
        || receipt.geometry.context_payload_tokens < receipt.geometry.context_target_tokens / 2
    {
        return Err("context band token measurement is outside producer bounds".into());
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
                    || !valid_phys_footprint_source(&p.source)
                    || p.mlx.source != "mlx_rs::memory"
                    || p.phys_footprint_peak_bytes < p.phys_footprint_bytes
                    || !valid_utc_timestamp(&p.timestamp)
            })
    {
        return Err("phase PID evidence is inconsistent".into());
    }
    if receipt.memory.phase_samples.windows(2).any(|w| {
        !utc_timestamp_before(&w[0].timestamp, &w[1].timestamp)
            || w[1].phys_footprint_peak_bytes < w[0].phys_footprint_peak_bytes
    }) || receipt
        .memory
        .phase_samples
        .windows(2)
        .enumerate()
        .any(|(index, w)| {
            // MLX peak memory is deliberately reset to zero at the sealed boundary immediately
            // before prefill.  Every other adjacent phase remains process-local and monotonic.
            index != 1 && w[1].mlx.peak_bytes < w[0].mlx.peak_bytes
        })
    {
        return Err("phase sequence or peak monotonicity failed".into());
    }
    if receipt.memory.phase_samples.iter().any(|p| {
        p.phys_footprint_bytes < p.mlx.active_bytes || p.mlx.peak_bytes < p.mlx.active_bytes
    }) {
        return Err("memory containment failed".into());
    }
    for event in &receipt.memory.allocation_events {
        let phase_index = REQUIRED_PHASES
            .iter()
            .position(|phase| *phase == event.phase)
            .ok_or_else(|| "allocation event is malformed".to_string())?;
        let phase_start = &receipt.memory.phase_samples[phase_index];
        let next_phase = receipt.memory.phase_samples.get(phase_index + 1);
        if event.bytes == 0
            || !["cache", "attention-workspace", "weights", "output"].contains(&event.role.as_str())
            || !["persistent", "transient", "released"].contains(&event.lifetime.as_str())
            || event.phase.is_empty()
            || event.kind.is_empty()
            || !REQUIRED_PHASES.contains(&event.phase.as_str())
            || !valid_utc_timestamp(&event.timestamp)
            || compare_utc_timestamps(&event.timestamp, &phase_start.timestamp)
                != Some(std::cmp::Ordering::Greater)
            || next_phase
                .is_some_and(|sample| !utc_timestamp_before(&event.timestamp, &sample.timestamp))
        {
            return Err("allocation event is malformed".into());
        }
        if event.lifetime == "released"
            && (event.role != "cache" || event.kind != "product-cache_release")
        {
            return Err("release lifecycle event is malformed".into());
        }
    }
    let max_role = |role: &str, lifetime: &str| {
        receipt
            .memory
            .allocation_events
            .iter()
            .filter(|e| e.role == role && e.lifetime == lifetime)
            .map(|event| event.bytes)
            .max()
            .unwrap_or(0)
    };
    let max_transient = receipt
        .memory
        .allocation_events
        .iter()
        .filter(|e| {
            e.lifetime == "transient"
                && (e.role == "cache" || e.role == "attention-workspace" || e.role == "output")
        })
        .map(|event| event.bytes)
        .max()
        .unwrap_or(0);
    let max_phase_role = |phase: &str, role: &str, lifetime: &str| {
        receipt
            .memory
            .allocation_events
            .iter()
            .filter(|event| {
                event.phase == phase && event.role == role && event.lifetime == lifetime
            })
            .map(|event| event.bytes)
            .max()
            .unwrap_or(0)
    };
    let max_phase_transient = |phase: &str| {
        receipt
            .memory
            .allocation_events
            .iter()
            .filter(|event| {
                event.phase == phase
                    && event.lifetime == "transient"
                    && (event.role == "cache"
                        || event.role == "attention-workspace"
                        || event.role == "output")
            })
            .map(|event| event.bytes)
            .max()
            .unwrap_or(0)
    };
    if max_role("weights", "persistent") != receipt.memory.model_weights_bytes
        || max_role("cache", "persistent") != receipt.memory.persistent_kv_bytes
        || max_transient != receipt.memory.transient_workspace_bytes
    {
        return Err("allocation totals do not reconcile".into());
    }
    let mut live_cache = None;
    let mut releases = 0_u64;
    for event in &receipt.memory.allocation_events {
        if event.role == "cache" && event.lifetime == "persistent" {
            live_cache = Some(event.bytes);
        } else if event.lifetime == "released" {
            if live_cache != Some(event.bytes) {
                return Err("cache release bytes do not match retained KV ownership".into());
            }
            live_cache = None;
            releases = releases
                .checked_add(1)
                .ok_or("cache release event count overflow")?;
        }
    }
    if releases == 0 || live_cache.is_some() {
        return Err("persistent KV ownership was not explicitly released".into());
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
    let prefill_footprint = receipt
        .memory
        .phase_samples
        .iter()
        .find(|sample| sample.phase == "prefill-peak")
        .ok_or("missing prefill-peak materiality evidence")?
        .phys_footprint_bytes;
    if (
        receipt.memory.dense_kv_share_bps,
        receipt.memory.below_memory_material_share,
    ) != dense_kv_share(dense, prefill_footprint, &receipt.matrix.context_band)?
    {
        return Err("dense KV share of the prefill footprint does not recompute".into());
    }
    if receipt.matrix.context_band == "fit-boundary"
        && (receipt.geometry.kv_length > receipt.geometry.context_window_tokens
            || u128::from(receipt.geometry.kv_length).saturating_mul(10_000)
                < u128::from(receipt.geometry.context_window_tokens)
                    .saturating_mul(u128::from(FIT_BOUNDARY_MIN_CONTEXT_BPS)))
    {
        return Err("fit-boundary cache occupancy is below the frozen admission ratio".into());
    }
    let transient_high_water = receipt
        .memory
        .allocation_events
        .iter()
        .filter(|e| {
            e.lifetime == "transient"
                && (e.role == "cache" || e.role == "attention-workspace" || e.role == "output")
        })
        .map(|event| event.bytes)
        .max()
        .unwrap_or(0);
    if receipt.mode == "compressed"
        && u128::from(transient_high_water) * 10 >= u128::from(dense) * 9
    {
        return Err("full-cache transient high-water detected".into());
    }
    // An explicitly witnessed dense full-cache temporary is rejected regardless of its size
    // (mirrors SceneWorks' detectFullCacheTemporary).
    if receipt.mode == "compressed"
        && receipt.memory.allocation_events.iter().any(|event| {
            event.lifetime == "transient"
                && [FULL_CACHE_MATERIALIZATION_KIND, DENSE_CACHE_TEMPORARY_KIND]
                    .contains(&event.kind.as_str())
        })
    {
        return Err("explicit full-cache temporary detected".into());
    }
    receipt.memory.admission.validate_admitted()?;
    let weights = receipt.memory.model_weights_bytes;
    let kv = receipt.memory.persistent_kv_bytes;
    let sample_for = |phase: &str| {
        receipt
            .memory
            .phase_samples
            .iter()
            .find(|sample| sample.phase == phase)
            .ok_or_else(|| format!("missing containment phase {phase}"))
    };
    let process_start = sample_for("process-start")?;
    let active_delta = |sample: &ReceiptPhase| {
        sample
            .mlx
            .active_bytes
            .checked_sub(process_start.mlx.active_bytes)
            .ok_or_else(|| {
                format!(
                    "{} MLX active bytes fell below process baseline",
                    sample.phase
                )
            })
    };
    let weights_loaded = sample_for("weights-loaded")?;
    if active_delta(weights_loaded)? < weights {
        return Err("weights-loaded MLX active bytes do not contain weights".into());
    }
    let prefill_peak_window = &receipt.memory.prefill_peak_window;
    if !valid_utc_timestamp(&prefill_peak_window.started_at)
        || compare_utc_timestamps(&prefill_peak_window.started_at, &weights_loaded.timestamp)
            != Some(std::cmp::Ordering::Greater)
        || prefill_peak_window.reset_peak_bytes != 0
        || prefill_peak_window.baseline_active_bytes < weights_loaded.mlx.active_bytes
    {
        return Err("MLX prefill peak window is invalid".into());
    }
    let prefill = sample_for("prefill-peak")?;
    if !utc_timestamp_before(&prefill_peak_window.started_at, &prefill.timestamp) {
        return Err("MLX prefill peak window is not ordered before prefill".into());
    }
    let prefill_kv = max_phase_role("prefill-peak", "cache", "persistent");
    let decode_kv = max_phase_role("decode-steady", "cache", "persistent");
    if prefill_kv == 0 || prefill_kv > kv || decode_kv != kv {
        return Err("phase-local persistent KV snapshots do not reconcile".into());
    }
    let prefill_workspace = max_phase_transient("prefill-peak");
    let decode_workspace = max_phase_transient("decode-steady");
    let prefill_active_floor = prefill_peak_window
        .baseline_active_bytes
        .checked_add(prefill_kv)
        .ok_or("prefill persistent memory floor overflows u64")?;
    let prefill_peak_floor = prefill_active_floor
        .checked_add(prefill_workspace)
        .ok_or("prefill attributed peak floor overflows u64")?;
    // The instantaneous sample must retain the phase-local baseline and KV. Transient workspace may
    // be released by the time the post-dispatch sample is captured, so the sealed reset window
    // binds the prefill peak without accepting a session-global load or warmup high-water.
    if prefill.mlx.active_bytes < prefill_active_floor
        || prefill.mlx.peak_bytes < prefill_peak_floor
    {
        return Err(format!(
            "prefill MLX samples do not contain phase-local attributed allocations: baselineActiveBytes={}, prefillPersistentKvBytes={prefill_kv}, prefillTransientBytes={prefill_workspace}, activeBytes={}, activeFloor={prefill_active_floor}, peakBytes={}, peakFloor={prefill_peak_floor}",
            prefill_peak_window.baseline_active_bytes,
            prefill.mlx.active_bytes,
            prefill.mlx.peak_bytes,
        ));
    }
    let decode = sample_for("decode-steady")?;
    let decode_active_floor = prefill_peak_window
        .baseline_active_bytes
        .checked_add(decode_kv)
        .ok_or("decode persistent memory floor overflows u64")?;
    let decode_peak_floor = decode_active_floor
        .checked_add(decode_workspace)
        .ok_or("decode attributed peak floor overflows u64")?;
    if decode.mlx.active_bytes < decode_active_floor || decode.mlx.peak_bytes < decode_peak_floor {
        return Err("decode MLX samples do not contain phase-local attributed allocations".into());
    }
    for phase in REQUIRED_PHASES
        .into_iter()
        .filter(|phase| !["prefill-peak", "decode-steady"].contains(phase))
    {
        let transient = max_phase_transient(phase);
        if transient == 0 {
            continue;
        }
        let persistent = max_phase_role(phase, "cache", "persistent");
        if persistent == 0 {
            return Err(format!(
                "{phase} transient evidence has no phase-local persistent KV snapshot"
            ));
        }
        let peak_floor = prefill_peak_window
            .baseline_active_bytes
            .checked_add(persistent)
            .and_then(|bytes| bytes.checked_add(transient))
            .ok_or("phase-local attributed peak floor overflows u64")?;
        if sample_for(phase)?.mlx.peak_bytes < peak_floor {
            return Err(format!(
                "{phase} MLX peak bytes do not contain phase-local attributed allocations"
            ));
        }
    }
    if receipt.mode == "dense"
        && (receipt.memory.reconciliation.tolerance_bytes != 0
            || receipt.memory.persistent_kv_bytes != dense)
    {
        return Err(format!(
            "dense KV physical bytes do not reconcile: observed={}, allocated={}, tolerance={}",
            receipt.memory.persistent_kv_bytes,
            dense,
            receipt.memory.reconciliation.tolerance_bytes,
        ));
    }
    let end = receipt.memory.phase_samples.last().unwrap();
    if receipt.memory.release.phys_footprint_tolerance_bytes
        != POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES
        || receipt.memory.release.mlx_active_tolerance_bytes
            != post_release_mlx_slack_bytes(weights_loaded.mlx.active_bytes)
        || receipt.memory.release.mlx_cache_tolerance_bytes
            != post_release_mlx_slack_bytes(weights_loaded.mlx.cache_bytes)
        || receipt.memory.release.mlx_active_residual_bytes
            != end
                .mlx
                .active_bytes
                .saturating_sub(weights_loaded.mlx.active_bytes)
        || receipt.memory.release.mlx_cache_residual_bytes
            != end
                .mlx
                .cache_bytes
                .saturating_sub(weights_loaded.mlx.cache_bytes)
    {
        return Err("release tolerances or residuals differ from the platform contract".into());
    }
    if end.phys_footprint_bytes
        > weights_loaded
            .phys_footprint_bytes
            .saturating_add(receipt.memory.release.phys_footprint_tolerance_bytes)
        || end.mlx.active_bytes
            > weights_loaded
                .mlx
                .active_bytes
                .saturating_add(receipt.memory.release.mlx_active_tolerance_bytes)
        || end.mlx.cache_bytes
            > weights_loaded
                .mlx
                .cache_bytes
                .saturating_add(receipt.memory.release.mlx_cache_tolerance_bytes)
    {
        return Err(format!(
            "release did not return within tolerance: endPhys={}, weightsLoadedPhys={}, physTolerance={}, endMlxActive={}, weightsLoadedMlxActive={}, activeTolerance={}, endMlxCache={}, weightsLoadedMlxCache={}, cacheTolerance={}",
            end.phys_footprint_bytes,
            weights_loaded.phys_footprint_bytes,
            receipt.memory.release.phys_footprint_tolerance_bytes,
            end.mlx.active_bytes,
            weights_loaded.mlx.active_bytes,
            receipt.memory.release.mlx_active_tolerance_bytes,
            end.mlx.cache_bytes,
            weights_loaded.mlx.cache_bytes,
            receipt.memory.release.mlx_cache_tolerance_bytes,
        ));
    }
    for sample in &receipt.timings.samples {
        sample.validate_steady_decode(receipt.geometry.context_window_tokens)?;
    }
    if receipt.timings.samples.len() != TIMING_REPEATS
        || !receipt.timings.samples.iter().all(|s| {
            [
                s.load_ms,
                s.prefill_ms,
                s.ttft_ms,
                s.first_token_ms,
                s.decode_tokens_per_second,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        })
        || ![
            receipt.timings.load_ms,
            receipt.timings.prefill_ms,
            receipt.timings.ttft_ms,
            receipt.timings.first_token_ms,
            receipt.timings.decode_tokens_per_second,
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
        return Err(format!(
            "timing policy failed: decodeSamples={:?}, coefficientOfVariation={}, maximum=0.05, summaryMean={}, summaryP95={}, summaryVariance={}, confidenceLow={}, confidenceHigh={}",
            receipt
                .timings
                .samples
                .iter()
                .map(|sample| sample.decode_tokens_per_second)
                .collect::<Vec<_>>(),
            receipt
                .timings
                .summary
                .decode_tokens_per_second_coefficient_of_variation,
            receipt.timings.summary.decode_tokens_per_second_mean,
            receipt.timings.summary.decode_tokens_per_second_p95,
            receipt.timings.summary.decode_tokens_per_second_variance,
            receipt.timings.summary.confidence_interval_low,
            receipt.timings.summary.confidence_interval_high,
        ));
    }
    validate_compile_attribution(&receipt.timings.compile_attribution, &receipt.matrix)?;
    if receipt.timings.cold_compile_ms != cold_compile_alias(&receipt.timings.compile_attribution)
        || (receipt.timings.warm_compile_ms
            - receipt.timings.compile_attribution.steady_dispatch_ms)
            .abs()
            > 1e-9
    {
        return Err("compile timing aliases do not match raw attribution".into());
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
    if receipt.quality.greedy_agreement_method != GREEDY_AGREEMENT_METHOD {
        return Err("greedy agreement is not teacher-forced".into());
    }
    if receipt.quality.multi_turn_prompt_cache_method != MULTI_TURN_PROMPT_CACHE_METHOD {
        return Err(
            "multi-turn prompt-cache agreement is not teacher-forced on a cache-hit turn 2".into(),
        );
    }
    receipt
        .quality
        .multi_turn_cache
        .candidate
        .validate("candidate")?;
    receipt
        .quality
        .multi_turn_cache
        .reference
        .validate("reference")?;
    if receipt
        .quality
        .multi_turn_free_running_first_divergence
        .is_some_and(|divergence| receipt.quality.multi_turn_matched_prefix_tokens > divergence)
    {
        return Err(
            "multi-turn matched prefix extends past its first free-running divergence".into(),
        );
    }
    let by_repeat = &receipt.quality.greedy_token_agreement_by_repeat;
    if by_repeat.len() != QUALITY_MEASUREMENTS
        || by_repeat.iter().any(|value| !(0.0..=1.0).contains(value))
        || by_repeat
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .to_bits()
            != receipt.quality.greedy_token_agreement.to_bits()
    {
        return Err("greedy agreement is not the row's one recorded quality measurement".into());
    }
    if !receipt.quality.parity_max_error.is_finite()
        || !receipt.quality.perplexity_delta.is_finite()
        || receipt.quality.parity_max_error < 0.0
        || [
            receipt.quality.greedy_token_agreement,
            receipt.quality.structured_tool_agreement,
            receipt.quality.needle_retrieval,
            receipt.quality.multi_turn_prompt_cache,
        ]
        .into_iter()
        .any(|v| !(0.0..=1.0).contains(&v))
    {
        return Err("quality evidence contains invalid numeric values".into());
    }
    // Kernel parity compares the fused reader with its independent host-fp32
    // dequantize-then-attend reference over the exact stored codes: a correctness check of the
    // kernel, not a quality-versus-dense metric, so a miss still refuses the row.
    if receipt.mode == "compressed"
        && receipt.quality.parity_max_error > COMPRESSED_PARITY_MAX_ERROR
    {
        return Err(format!(
            "kernel parity failed: metric=parityMaxError value={} threshold={COMPRESSED_PARITY_MAX_ERROR} comparison=maximum fixture=kernel-fp32-reference repeat=all (one fused-reader probe against the host-fp32 dequantize-then-attend reference)",
            receipt.quality.parity_max_error,
        ));
    }
    if receipt.quality.fixture_evidence.len() != 4
        || receipt.quality.statistics != contract_statistics()
    {
        return Err(format!(
            "quality contract evidence incomplete: fixtures={}, statistics={:?}",
            receipt.quality.fixture_evidence.len(),
            receipt.quality.statistics
        ));
    }
    let lowercase_digest = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    for name in REQUIRED_FIXTURES {
        let fixture = receipt
            .quality
            .fixture_evidence
            .get(name)
            .ok_or_else(|| format!("fixture evidence failed: fixture={name} is missing"))?;
        let problem = if !fixture.passed {
            Some("passed=false".to_string())
        } else if fixture.artifact_name != format!("fixtures/{name}.json") {
            Some(format!("artifactName={}", fixture.artifact_name))
        } else if !lowercase_digest(&fixture.artifact_sha256) {
            Some(format!("artifactSha256={}", fixture.artifact_sha256))
        } else if fixture.independent_reference.is_empty() {
            Some("independentReference is empty".into())
        } else if !lowercase_digest(&fixture.artifact_sidecar_sha256) {
            Some(format!(
                "artifactSidecarSha256={}",
                fixture.artifact_sidecar_sha256
            ))
        } else {
            None
        };
        if let Some(problem) = problem {
            return Err(format!("fixture evidence failed: fixture={name} {problem}"));
        }
    }
    // Contract v3: a compressed receipt's quality denominator is the dense-KV run on the same
    // weights, never the bf16 model; dense fixtures are bf16 characterization and must not claim
    // to be a same-weights gate reference.
    for name in REQUIRED_FIXTURES {
        let same_weights = fixture_independent_reference(
            name,
            QualityReference::DenseKvSameWeights,
            &receipt.provenance.model_file_sha256,
        );
        if (receipt.mode == "compressed")
            != (receipt.quality.fixture_evidence[name].independent_reference == same_weights)
        {
            return Err(format!(
                "{name} quality reference is not the contract v3 denominator for a {} receipt",
                receipt.mode
            ));
        }
    }
    // Discrimination is the AND over all repeats; a dense row that missed the needle in its
    // primary repeat cannot claim every repeat recovered it.
    if receipt.mode == "dense"
        && receipt.quality.needle_discriminating
        && receipt.quality.needle_retrieval != 1.0
    {
        return Err(
            "dense needle discrimination claims recovery the dense run did not show".into(),
        );
    }
    if !receipt.memory.release.verified || !receipt.cancellation.cleanup_verified {
        return Err("release/cancellation evidence failed".into());
    }
    let warm = &receipt.warmup;
    if warm.worker_pid == 0
        || (warm.required != (receipt.matrix.process_temperature == "warm"))
        || (warm.required && (!warm.completed || warm.suite_sha256.len() != 64))
        || (!warm.required && (warm.completed || !warm.suite_sha256.is_empty()))
        || warm.required && warm.session_id != receipt.provenance.campaign_session_id
        || warm.required
            && (warm.cache_state_version == 0
                || warm.cache_state_version > receipt.provenance.campaign_cache_state_version)
        || !warm.required && (!warm.session_id.is_empty() || warm.cache_state_version != 0)
    {
        return Err("warm worker discipline evidence failed".into());
    }
    if warm.required {
        let expected_suite_sha256 = warmup_probe_suite_sha256(
            &warm.session_id,
            warm.worker_pid,
            &receipt.timings.compile_attribution.probe_evidence,
        )?;
        if warm.suite_sha256 != expected_suite_sha256 {
            return Err("warmup suite seal is not bound to compile probe evidence".into());
        }
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
    match (receipt.mode.as_str(), &receipt.compression) {
        ("compressed", Some(compression)) => validate_receipt_compression(receipt, compression)?,
        ("dense", None) => {}
        _ => {
            return Err(
                "compression evidence must be present exactly on compressed receipts".into(),
            )
        }
    }
    // Last: every integrity, identity, safety, and fixture check above refuses first; the gate
    // record is then checked against the measured values.
    validate_quality_gate(receipt)
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
    receipt: Receipt,
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
    receipt.receipt_sha256 = receipt_semantic_seal(&receipt)?;
    let bytes = receipt.bytes().map_err(|e| e.to_string())?;
    let released_cache_bytes = receipt
        .memory
        .allocation_events
        .iter()
        .filter(|event| event.role == "cache" && event.lifetime == "released")
        .map(|event| event.bytes)
        .max()
        .unwrap_or(0);
    let compile_probes = receipt
        .timings
        .compile_attribution
        .probe_durations_ms
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    // Compressed rows state their measured quality-gate outcome; dense bytes are unchanged.
    let quality_gate = receipt
        .quality
        .quality_gate
        .as_ref()
        .map(|gate| format!("- Quality gate: {}\n", quality_gate_summary(gate)))
        .unwrap_or_default();
    let human = format!(
        "# {} KV receipt\n\n- Run: {}\n- Mode: {}\n- Released cache ownership bytes: {}\n- Compile attribution: {} / {} / {}\n- Compile probes: {} ms\n- First dispatch excess: {} ms\n- Compile cost: {}\n- Steady dispatch: {} ms\n{quality_gate}- Receipt hash: {}\n",
        if receipt.mode == "dense" {
            "Dense"
        } else {
            "Compressed"
        },
        receipt.run_id,
        receipt.mode,
        released_cache_bytes,
        receipt.timings.compile_attribution.method,
        receipt.timings.compile_attribution.operation,
        receipt.timings.compile_attribution.source,
        compile_probes,
        receipt.timings.compile_attribution.first_dispatch_excess_ms,
        compile_cost_summary(&receipt.timings.compile_attribution),
        receipt.timings.warm_compile_ms,
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
    if bundle.fixtures.len() != REQUIRED_FIXTURES.len() * QUALITY_MEASUREMENTS {
        return Err(
            "complete product evidence requires every fixture of the row's quality measurement"
                .into(),
        );
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
        .is_none_or(|v| v.len() != 8)
        || memory["allocationEvents"]
            .as_array()
            .is_none_or(|v| v.is_empty())
    {
        return Err("receipt memory evidence is incomplete".into());
    }
    let timings = object["timings"]
        .as_object()
        .ok_or("timings is not an object")?;
    if timings["samples"]
        .as_array()
        .is_none_or(|v| v.len() != TIMING_REPEATS)
    {
        return Err("receipt timing samples are incomplete".into());
    }
    let quality = object["quality"]
        .as_object()
        .ok_or("quality is not an object")?;
    if quality["fixtureEvidence"]
        .as_object()
        .is_none_or(|v| v.len() != 4)
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
        validate_fixture_binding(&typed, name, &artifact.bytes, 0)?;
    }
    let mut repeat_artifacts = Vec::with_capacity(QUALITY_MEASUREMENTS * REQUIRED_FIXTURES.len());
    for repeat in 0..QUALITY_MEASUREMENTS {
        for fixture in REQUIRED_FIXTURES {
            let artifact_name = fixture_artifact_name(fixture, repeat);
            let artifact = bundle
                .fixtures
                .iter()
                .find(|artifact| artifact.name == artifact_name)
                .ok_or_else(|| format!("missing repeat-bound fixture {artifact_name}"))?;
            validate_fixture_binding(&typed, fixture, &artifact.bytes, repeat)?;
            repeat_artifacts.push((fixture, artifact.bytes.as_slice()));
        }
    }
    if typed.mode == "compressed" {
        let repeats = sealed_repeat_quality_metrics(&typed, &repeat_artifacts)?;
        validate_sealed_quality_gate(&typed, &repeats)?;
    }
    validate_repeat_discrimination(&typed, repeat_artifacts)?;
    let mut core = value.clone();
    core.as_object_mut()
        .ok_or("receipt is not an object")?
        .remove("receiptSha256");
    let expected = seal_bytes(&canonical_semantic_seal_bytes(&core)?);
    if object.get("receiptSha256").and_then(|v| v.as_str()) != Some(expected.as_str()) {
        return Err("receipt semantic hash mismatch".into());
    }
    let human = std::str::from_utf8(&bundle.human).map_err(|_| "human receipt is not UTF-8")?;
    if !human.contains(&format!("- Receipt hash: {expected}\n")) {
        return Err("human receipt is not bound to the sealed JSON receipt".into());
    }
    Ok(())
}

/// Re-derive each measured repeat's quality metrics from the raw evidence of its sealed fixture
/// artifacts (`artifacts` is repeat-major, [`REQUIRED_FIXTURES`] order within a repeat). A
/// compressed repeat's kernel evidence carries the row's forced continuation, which must be the
/// receipt's.
fn sealed_repeat_quality_metrics(
    receipt: &Receipt,
    artifacts: &[(&str, &[u8])],
) -> Result<Vec<QualityMetrics>, String> {
    if artifacts.len() != QUALITY_MEASUREMENTS * REQUIRED_FIXTURES.len() {
        return Err(
            "sealed quality evidence requires every fixture of the quality measurement".into(),
        );
    }
    artifacts
        .chunks(REQUIRED_FIXTURES.len())
        .enumerate()
        .map(|(repeat, fixtures)| {
            let mut metrics = QualityMetrics {
                parity_max_error: 0.0,
                perplexity_delta: 0.0,
                greedy_token_agreement: 0.0,
                structured_tool_agreement: 0.0,
                needle_retrieval: 0.0,
                multi_turn_prompt_cache: 0.0,
            };
            for ((name, bytes), expected) in fixtures.iter().zip(REQUIRED_FIXTURES) {
                if *name != expected {
                    return Err(format!("repeat {repeat} fixture order differs at {name}"));
                }
                let value: serde_json::Value = serde_json::from_slice(bytes)
                    .map_err(|e| format!("fixture {name} repeat {repeat} JSON: {e}"))?;
                let evidence = value
                    .get("evidence")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| format!("fixture {name} repeat {repeat} lacks evidence"))?;
                let number = |key: &str| {
                    evidence
                        .get(key)
                        .and_then(serde_json::Value::as_f64)
                        .filter(|value| value.is_finite())
                        .ok_or_else(|| {
                            format!("fixture {name} repeat {repeat} lacks finite evidence {key}")
                        })
                };
                let ratio = |matches: &str, total: &str| -> Result<f64, String> {
                    let (matches, total) = (number(matches)?, number(total)?);
                    if total < 1.0 || matches < 0.0 || matches > total {
                        return Err(format!(
                            "fixture {name} repeat {repeat} evidence has {matches} of {total} matches"
                        ));
                    }
                    Ok(matches / total)
                };
                match *name {
                    "kernel-fp32-reference" => {
                        let forced = evidence
                            .get("forcedContinuation")
                            .map(|value| {
                                serde_json::from_value::<ReceiptForcedContinuation>(value.clone())
                            })
                            .transpose()
                            .map_err(|e| format!("repeat {repeat} forced continuation: {e}"))?;
                        if forced.as_ref() != receipt.quality.forced_continuation.as_ref() {
                            return Err(format!(
                                "repeat {repeat} kernel fixture forced continuation is not the receipt's"
                            ));
                        }
                        if forced.as_ref().is_some_and(|forced| {
                            number("greedyMatches").ok() != Some(forced.matches as f64)
                                || number("greedyTotal").ok() != Some(forced.tokens as f64)
                        }) {
                            return Err(format!(
                                "repeat {repeat} kernel greedy counts are not its forced continuation's"
                            ));
                        }
                        metrics.greedy_token_agreement = ratio("greedyMatches", "greedyTotal")?;
                        metrics.perplexity_delta =
                            number("candidatePerplexity")? - number("referencePerplexity")?;
                        metrics.parity_max_error = evidence
                            .get("parityErrors")
                            .and_then(serde_json::Value::as_array)
                            .filter(|errors| !errors.is_empty())
                            .ok_or_else(|| format!("repeat {repeat} kernel fixture lacks parity errors"))?
                            .iter()
                            .map(|error| {
                                error.as_f64().filter(|e| e.is_finite() && *e >= 0.0).ok_or_else(|| {
                                    format!("repeat {repeat} kernel parity error is not finite")
                                })
                            })
                            .try_fold(0.0_f64, |maximum, error| Ok::<_, String>(maximum.max(error?)))?;
                    }
                    "structured-tool-call" => {
                        metrics.structured_tool_agreement = ratio("matches", "total")?
                    }
                    "long-context-needle" => metrics.needle_retrieval = ratio("matches", "total")?,
                    _ => {
                        let forced = evidence
                            .get("forcedContinuation")
                            .map(|value| {
                                serde_json::from_value::<ReceiptForcedContinuation>(value.clone())
                            })
                            .transpose()
                            .map_err(|e| format!("repeat {repeat} turn-2 forced continuation: {e}"))?;
                        if forced.as_ref()
                            != receipt.quality.multi_turn_forced_continuation.as_ref()
                        {
                            return Err(format!(
                                "repeat {repeat} multi-turn fixture forced continuation is not the receipt's"
                            ));
                        }
                        if forced.as_ref().is_some_and(|forced| {
                            number("matches").ok() != Some(forced.matches as f64)
                                || number("total").ok() != Some(forced.tokens as f64)
                        }) {
                            return Err(format!(
                                "repeat {repeat} multi-turn counts are not its turn-2 forced continuation's"
                            ));
                        }
                        if evidence.get("method").and_then(serde_json::Value::as_str)
                            != Some(MULTI_TURN_PROMPT_CACHE_METHOD)
                        {
                            return Err(format!(
                                "repeat {repeat} multi-turn fixture is not teacher-forced on a cache-hit turn 2"
                            ));
                        }
                        let mut sealed = ReceiptMultiTurnCache::default();
                        for (arm, slot) in [
                            ("candidate", &mut sealed.candidate),
                            ("reference", &mut sealed.reference),
                        ] {
                            *slot = evidence
                                .get("turns")
                                .and_then(|turns| turns.get(arm))
                                .cloned()
                                .ok_or_else(|| format!("repeat {repeat} {arm} has no turn records"))
                                .and_then(|turns| {
                                    serde_json::from_value::<MultiTurnPromptCacheTurns>(turns)
                                        .map_err(|e| format!("repeat {repeat} {arm} turns: {e}"))
                                })?;
                            slot.validate(arm)?;
                        }
                        // The receipt's per-turn records are the primary repeat's sealed ones.
                        if repeat == 0 && sealed != receipt.quality.multi_turn_cache {
                            return Err(
                                "receipt multiTurnCache is not the primary repeat's sealed turn records"
                                    .into(),
                            );
                        }
                        let pass = evidence
                            .get("forcedPass")
                            .map(|value| {
                                serde_json::from_value::<MultiTurnForcedPass>(value.clone())
                            })
                            .transpose()
                            .map_err(|e| format!("repeat {repeat} turn-2 forced pass: {e}"))?;
                        if pass.as_ref() != receipt.quality.multi_turn_forced_pass.as_ref() {
                            return Err(format!(
                                "repeat {repeat} multi-turn forced-pass turn records are not the receipt's"
                            ));
                        }
                        metrics.multi_turn_prompt_cache = ratio("matches", "total")?
                    }
                }
            }
            Ok(metrics)
        })
        .collect()
}

/// A compressed receipt's recorded quality must be its sealed repeats': per-repeat greedy
/// agreement, the primary repeat's receipt-level values, and a quality gate exactly equal to the
/// frozen-threshold evaluation of every repeat (so a failing value can never be recorded as a pass).
/// Kernel parity is refused here as in [`validate_receipt_semantics`].
fn validate_sealed_quality_gate(
    receipt: &Receipt,
    repeats: &[QualityMetrics],
) -> Result<(), String> {
    let quality = &receipt.quality;
    for (repeat, metrics) in repeats.iter().enumerate() {
        if metrics.parity_max_error > COMPRESSED_PARITY_MAX_ERROR {
            return Err(format!(
                "kernel parity failed: metric=parityMaxError value={} threshold={COMPRESSED_PARITY_MAX_ERROR} comparison=maximum fixture=kernel-fp32-reference repeat={repeat}",
                metrics.parity_max_error
            ));
        }
        if quality
            .greedy_token_agreement_by_repeat
            .get(repeat)
            .map(|value| value.to_bits())
            != Some(metrics.greedy_token_agreement.to_bits())
        {
            return Err(format!(
                "greedyTokenAgreement repeat {repeat} is not its sealed agreement {}",
                metrics.greedy_token_agreement
            ));
        }
    }
    let primary = repeats.first().ok_or("no sealed repeats")?;
    for (metric, recorded, sealed) in [
        (
            "parityMaxError",
            quality.parity_max_error,
            primary.parity_max_error,
        ),
        (
            "perplexityDelta",
            quality.perplexity_delta,
            primary.perplexity_delta,
        ),
        (
            "structuredToolAgreement",
            quality.structured_tool_agreement,
            primary.structured_tool_agreement,
        ),
        (
            "needleRetrieval",
            quality.needle_retrieval,
            primary.needle_retrieval,
        ),
        (
            "multiTurnPromptCache",
            quality.multi_turn_prompt_cache,
            primary.multi_turn_prompt_cache,
        ),
    ] {
        if recorded.to_bits() != sealed.to_bits() {
            return Err(format!(
                "receipt {metric} {recorded} is not the primary repeat's sealed value {sealed}"
            ));
        }
    }
    let expected = quality_gate_from_repeats(repeats, quality.needle_discriminating);
    if quality.quality_gate.as_ref() != Some(&expected) {
        return Err(format!(
            "receipt quality gate ({}) is not the gate of its sealed repeats ({})",
            quality
                .quality_gate
                .as_ref()
                .map_or_else(|| "absent".into(), quality_gate_summary),
            quality_gate_summary(&expected)
        ));
    }
    Ok(())
}

/// Every repeat's tool/needle outcomes must derive its own metric and flag, and the receipt's
/// discrimination flags must be exactly the AND over all sealed repeats.
fn validate_repeat_discrimination<'a>(
    receipt: &Receipt,
    artifacts: impl IntoIterator<Item = (&'a str, &'a [u8])>,
) -> Result<(), String> {
    let (mut needle, mut tool) = (true, true);
    for (fixture, bytes) in artifacts {
        match (fixture, fixture_discrimination(receipt, fixture, bytes)?) {
            ("long-context-needle", Some(flag)) => needle &= flag,
            ("structured-tool-call", Some(flag)) => tool &= flag,
            _ => {}
        }
    }
    if receipt.quality.needle_discriminating != needle
        || receipt.quality.tool_discriminating != tool
    {
        return Err("receipt discrimination is not the AND of its sealed repeats".into());
    }
    Ok(())
}

/// Re-derive a structured-tool or needle artifact's metric and discrimination flag from the raw
/// outcomes it records, and return the per-repeat flag.
fn fixture_discrimination(
    receipt: &Receipt,
    name: &str,
    bytes: &[u8],
) -> Result<Option<bool>, String> {
    if !matches!(name, "structured-tool-call" | "long-context-needle") {
        return Ok(None);
    }
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("fixture {name} JSON: {e}"))?;
    let evidence = value
        .get("evidence")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("fixture {name} lacks outcome evidence"))?;
    let flag = |key: &str| {
        evidence
            .get(key)
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| format!("fixture {name} lacks boolean {key}"))
    };
    let count = |key: &str| {
        evidence
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| format!("fixture {name} lacks count {key}"))
    };
    let compressed = receipt.mode == "compressed";
    let discriminating = flag("discriminating")?;
    let outputs_match = flag("outputsMatch")?;
    let (expected_discriminating, expected_match) = if name == "long-context-needle" {
        let candidate = flag("candidateRecovered")?;
        let reference = flag("referenceRecovered")?;
        let same_weights_dense = if compressed { reference } else { candidate };
        // Contract v5: retrieval is the candidate's own recovery either way; a non-discriminating
        // one is an ungated observation (`outputsMatch` stays recorded beside it).
        (same_weights_dense, candidate)
    } else {
        let candidate = flag("candidateValid")?;
        let reference = flag("referenceValid")?;
        (
            if compressed { reference } else { candidate },
            outputs_match,
        )
    };
    let (matches, total) = (count("matches")?, count("total")?);
    // A compressed repeat that validly measured a miss is gate evidence (see
    // `validate_sealed_quality_gate`), not a malformed artifact.
    if total != 1
        || discriminating != expected_discriminating
        || matches != u64::from(expected_match)
    {
        return Err(format!(
            "fixture {name} outcome evidence does not derive its metric and discrimination"
        ));
    }
    Ok(Some(discriminating))
}

fn validate_fixture_binding(
    receipt: &Receipt,
    name: &str,
    bytes: &[u8],
    expected_repeat: usize,
) -> Result<(), String> {
    let value: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|e| format!("fixture {name} JSON: {e}"))?;
    let binding = value
        .get("binding")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("fixture {name} lacks producer binding"))?;
    let expected_coordinate = coordinate_slug(&Coordinate {
        family: match receipt.matrix.family.as_str() {
            "llama" => "llama",
            "qwen" => "qwen",
            _ => return Err("fixture receipt family".into()),
        },
        context_band: match receipt.matrix.context_band.as_str() {
            "short" => "short",
            "medium" => "medium",
            "memory-material" => "memory-material",
            "fit-boundary" => "fit-boundary",
            _ => return Err("fixture receipt context".into()),
        },
        request_mode: match receipt.matrix.request_mode.as_str() {
            "single" => "single",
            "supported-batch" => "supported-batch",
            _ => return Err("fixture receipt request mode".into()),
        },
        prefill_mode: match receipt.matrix.prefill_mode.as_str() {
            "chunked" => "chunked",
            "single-shot" => "single-shot",
            _ => return Err("fixture receipt prefill mode".into()),
        },
        process_temperature: match receipt.matrix.process_temperature.as_str() {
            "cold" => "cold",
            "warm" => "warm",
            _ => return Err("fixture receipt temperature".into()),
        },
    });
    if binding
        .get("coordinate")
        .and_then(serde_json::Value::as_str)
        != Some(expected_coordinate.as_str())
        || binding.get("repeat").and_then(serde_json::Value::as_u64) != Some(expected_repeat as u64)
    {
        return Err(format!("fixture {name} coordinate/repeat binding mismatch"));
    }
    let candidate = binding
        .get("candidate")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("fixture {name} candidate binding"))?;
    let reference = binding
        .get("reference")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| format!("fixture {name} reference binding"))?;
    fn field<'a>(
        object: &'a serde_json::Map<String, serde_json::Value>,
        key: &str,
    ) -> Option<&'a str> {
        object.get(key).and_then(serde_json::Value::as_str)
    }
    // Contract v3: a compressed row's denominator is bound (not merely labelled) to the dense-KV
    // run on the candidate's own weights; a dense row's reference is the bf16 model.
    let reference_inventory = if receipt.mode == "compressed" {
        receipt.provenance.model_file_sha256.as_str()
    } else {
        receipt.provenance.reference_model_sha256.as_str()
    };
    let reference_session = field(reference, "coordinateSessionId")
        .filter(|session| {
            session.len() == 64 && session.bytes().all(|byte| byte.is_ascii_hexdigit())
        })
        .ok_or_else(|| format!("fixture {name} lacks per-run reference session evidence"))?;
    let expected_operation = if receipt.matrix.request_mode == "supported-batch" {
        "supported-batch"
    } else if receipt.matrix.prefill_mode == "chunked" {
        "chunked-prefix-reuse"
    } else {
        "single-shot-generation"
    };
    let needs_secondary = receipt.matrix.request_mode == "supported-batch"
        && receipt.matrix.prefill_mode == "chunked";
    let valid_secondary = |object: &serde_json::Map<String, serde_json::Value>| {
        let secondary = object
            .get("secondaryOperation")
            .and_then(serde_json::Value::as_object);
        if !needs_secondary {
            return secondary.is_none();
        }
        secondary.is_some_and(|secondary| {
            secondary
                .get("operation")
                .and_then(serde_json::Value::as_str)
                == Some("chunked-prefix-reuse")
                && secondary
                    .get("outputSha256")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|hash| {
                        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
                && secondary
                    .get("evidenceSha256")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|hash| {
                        hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
                && secondary
                    .get("sessionId")
                    .and_then(serde_json::Value::as_str)
                    == field(object, "coordinateSessionId")
                && secondary
                    .get("cacheStateVersion")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|version| version > 0)
                && secondary
                    .get("generatedTokens")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|tokens| tokens > 0)
                && secondary
                    .get("promptTokens")
                    .and_then(serde_json::Value::as_u64)
                    .is_some_and(|tokens| tokens > 0)
        })
    };
    if field(candidate, "coordinateInventorySha256")
        != Some(receipt.provenance.model_file_sha256.as_str())
        || field(candidate, "qualityInventorySha256")
            != Some(receipt.provenance.model_file_sha256.as_str())
        || field(candidate, "coordinateSessionId")
            != Some(receipt.provenance.campaign_session_id.as_str())
        || field(candidate, "qualitySessionId")
            != Some(receipt.provenance.campaign_session_id.as_str())
        || field(candidate, "operation") != Some(expected_operation)
        || field(candidate, "operationOutputSha256").is_none_or(|hash| hash.len() != 64)
        || field(candidate, "operationEvidenceSha256").is_none_or(|hash| hash.len() != 64)
        || field(candidate, "coordinateEvidenceSha256").is_none_or(|hash| hash.len() != 64)
        || field(candidate, "qualityTranscriptSha256").is_none_or(|hash| hash.len() != 64)
        || !valid_secondary(candidate)
        || field(reference, "coordinateInventorySha256") != Some(reference_inventory)
        || field(reference, "qualityInventorySha256") != Some(reference_inventory)
        || field(reference, "qualitySessionId") != Some(reference_session)
        || field(reference, "operation") != Some(expected_operation)
        || field(reference, "operationOutputSha256").is_none_or(|hash| hash.len() != 64)
        || field(reference, "operationEvidenceSha256").is_none_or(|hash| hash.len() != 64)
        || field(reference, "coordinateEvidenceSha256").is_none_or(|hash| hash.len() != 64)
        || field(reference, "qualityTranscriptSha256").is_none_or(|hash| hash.len() != 64)
        || !valid_secondary(reference)
    {
        return Err(format!("fixture {name} producer binding mismatch"));
    }
    if name == "kernel-fp32-reference"
        && field(candidate, "coordinateEvidenceSha256")
            != Some(receipt.provenance.coordinate_operation_sha256.as_str())
    {
        return Err("receipt coordinate evidence digest is not fixture-bound".into());
    }
    if name == "kernel-fp32-reference" && receipt.matrix.process_temperature == "cold" {
        let probe = receipt
            .timings
            .compile_attribution
            .probe_evidence
            .get(expected_repeat)
            .ok_or("cold kernel fixture has no matching compile probe evidence")?;
        let setup_ms = candidate
            .get("compileSetupMs")
            .and_then(serde_json::Value::as_f64)
            .ok_or("cold kernel fixture lacks compileSetupMs")?;
        let dispatch_ms = candidate
            .get("compileDispatchMs")
            .and_then(serde_json::Value::as_f64)
            .ok_or("cold kernel fixture lacks compileDispatchMs")?;
        if setup_ms != probe.setup_ms
            || dispatch_ms != probe.dispatch_ms
            || field(candidate, "operationEvidenceSha256")
                != Some(probe.operation_evidence_sha256.as_str())
        {
            return Err("cold compile probe evidence is not bound to its repeat fixture".into());
        }
    }
    if name == "kernel-fp32-reference" && receipt.matrix.process_temperature == "warm" {
        if let Some(samples) = &receipt.timings.compile_attribution.noise_samples_ms {
            let dispatch_ms = candidate
                .get("compileDispatchMs")
                .and_then(serde_json::Value::as_f64)
                .ok_or("warm kernel fixture lacks compileDispatchMs")?;
            if samples.get(expected_repeat) != Some(&dispatch_ms) {
                return Err("warm compile noise sample is not bound to its repeat fixture".into());
            }
        }
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

pub fn campaign_global_identity(receipt: &Receipt) -> Result<Vec<u8>, String> {
    let provenance = &receipt.provenance;
    canonical_json_bytes(&serde_json::json!({
        "sceneWorksRepository": provenance.scene_works_repository,
        "inferenceRepository": provenance.inference_repository,
        "sceneWorksRevision": provenance.scene_works_revision,
        "inferenceRevision": provenance.inference_revision,
        "mlxVersion": provenance.mlx_version,
        "mlxSource": provenance.mlx_source,
        "mlxRevision": provenance.mlx_revision,
        "dependencyLockSha256": provenance.dependency_lock_sha256,
        "os": provenance.os,
        "xcode": provenance.xcode,
        "hardware": provenance.hardware,
        "commandTemplate": provenance.command_template,
    }))
    .map_err(|error| error.to_string())
}

/// Recorded in the campaign manifest, never refused: rows started in more than one power mode or
/// thermal state, or some row's host state changed during the row.
pub fn campaign_host_state_varied<'a>(receipts: impl IntoIterator<Item = &'a Receipt>) -> bool {
    let mut starts = std::collections::BTreeSet::new();
    let mut changed = false;
    for receipt in receipts {
        let p = &receipt.provenance;
        starts.insert((p.power_mode.clone(), p.thermal_state.clone()));
        changed |= p.thermal_changed_during_row || p.power_mode_changed_during_row;
    }
    changed || starts.len() > 1
}

pub fn campaign_family_identity(receipt: &Receipt) -> Result<Vec<u8>, String> {
    let provenance = &receipt.provenance;
    let geometry = &receipt.geometry;
    canonical_json_bytes(&serde_json::json!({
        "family": receipt.matrix.family,
        "modelId": provenance.model_id,
        "modelFileSha256": provenance.model_file_sha256,
        "modelFileBytes": provenance.model_file_bytes,
        "referenceModelId": provenance.reference_model_id,
        "referenceModelSha256": provenance.reference_model_sha256,
        "referenceModelBytes": provenance.reference_model_bytes,
        "queryHeads": geometry.query_heads,
        "kvHeads": geometry.kv_heads,
        "headDimension": geometry.head_dimension,
        "layers": geometry.layers,
        "elementBytes": geometry.element_bytes,
        "contextWindowTokens": geometry.context_window_tokens,
    }))
    .map_err(|error| error.to_string())
}

/// A refused estimate must exceed the row deadline by this factor, so estimation error alone
/// never refuses a row that would have finished.
pub const DURATION_REFUSAL_MARGIN: f64 = 1.25;
/// Assumed bits per parameter of the 4-bit candidate's MLX weights (4-bit codes plus group
/// scales/biases). It converts weight bytes into a parameter count for the attention crossover.
const CANDIDATE_BITS_PER_PARAMETER: f64 = 4.5;

/// Row shape as it drives work: how many measurement units run and how many full-context prefills
/// the row issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowWork {
    /// Measurement units (contract v5): the candidate's timing runs (`warmups + 5 repeats`) and
    /// the two arms' one quality measurement each.
    pub units: u64,
    /// Full-context prefills: every timing run is the kernel coordinate operation (2 sequences for
    /// supported-batch, plus the secondary chunked operation on a batch+chunked row) and a steady
    /// decode; each arm's quality measurement runs 4 fixtures, each a quality request plus the
    /// coordinate operation, except the candidate's kernel fixture, which reuses timing repeat 0's
    /// coordinate operation; then the dense forced continuation and the candidate's teacher-forced
    /// pass.
    pub prefills: u64,
}

pub fn row_work(process_temperature: &str, request_mode: &str, prefill_mode: &str) -> RowWork {
    let warmups = if process_temperature == "warm" {
        TIMING_WARMUPS as u64
    } else {
        0
    };
    let timing_runs = warmups + TIMING_REPEATS as u64;
    let quality_halves = 2 * QUALITY_MEASUREMENTS as u64;
    let operation_units = match (request_mode, prefill_mode) {
        ("supported-batch", "chunked") => 3,
        ("supported-batch", _) => 2,
        _ => 1,
    };
    RowWork {
        units: timing_runs + quality_halves,
        prefills: timing_runs * (operation_units + 1) + quality_halves * 4 * (1 + operation_units)
            - operation_units
            + 2,
    }
}

/// Prefill cost of a `tokens`-long context, up to a constant: linear (projections and MLP) plus
/// quadratic attention, equal at `crossover` tokens. This sets the effective exponent between 1
/// and 2 from the model itself.
pub fn prefill_scale(tokens: f64, crossover: f64) -> f64 {
    tokens * (1.0 + tokens / crossover)
}

/// A completed row's measurements that the pre-row estimate scales from.
#[derive(Clone, Debug, PartialEq)]
pub struct RowDurationBasis {
    pub coordinate: String,
    pub family: String,
    pub context_window_tokens: u64,
    pub context_target_tokens: u64,
    /// Row-start to row-end host-state span.
    pub wall_seconds: f64,
    /// One sequence's full-context prefill: the measured coordinate prefill over its batch.
    pub prefill_seconds: f64,
    pub work: RowWork,
    /// Context length at which attention matches the linear per-token cost:
    /// `parameters / (2 * layers * query_heads * head_dimension)`.
    pub attention_crossover_tokens: f64,
}

/// The basis a completed receipt provides. A chunked row's measured prefill is the short suffix
/// after a prefix hit, not a full-context prefill, so it provides none.
pub fn row_duration_basis(receipt: &Receipt) -> Option<RowDurationBasis> {
    let [start, end] = receipt.provenance.host_states.as_slice() else {
        return None;
    };
    if receipt.matrix.prefill_mode == "chunked" {
        return None;
    }
    let wall_seconds =
        utc_timestamp_seconds(&end.captured_at)? - utc_timestamp_seconds(&start.captured_at)?;
    let geometry = &receipt.geometry;
    let prefill_seconds = receipt.timings.prefill_ms / 1_000.0 / geometry.batch.max(1) as f64;
    let parameters = receipt.memory.model_weights_bytes as f64 * 8.0 / CANDIDATE_BITS_PER_PARAMETER;
    let attention_width = 2 * geometry.layers * geometry.query_heads * geometry.head_dimension;
    let valid = wall_seconds.is_finite()
        && wall_seconds > 0.0
        && prefill_seconds.is_finite()
        && prefill_seconds > 0.0
        && geometry.context_target_tokens > 0
        && attention_width > 0
        && parameters > 0.0;
    valid.then(|| RowDurationBasis {
        coordinate: format!(
            "{}-{}-{}-{}-{}",
            receipt.matrix.family,
            receipt.matrix.context_band,
            receipt.matrix.request_mode,
            receipt.matrix.prefill_mode,
            receipt.matrix.process_temperature
        ),
        family: receipt.matrix.family.clone(),
        context_window_tokens: geometry.context_window_tokens,
        context_target_tokens: geometry.context_target_tokens,
        wall_seconds,
        prefill_seconds,
        work: row_work(
            &receipt.matrix.process_temperature,
            &receipt.matrix.request_mode,
            &receipt.matrix.prefill_mode,
        ),
        attention_crossover_tokens: parameters / attention_width as f64,
    })
}

/// A pre-row duration estimate and its parts.
#[derive(Clone, Debug, PartialEq)]
pub struct RowDurationEstimate {
    pub seconds: f64,
    /// Measured context-independent time (loads, fixtures' decode, steady decode, overhead),
    /// scaled only by the fixture-half count.
    pub fixed_seconds: f64,
    /// Full-context prefills, scaled by [`prefill_scale`].
    pub prefill_seconds: f64,
    pub basis_coordinate: String,
}

/// Estimate a row from the largest-context completed row of the same family.
///
/// The basis row's wall time is split into its measured prefill part
/// (`prefills x per-sequence prefill`) and the fixed remainder. The fixed part is scaled only by
/// the fixture-half ratio; the prefill part is scaled by the target row's prefill count and
/// `prefill_scale(target) / prefill_scale(basis)`. The reference arm is assumed to prefill at
/// the candidate's rate. `None` when no same-family row with a full-context prefill has completed.
pub fn estimate_row_seconds(
    family: &str,
    context_band: &str,
    target: RowWork,
    basis: &[RowDurationBasis],
) -> Option<RowDurationEstimate> {
    let row = basis
        .iter()
        .filter(|row| row.family == family)
        .max_by_key(|row| row.context_target_tokens)?;
    let target_tokens = context_band_target(row.context_window_tokens, context_band).ok()? as f64;
    let basis_prefill = row.work.prefills as f64 * row.prefill_seconds;
    let basis_fixed = (row.wall_seconds - basis_prefill).max(0.0);
    let fixed_seconds = basis_fixed * target.units as f64 / row.work.units as f64;
    let prefill_seconds = target.prefills as f64
        * row.prefill_seconds
        * prefill_scale(target_tokens, row.attention_crossover_tokens)
        / prefill_scale(
            row.context_target_tokens as f64,
            row.attention_crossover_tokens,
        );
    Some(RowDurationEstimate {
        seconds: fixed_seconds + prefill_seconds,
        fixed_seconds,
        prefill_seconds,
        basis_coordinate: row.coordinate.clone(),
    })
}

/// Whether an estimate refuses a row: it must exceed the deadline by [`DURATION_REFUSAL_MARGIN`].
pub fn duration_estimate_refuses(estimate: &RowDurationEstimate, deadline_seconds: u64) -> bool {
    estimate.seconds > deadline_seconds as f64 * DURATION_REFUSAL_MARGIN
}

/// Atomically publish the whole eight-coordinate receipt collection. Individual worker output is
/// intentionally not a campaign result; only this function creates `destination`, and it does so
/// after every receipt, sidecar, coordinate, and product-owned worker PID has been validated.
pub fn publish_complete_campaign(
    destination: &Path,
    prepared: &[PreparedCoordinateReceipt],
    resume_identity: &serde_json::Value,
    policy: &CampaignSafetyPolicy,
) -> Result<(), String> {
    publish_campaign(destination, prepared, resume_identity, policy, None)
}

/// Publish a complete campaign, or (`only_coordinate`) the single row of a partial run under a
/// manifest marked partial and non-publishable that no campaign loader accepts.
fn publish_campaign(
    destination: &Path,
    prepared: &[PreparedCoordinateReceipt],
    resume_identity: &serde_json::Value,
    policy: &CampaignSafetyPolicy,
    only_coordinate: Option<&str>,
) -> Result<(), String> {
    let policy_sha256 = policy.seal()?;
    if destination.exists() {
        return Err("campaign destination must be absent for atomic publication".into());
    }
    if resume_identity
        .get("onlyCoordinate")
        .and_then(serde_json::Value::as_str)
        != only_coordinate
    {
        return Err(
            "publication scope differs from the resume identity's coordinate filter".into(),
        );
    }
    let full_schedule = required_schedule();
    let schedule = selected_schedule_indices(only_coordinate)?
        .into_iter()
        .map(|index| full_schedule[index].clone())
        .collect::<Vec<_>>();
    if prepared.len() != schedule.len() {
        return Err(match only_coordinate {
            None => "complete campaign requires exactly eight prepared receipts".into(),
            Some(slug) => {
                format!("partial run of {slug} requires exactly its one prepared receipt")
            }
        });
    }
    let mut outcomes = Vec::with_capacity(prepared.len());
    let mut seen = std::collections::BTreeSet::new();
    let mut global_identity = None;
    let mut family_identities = std::collections::BTreeMap::new();
    let mut campaign_mode = None;
    for item in prepared {
        validate_artifact_bundle(&item.bundle)?;
        let receipt: Receipt = serde_json::from_slice(&item.bundle.receipt)
            .map_err(|e| format!("prepared receipt is not decodable: {e}"))?;
        receipt.memory.admission.validate_against(policy)?;
        let mode = (
            receipt.mode.clone(),
            receipt.compression.as_ref().map(|c| c.method.clone()),
        );
        if campaign_mode.get_or_insert_with(|| mode.clone()) != &mode {
            return Err("campaign mixes dense and compressed (or compressed-method) rows".into());
        }
        let identity = campaign_global_identity(&receipt)?;
        if global_identity
            .as_ref()
            .is_some_and(|expected| expected != &identity)
        {
            return Err("campaign source/toolchain/hardware identity drift".into());
        }
        global_identity.get_or_insert(identity);
        let family_identity = campaign_family_identity(&receipt)?;
        if family_identities
            .get(&receipt.matrix.family)
            .is_some_and(|expected| expected != &family_identity)
        {
            return Err(format!(
                "campaign model/reference identity drift within {}",
                receipt.matrix.family
            ));
        }
        family_identities
            .entry(receipt.matrix.family.clone())
            .or_insert(family_identity);
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
    match only_coordinate {
        None => validate_schedule_outcomes(&schedule, &outcomes)?,
        // A partial run is its one selected row, never a matrix.
        Some(_) => match (schedule.as_slice(), outcomes.as_slice()) {
            ([row], [(coordinate, discipline, pid)])
                if *coordinate == row.coordinate && *discipline == row.discipline && *pid != 0 => {}
            _ => return Err("partial run outcome does not equal its selected row".into()),
        },
    }

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
        let mut receipts = Vec::with_capacity(prepared.len());
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
            let mut row = serde_json::json!({
                "coordinate": slug,
                "receiptSha256": receipt.receipt_sha256,
                "workerPid": receipt.memory.phase_samples[0].pid,
                "files": files,
            });
            // Compressed rows publish their measured gate outcome; dense rows carry none.
            if let Some(gate) = &receipt.quality.quality_gate {
                row["qualityGatePassed"] = serde_json::json!(gate.passed);
            }
            manifest_rows.push(row);
            receipts.push(receipt);
        }
        let mut manifest = serde_json::json!({
            "schemaVersion": 2,
            "kind": if only_coordinate.is_some() { SC20671_PARTIAL_RUN_KIND } else { SC20671_CAMPAIGN_KIND },
            "scheduleVersion": SC20671_SCHEDULE_VERSION,
            "policySha256": policy_sha256,
            "resumeIdentitySha256": seal_bytes(&canonical_json_bytes(resume_identity).map_err(|e| e.to_string())?),
            "hostStateVaried": campaign_host_state_varied(&receipts),
            "coordinates": manifest_rows,
        });
        if let Some(passed) = campaign_quality_gate_passed(&receipts)? {
            manifest["qualityGatePassed"] = serde_json::json!(passed);
        }
        if let Some(slug) = only_coordinate {
            manifest["partial"] = serde_json::json!(true);
            manifest["publishable"] = serde_json::json!(false);
            manifest["onlyCoordinate"] = serde_json::json!(slug);
        }
        let (identity_bytes, identity_sha) = seal_json(resume_identity)?;
        let (policy_bytes, _) =
            seal_json(&serde_json::to_value(policy).map_err(|e| e.to_string())?)?;
        fs::write(staging.join("safety-policy.json"), policy_bytes).map_err(|e| e.to_string())?;
        fs::write(staging.join("resume-identity.json"), identity_bytes)
            .map_err(|e| e.to_string())?;
        fs::write(
            staging.join("resume-identity.json.sha256"),
            format!("{identity_sha}  resume-identity.json\n"),
        )
        .map_err(|e| e.to_string())?;
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

/// A compressed campaign's measured quality-gate outcome: passed only when every row's gate
/// passed. `None` for a dense campaign (characterization, never gated).
pub fn campaign_quality_gate_passed(receipts: &[Receipt]) -> Result<Option<bool>, String> {
    let gates = receipts
        .iter()
        .map(|receipt| receipt.quality.quality_gate.as_ref())
        .collect::<Vec<_>>();
    if gates.iter().all(Option::is_none) {
        return Ok(None);
    }
    gates
        .iter()
        .try_fold(true, |passed, gate| {
            gate.map(|gate| passed && gate.passed)
                .ok_or("campaign mixes gated and ungated rows")
        })
        .map(Some)
        .map_err(Into::into)
}

/// One stderr line per row plus the campaign verdict: the measured quality-gate outcomes a
/// Go/No-Go decision reads.
fn report_campaign_quality_gates(receipts: &[Receipt]) -> Result<(), String> {
    let Some(passed) = campaign_quality_gate_passed(receipts)? else {
        return Ok(());
    };
    for receipt in receipts {
        if let Some(gate) = &receipt.quality.quality_gate {
            eprintln!(
                "sc20671-kv-baseline: quality gate {}-{}-{}-{}-{}: {}",
                receipt.matrix.family,
                receipt.matrix.context_band,
                receipt.matrix.request_mode,
                receipt.matrix.prefill_mode,
                receipt.matrix.process_temperature,
                quality_gate_summary(gate)
            );
        }
    }
    eprintln!("sc20671-kv-baseline: compressed campaign qualityGatePassed={passed}");
    Ok(())
}

/// The one SC-20671 coordinate SC-20676 may bind for a family. It is deliberately the frozen
/// memory-material long-context row, not the campaign's near-fit calibration row.
pub const SC20676_BASELINE_CONTEXT_BAND: &str = "memory-material";

/// A sealed candidate baseline selected from a complete SC-20671 publication.
/// `inventory` is recomputed from the tested snapshot by the consumer and compared byte-for-byte.
#[derive(Clone, Debug)]
pub struct Sc20676BaselineRow {
    pub receipt: Receipt,
    pub coordinate: Coordinate,
}

fn manifest_file_bindings(
    bundle: &ArtifactBundle,
) -> std::collections::BTreeMap<String, (String, String)> {
    let mut files = std::collections::BTreeMap::new();
    files.insert(
        bundle.receipt_name.clone(),
        (
            seal_bytes(&bundle.receipt),
            seal_bytes(bundle.receipt_sidecar.as_bytes()),
        ),
    );
    files.insert(
        bundle.human_name.clone(),
        (
            seal_bytes(&bundle.human),
            seal_bytes(bundle.human_sidecar.as_bytes()),
        ),
    );
    for fixture in &bundle.fixtures {
        files.insert(
            fixture.name.clone(),
            (
                seal_bytes(&fixture.bytes),
                seal_bytes(fixture.sidecar.as_bytes()),
            ),
        );
    }
    files
}

/// Reconstruct and validate the exact published SC-20671 campaign, including every artifact
/// bundle named by the campaign manifest.  This is stricter than a receipt-only consumer: a
/// selected row is trusted only after all receipt/human/fixture bytes, sidecars, identities,
/// and product worker PIDs have been bound back to the whole campaign.
pub fn load_validated_complete_campaign(
    campaign_directory: &Path,
) -> Result<Vec<PreparedCoordinateReceipt>, String> {
    let manifest = fs::read(campaign_directory.join("campaign.json")).map_err(|e| e.to_string())?;
    let sidecar = fs::read_to_string(campaign_directory.join("campaign.json.sha256"))
        .map_err(|e| e.to_string())?;
    if sidecar != format!("{}  campaign.json\n", seal_bytes(&manifest)) {
        return Err("SC-20671 campaign manifest sidecar does not match exact bytes".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&manifest).map_err(|e| e.to_string())?;
    let (schedule, resume_identity, safety_policy) = match (
        value
            .get("schemaVersion")
            .and_then(serde_json::Value::as_u64),
        value.get("kind").and_then(serde_json::Value::as_str),
    ) {
        (Some(1), Some("sc-20671-complete-coordinate-set")) => {
            (legacy_required_schedule(), None, None)
        }
        (Some(2), Some(SC20671_CAMPAIGN_KIND))
            if value
                .get("scheduleVersion")
                .and_then(serde_json::Value::as_u64)
                == Some(SC20671_SCHEDULE_VERSION) =>
        {
            for key in ["policySha256", "resumeIdentitySha256"] {
                let digest = value
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .ok_or("SC-20671 v2 manifest lacks safety identity")?;
                if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return Err("SC-20671 v2 manifest has malformed safety identity".into());
                }
            }
            let identity_bytes = fs::read(campaign_directory.join("resume-identity.json"))
                .map_err(|e| e.to_string())?;
            let identity_sha = seal_bytes(&identity_bytes);
            if value
                .get("resumeIdentitySha256")
                .and_then(serde_json::Value::as_str)
                != Some(identity_sha.as_str())
                || fs::read_to_string(campaign_directory.join("resume-identity.json.sha256"))
                    .map_err(|e| e.to_string())?
                    != format!("{identity_sha}  resume-identity.json\n")
            {
                return Err("SC-20671 published resume identity bytes or sidecar drifted".into());
            }
            let identity: serde_json::Value =
                serde_json::from_slice(&identity_bytes).map_err(|e| e.to_string())?;
            if value.get("policySha256") != identity.get("policySha256")
                || identity
                    .get("scheduleVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(SC20671_SCHEDULE_VERSION)
                || identity
                    .get("schemaVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(1)
                || identity.get("kind").and_then(serde_json::Value::as_str)
                    != Some("sc-20671-resume-identity")
                || identity.get("coordinates")
                    != Some(&serde_json::json!(required_coordinates()
                        .iter()
                        .map(coordinate_slug)
                        .collect::<Vec<_>>()))
                || seal_json(&identity)?.0 != identity_bytes
            {
                return Err("SC-20671 published resume identity semantics drifted".into());
            }
            let policy_bytes = fs::read(campaign_directory.join("safety-policy.json"))
                .map_err(|e| e.to_string())?;
            let policy: CampaignSafetyPolicy =
                serde_json::from_slice(&policy_bytes).map_err(|e| e.to_string())?;
            policy.validate()?;
            if policy.seal()?.as_str()
                != value
                    .get("policySha256")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                || seal_json(&serde_json::to_value(&policy).map_err(|e| e.to_string())?)?.0
                    != policy_bytes
            {
                return Err("SC-20671 published safety policy bytes drifted".into());
            }
            (required_schedule(), Some(identity), Some(policy))
        }
        _ => return Err("SC-20671 campaign manifest identity is invalid".into()),
    };
    let rows = value
        .get("coordinates")
        .and_then(serde_json::Value::as_array)
        .ok_or("SC-20671 campaign manifest has no coordinate rows")?;
    if rows.len() != schedule.len() {
        return Err("SC-20671 campaign manifest is not a complete scheduled set".into());
    }
    let mut prepared = Vec::with_capacity(schedule.len());
    let mut seen = std::collections::BTreeSet::new();
    let mut global_identity = None;
    let mut family_identities = std::collections::BTreeMap::new();
    let mut outcomes = Vec::with_capacity(schedule.len());
    let mut host_state_receipts = Vec::with_capacity(schedule.len());
    for row in rows {
        let slug = row
            .get("coordinate")
            .and_then(serde_json::Value::as_str)
            .ok_or("campaign coordinate is missing")?;
        let expected_sha = row
            .get("receiptSha256")
            .and_then(serde_json::Value::as_str)
            .ok_or("campaign receipt seal is missing")?;
        if !seen.insert(slug.to_owned()) {
            return Err("SC-20671 campaign manifest repeats a coordinate".into());
        }
        let coordinate = schedule
            .iter()
            .find(|entry| coordinate_slug(&entry.coordinate) == slug)
            .map(|entry| entry.coordinate.clone())
            .ok_or("SC-20671 campaign manifest contains an unknown coordinate")?;
        let item =
            load_prepared_coordinate_receipt(&campaign_directory.join(slug), coordinate.clone())?;
        let receipt: Receipt =
            serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
        validate_sealed_receipt(&receipt)?;
        if let Some(identity) = &resume_identity {
            validate_receipt_launch_identity(&receipt, &coordinate, identity)?;
        }
        if let Some(policy) = &safety_policy {
            if receipt.geometry.context_target_tokens > policy.max_context_tokens
                || receipt.geometry.query_length > policy.max_request_tokens
            {
                return Err("SC-20671 published row exceeds declared safety ceilings".into());
            }
        }
        let worker_pid = receipt
            .memory
            .phase_samples
            .first()
            .ok_or("SC-20671 receipt has no process-start phase")?
            .pid;
        if receipt.receipt_sha256 != expected_sha
            || receipt.matrix.family != coordinate.family
            || receipt.matrix.context_band != coordinate.context_band
            || receipt.matrix.request_mode != coordinate.request_mode
            || receipt.matrix.prefill_mode != coordinate.prefill_mode
            || receipt.matrix.process_temperature != coordinate.process_temperature
            || row.get("workerPid").and_then(serde_json::Value::as_u64)
                != Some(u64::from(worker_pid))
        {
            return Err("SC-20671 manifest/receipt coordinate, PID, or seal mismatch".into());
        }
        if row.get("qualityGatePassed")
            != receipt
                .quality
                .quality_gate
                .as_ref()
                .map(|gate| serde_json::json!(gate.passed))
                .as_ref()
        {
            return Err(format!(
                "SC-20671 manifest row {slug} qualityGatePassed does not recompute from its receipt"
            ));
        }
        let listed = row
            .get("files")
            .and_then(serde_json::Value::as_array)
            .ok_or("SC-20671 campaign manifest row has no artifact files")?;
        let mut manifest_files = std::collections::BTreeMap::new();
        for file in listed {
            let name = file
                .get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or("campaign artifact name is missing")?;
            let bytes = file
                .get("sha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("campaign artifact byte seal is missing")?;
            let sidecar = file
                .get("sidecarSha256")
                .and_then(serde_json::Value::as_str)
                .ok_or("campaign artifact sidecar seal is missing")?;
            if manifest_files
                .insert(name.to_owned(), (bytes.to_owned(), sidecar.to_owned()))
                .is_some()
            {
                return Err("SC-20671 campaign manifest repeats an artifact file".into());
            }
        }
        if manifest_files != manifest_file_bindings(&item.bundle) {
            return Err("SC-20671 manifest artifact bytes or sidecars do not match bundle".into());
        }
        let discipline = if coordinate.process_temperature == "cold" {
            ProcessDiscipline::FreshChild
        } else {
            ProcessDiscipline::ReusedWarmWorker
        };
        outcomes.push((coordinate.clone(), discipline, worker_pid));
        let identity = campaign_global_identity(&receipt)?;
        if let Some(expected) = &global_identity {
            if expected != &identity {
                return Err("SC-20671 campaign global identity drifted".into());
            }
        } else {
            global_identity = Some(identity);
        }
        let family_identity = campaign_family_identity(&receipt)?;
        if let Some(expected) = family_identities.get(coordinate.family) {
            if expected != &family_identity {
                return Err("SC-20671 campaign family identity drifted".into());
            }
        } else {
            family_identities.insert(coordinate.family, family_identity);
        }
        host_state_receipts.push(receipt);
        prepared.push(item);
    }
    if seen.len() != schedule.len() || family_identities.len() != 2 {
        return Err("SC-20671 campaign omits a scheduled coordinate or family identity".into());
    }
    if resume_identity.is_some()
        && value
            .get("hostStateVaried")
            .and_then(serde_json::Value::as_bool)
            != Some(campaign_host_state_varied(&host_state_receipts))
    {
        return Err("SC-20671 manifest host-state variation flag does not recompute".into());
    }
    if value.get("qualityGatePassed")
        != campaign_quality_gate_passed(&host_state_receipts)?
            .map(serde_json::Value::Bool)
            .as_ref()
    {
        return Err("SC-20671 manifest qualityGatePassed does not recompute from its rows".into());
    }
    validate_schedule_outcomes(&schedule, &outcomes)?;
    Ok(prepared)
}

/// Load and validate a whole published SC-20671 campaign before selecting the one exact long-
/// context candidate row for `family`. This never accepts a standalone receipt: all immutable
/// schedule rows, manifest hashes, receipt seals, and coordinate/family matches must be present.
pub fn select_sc20676_baseline_row(
    campaign_directory: &Path,
    family: &str,
) -> Result<Sc20676BaselineRow, String> {
    if !["llama", "qwen"].contains(&family) {
        return Err("SC-20676 baseline family must be llama or qwen".into());
    }
    let mut selected = None;
    for item in load_validated_complete_campaign(campaign_directory)? {
        let coordinate = item.coordinate;
        let receipt: Receipt =
            serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
        if receipt.mode != "dense" {
            return Err("SC-20676 baseline requires a dense SC-20671 campaign".into());
        }
        if coordinate.family == family
            && coordinate.context_band == SC20676_BASELINE_CONTEXT_BAND
            && coordinate.request_mode == "single"
            && coordinate.prefill_mode == "single-shot"
            && coordinate.process_temperature == "warm"
            && selected
                .replace(Sc20676BaselineRow {
                    receipt,
                    coordinate,
                })
                .is_some()
        {
            return Err("SC-20671 campaign contains multiple SC-20676 baseline rows".into());
        }
    }
    selected.ok_or("SC-20671 complete campaign lacks the required long-context baseline row".into())
}

/// Reuse SC-20671's immutable candidate snapshot contract rather than accepting an arbitrary
/// Llama/Qwen directory for SC-20676 evidence.
pub fn validate_sc20676_candidate_snapshot(
    family: &str,
    snapshot: &Path,
) -> Result<SnapshotInventory, String> {
    let spec = benchmark_model(family, false)?;
    validate_benchmark_snapshot(snapshot, spec)?;
    inventory_snapshot(snapshot).map_err(|e| e.to_string())
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
    let _: Receipt = serde_json::from_slice(&receipt).map_err(|e| e.to_string())?;
    let mut fixtures = Vec::with_capacity(REQUIRED_FIXTURES.len() * QUALITY_MEASUREMENTS);
    for repeat in 0..QUALITY_MEASUREMENTS {
        for fixture in REQUIRED_FIXTURES {
            let name = fixture_artifact_name(fixture, repeat);
            let bytes = fs::read(directory.join(&name)).map_err(|e| e.to_string())?;
            let sidecar = fs::read_to_string(directory.join(format!("{name}.sha256")))
                .map_err(|e| e.to_string())?;
            fixtures.push(SealedFixtureArtifact {
                name,
                bytes,
                sidecar,
            });
        }
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

/// Process exit status of a campaign parent halted by its operator stop file between rows. It is
/// distinct from success and from every refusal/failure status: sysexits `EX_TEMPFAIL`, "try
/// again later" — rerunning the same command against the same resume directory (with the stop
/// file removed) resumes at the row that was not started.
pub const OPERATOR_STOP_EXIT_CODE: u8 = 75;
/// Default stop-file name inside a campaign resume directory (`--stop-file` overrides the path).
pub const OPERATOR_STOP_FILE_NAME: &str = "STOP";

/// A durable "stopped by operator before row N" status. Rows `0..before_row` are accepted (or
/// resumed) in the resume directory; row `before_row` and later were never started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OperatorStop {
    pub before_row: usize,
    pub row: String,
    pub rows_total: usize,
    /// The sealed status record written under the resume directory's `logs/`.
    pub record: PathBuf,
}

/// How a campaign parent invocation ended without an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CampaignOutcome {
    Completed,
    StoppedByOperator(OperatorStop),
    /// Every row was attempted but these were refused before spawn; nothing is published.
    IncompleteWithRefusals(Vec<String>),
}

impl CampaignOutcome {
    pub fn exit_code(&self) -> u8 {
        match self {
            Self::Completed => 0,
            Self::StoppedByOperator(_) => OPERATOR_STOP_EXIT_CODE,
            Self::IncompleteWithRefusals(_) => 1,
        }
    }
}

/// The operator stop files a parent honours: always `<resume-dir>/STOP`, plus `--stop-file <path>`
/// when given (either present stops the campaign). Stop files are operator control only: never
/// part of a resume identity and never forwarded to a worker.
pub(crate) fn operator_stop_files(
    args: &[String],
    resume_dir: &Path,
) -> Result<Vec<PathBuf>, String> {
    let mut files = vec![resume_dir.join(OPERATOR_STOP_FILE_NAME)];
    if args.iter().any(|arg| arg == "--stop-file") {
        let custom = PathBuf::from(required_flag(args, "--stop-file")?);
        if !files.contains(&custom) {
            files.push(custom);
        }
    }
    Ok(files)
}

/// Whether resume-directory entry `name` is an operator stop file rather than campaign state.
pub(crate) fn is_operator_stop_entry(
    resume_dir: &Path,
    stop_files: &[PathBuf],
    name: &str,
) -> bool {
    name == OPERATOR_STOP_FILE_NAME || stop_files.contains(&resume_dir.join(name))
}

/// A stop file is present when anything (even a dangling symlink) exists at its path. Only
/// `NotFound` means absent; any other stat failure is an error, never a silent "keep going".
fn operator_stop_present(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(format!(
            "stat operator stop file {}: {error}",
            path.display()
        )),
    }
}

/// Create `path` exclusively and write `bytes` durably (fsync).
fn write_new_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    std::io::Write::write_all(&mut file, bytes)?;
    file.sync_all()
}

/// Checked by a campaign parent only between rows, immediately before it would spawn row
/// `before_row`'s worker; a running worker is never signalled (killing an MLX render mid command
/// buffer can wedge the GPU). When any stop file exists this writes a sealed, never-overwritten
/// `logs/operator-stop.attempt-<n>.json` status record plus its `.sha256` sidecar and returns the
/// stop.
pub(crate) fn operator_stop_before_row(
    stop_files: &[PathBuf],
    logs: &Path,
    kind: &str,
    before_row: usize,
    row: &str,
    rows_total: usize,
) -> Result<Option<OperatorStop>, String> {
    let mut present = Vec::new();
    for path in stop_files {
        if operator_stop_present(path)? {
            present.push(path.display().to_string());
        }
    }
    if present.is_empty() {
        return Ok(None);
    }
    fs::create_dir_all(logs).map_err(|e| format!("create operator stop log directory: {e}"))?;
    let value = serde_json::json!({
        "schemaVersion": 1,
        "kind": kind,
        "status": "stopped-by-operator",
        "beforeRow": before_row,
        "beforeRowSlug": row,
        "rowsTotal": rows_total,
        "rowsAccepted": before_row,
        "stopFiles": present,
        "recordedAt": timestamp_now(),
        "resume": "remove the stop file and rerun the same command with the same resume directory",
    });
    let (bytes, sha256) = seal_json(&value)?;
    let mut attempt = 0_u64;
    let record = loop {
        let name = format!("operator-stop.attempt-{attempt}.json");
        let path = logs.join(&name);
        match write_new_durable(&path, &bytes) {
            Ok(()) => {
                write_new_durable(
                    &logs.join(format!("{name}.sha256")),
                    format!("{sha256}  {name}\n").as_bytes(),
                )
                .map_err(|e| format!("write operator stop record seal: {e}"))?;
                break path;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                attempt = attempt
                    .checked_add(1)
                    .ok_or("no unused operator stop record path")?;
            }
            Err(error) => return Err(format!("write operator stop record: {error}")),
        }
    };
    eprintln!(
        "stopped by operator before row {}/{rows_total} ({row}); stop file(s) {}; status {}",
        before_row + 1,
        present.join(", "),
        record.display(),
    );
    Ok(Some(OperatorStop {
        before_row,
        row: row.into(),
        rows_total,
        record,
    }))
}

/// One parent-loop step for a scheduled row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RowStep {
    /// Accept the row from the resume directory; `Ok(false)` means it must run.
    Resume,
    /// Spawn, supervise, and accept the row's worker; `Ok(false)` records a pre-spawn refusal
    /// and the loop continues with the next row.
    Run,
}

/// The campaign parent's sequential row loop: resumed rows are accepted without a worker, and the
/// operator stop file is consulted between rows, before each worker spawn.
pub(crate) fn drive_rows(
    slugs: &[String],
    stop_files: &[PathBuf],
    logs: &Path,
    kind: &str,
    mut row: impl FnMut(usize, RowStep) -> Result<bool, String>,
) -> Result<DriveOutcome, String> {
    let mut refused = Vec::new();
    for (index, slug) in slugs.iter().enumerate() {
        if row(index, RowStep::Resume)? {
            eprintln!("coordinate {}/{} resumed: {slug}", index + 1, slugs.len());
            continue;
        }
        if let Some(stop) =
            operator_stop_before_row(stop_files, logs, kind, index, slug, slugs.len())?
        {
            return Ok(DriveOutcome {
                stop: Some(stop),
                refused,
            });
        }
        if row(index, RowStep::Run)? {
            eprintln!("coordinate {}/{} accepted: {slug}", index + 1, slugs.len());
        } else {
            eprintln!(
                "coordinate {}/{} refused before spawn: {slug}",
                index + 1,
                slugs.len()
            );
            refused.push(slug.clone());
        }
    }
    Ok(DriveOutcome {
        stop: None,
        refused,
    })
}

/// How the parent's row loop ended: an operator stop, and the rows refused before spawn (whose
/// unaccepted records are in the logs).
#[derive(Debug)]
pub(crate) struct DriveOutcome {
    pub(crate) stop: Option<OperatorStop>,
    pub(crate) refused: Vec<String>,
}

/// Immutable parent inputs.  The only model-related choices are snapshot paths; model identity,
/// geometry, memory, timing, and quality fields are collected by the child from the loaded product
/// and are never accepted by this command-line boundary.
#[derive(Clone, Debug)]
pub struct CampaignLaunch {
    pub executable: PathBuf,
    pub llama_snapshot: PathBuf,
    pub qwen_snapshot: PathBuf,
    pub llama_fp32_reference_snapshot: PathBuf,
    pub qwen_fp32_reference_snapshot: PathBuf,
    pub prompt_file: PathBuf,
    pub destination: PathBuf,
    pub resume_dir: PathBuf,
    /// Operator stop files checked between rows; never part of the resume identity.
    pub stop_files: Vec<PathBuf>,
    pub safety_policy: PathBuf,
    /// `Some` launches every scheduled row in `compressed` mode with this KV method; `None` is the
    /// dense baseline campaign.
    pub compressed: Option<CompressedKvMethod>,
    /// `--only-coordinate <slug>`: run just this scheduled row. The run publishes a partial,
    /// non-publishable manifest ([`SC20671_PARTIAL_RUN_KIND`]), never a campaign.
    pub only_coordinate: Option<String>,
}

/// Indices into [`required_schedule`] a launch runs: all of them, or the one `--only-coordinate`
/// names (an unscheduled name is refused, listing the schedule).
fn selected_schedule_indices(only_coordinate: Option<&str>) -> Result<Vec<usize>, String> {
    let slugs = required_coordinates()
        .iter()
        .map(coordinate_slug)
        .collect::<Vec<_>>();
    match only_coordinate {
        None => Ok((0..slugs.len()).collect()),
        Some(slug) => slugs
            .iter()
            .position(|candidate| candidate == slug)
            .map(|index| vec![index])
            .ok_or_else(|| {
                format!("--only-coordinate {slug:?} is not a scheduled coordinate; expected one of {slugs:?}")
            }),
    }
}

pub(crate) fn file_seal(path: &Path) -> Result<String, String> {
    let mut input = File::open(path).map_err(|e| e.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(&digest.finalize()))
}

fn resume_identity(
    launch: &CampaignLaunch,
    policy_sha256: &str,
    inventories: &[SnapshotInventory; 4],
) -> Result<serde_json::Value, String> {
    let inference_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let scene_works_root = PathBuf::from(required_campaign_env("SCENEWORKS_ROOT")?);
    checked_repository_identity(inference_root, INFERENCE_REPOSITORY)?;
    checked_repository_identity(&scene_works_root, SCENEWORKS_REPOSITORY)?;
    let inventory_value = |index: usize| {
        serde_json::json!({
            "sha256": inventories[index].sha256,
            "bytes": inventories[index].bytes,
        })
    };
    let mut identity = serde_json::json!({
        "schemaVersion": 1,
        "kind": "sc-20671-resume-identity",
        "scheduleVersion": SC20671_SCHEDULE_VERSION,
        "coordinates": required_coordinates().iter().map(coordinate_slug).collect::<Vec<_>>(),
        "inferenceRevision": checked_git_revision(inference_root)?,
        "sceneWorksRevision": checked_git_revision(&scene_works_root)?,
        "executableSha256": file_seal(&launch.executable)?,
        "promptSha256": file_seal(&launch.prompt_file)?,
        "policySha256": policy_sha256,
        "llamaCandidate": inventory_value(0),
        "qwenCandidate": inventory_value(1),
        "llamaReference": inventory_value(2),
        "qwenReference": inventory_value(3),
    });
    bind_resume_mode(&mut identity, launch.compressed);
    if let Some(slug) = &launch.only_coordinate {
        // A single-row run never shares a resume directory with a full campaign.
        identity["onlyCoordinate"] = slug.as_str().into();
    }
    Ok(identity)
}

/// A compressed campaign's resume identity names its mode and method, so dense and compressed
/// rows (or two methods) can never resume into each other. Dense identities keep their exact
/// historical bytes.
fn bind_resume_mode(identity: &mut serde_json::Value, compressed: Option<CompressedKvMethod>) {
    if let Some(method) = compressed {
        identity["mode"] = "compressed".into();
        identity["kvMethod"] = method.id().into();
    }
}

/// The row mode a sealed resume identity launched: `None` for dense.
fn resume_identity_mode(
    identity: &serde_json::Value,
) -> Result<Option<CompressedKvMethod>, String> {
    match (
        identity.get("mode").map(serde_json::Value::as_str),
        identity.get("kvMethod").map(serde_json::Value::as_str),
    ) {
        (None, None) => Ok(None),
        (Some(Some("compressed")), Some(Some(method))) => {
            CompressedKvMethod::parse(method).map(Some)
        }
        _ => Err("resume identity has a malformed compressed mode binding".into()),
    }
}

fn seal_json(value: &serde_json::Value) -> Result<(Vec<u8>, String), String> {
    let bytes = canonical_json_bytes(value).map_err(|e| e.to_string())?;
    let sha = seal_bytes(&bytes);
    Ok((bytes, sha))
}

fn preserve_captured_source(root: &Path, identity: &mut serde_json::Value) -> Result<(), String> {
    if root.exists() {
        let prior: serde_json::Value = serde_json::from_slice(
            &fs::read(root.join("identity.json")).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        // A moved source ref cannot stale identical executable/model/contract bytes. Preserve the
        // captured refs as provenance; all new rows must stamp that same executable provenance.
        for key in ["inferenceRevision", "sceneWorksRevision"] {
            let captured = prior
                .get(key)
                .and_then(serde_json::Value::as_str)
                .ok_or("resume identity lacks captured source revision")?;
            identity[key] = serde_json::Value::String(captured.to_owned());
        }
    }
    Ok(())
}

fn prepare_resume_root(root: &Path, identity: &mut serde_json::Value) -> Result<String, String> {
    preserve_captured_source(root, identity)?;
    let (bytes, sha) = seal_json(identity)?;
    let sidecar = format!("{sha}  identity.json\n");
    if root.exists() {
        if !root.is_dir()
            || fs::read(root.join("identity.json")).map_err(|e| e.to_string())? != bytes
            || fs::read_to_string(root.join("identity.json.sha256")).map_err(|e| e.to_string())?
                != sidecar
        {
            return Err("SC-20671 resume identity differs from current source, models, executable, schedule, prompt, or policy".into());
        }
    } else {
        fs::create_dir(root).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json"), &bytes).map_err(|e| e.to_string())?;
        fs::write(root.join("identity.json.sha256"), sidecar).map_err(|e| e.to_string())?;
    }
    Ok(sha)
}

/// Read-only preflight used to hand the exact producer-calculated identity and policy seal to
/// an independent consumer before any model child is started.
pub fn preflight_complete_campaign(launch: &CampaignLaunch) -> Result<serde_json::Value, String> {
    let policy = load_campaign_safety_policy(&launch.safety_policy)?;
    let inventories = [
        validate_benchmark_snapshot(&launch.llama_snapshot, &LLAMA_CANDIDATE)?,
        validate_benchmark_snapshot(&launch.qwen_snapshot, &QWEN_CANDIDATE)?,
        validate_benchmark_snapshot(&launch.llama_fp32_reference_snapshot, &LLAMA_REFERENCE)?,
        validate_benchmark_snapshot(&launch.qwen_fp32_reference_snapshot, &QWEN_REFERENCE)?,
    ];
    let prompt = fs::read_to_string(&launch.prompt_file).map_err(|e| e.to_string())?;
    if prompt.trim().is_empty() {
        return Err("campaign prompt must not be empty".into());
    }
    let mut row_admission = Vec::new();
    let schedule = required_schedule();
    for row in selected_schedule_indices(launch.only_coordinate.as_deref())?
        .into_iter()
        .map(|index| schedule[index].clone())
    {
        let (candidate, reference) = if row.coordinate.family == "llama" {
            (
                &launch.llama_snapshot,
                &launch.llama_fp32_reference_snapshot,
            )
        } else {
            (&launch.qwen_snapshot, &launch.qwen_fp32_reference_snapshot)
        };
        let coordinate = coordinate_slug(&row.coordinate);
        match static_row_requirements(
            &row.coordinate,
            candidate,
            reference,
            &prompt,
            &policy,
            launch.compressed.is_some(),
        )
        .and_then(|(total, request, footprint)| {
            Ok((
                total,
                request,
                runtime_guarded_admission(&policy, footprint, SC20671_ESTIMATE_SOURCE)?,
            ))
        }) {
            Ok((total, request, admission)) => row_admission.push(serde_json::json!({
                "coordinate": coordinate, "staticPreflightPassed": true,
                "totalLiveTokenBound": total, "requestTokenBound": request,
                "knownFootprintBudgetBytes": admission.static_footprint_floor_bytes,
                "admission": admission,
            })),
            Err(reason) => row_admission.push(serde_json::json!({
                "coordinate": coordinate, "staticPreflightPassed": false, "reason": reason,
            })),
        }
    }
    let policy_sha256 = policy.seal()?;
    let mut identity = resume_identity(launch, &policy_sha256, &inventories)?;
    preserve_captured_source(&launch.resume_dir, &mut identity)?;
    let (identity_bytes, resume_identity_sha256) = seal_json(&identity)?;
    if launch.resume_dir.exists()
        && (fs::read(launch.resume_dir.join("identity.json")).map_err(|e| e.to_string())?
            != identity_bytes
            || fs::read_to_string(launch.resume_dir.join("identity.json.sha256"))
                .map_err(|e| e.to_string())?
                != format!("{resume_identity_sha256}  identity.json\n"))
    {
        return Err(
            "SC-20671 existing resume identity does not match current behavior inputs".into(),
        );
    }
    Ok(serde_json::json!({
        "schemaVersion": 1,
        "policySha256": policy_sha256,
        "resumeIdentitySha256": resume_identity_sha256,
        "identity": identity,
        "completeStaticPreflightPassed": row_admission.iter().all(|row|
            row.get("staticPreflightPassed").and_then(serde_json::Value::as_bool) == Some(true)),
        "runtimeAdmission": "not-evaluated-by-read-only-preflight",
        "rows": row_admission,
    }))
}

fn row_binding_path(root: &Path, slug: &str) -> PathBuf {
    root.join(format!("{slug}.binding.json"))
}

fn row_binding(
    identity_sha256: &str,
    receipt: &Receipt,
    receipt_bytes: &[u8],
) -> serde_json::Value {
    serde_json::json!({
        "schemaVersion": 1,
        "resumeIdentitySha256": identity_sha256,
        "receiptSha256": receipt.receipt_sha256,
        "receiptFileSha256": seal_bytes(receipt_bytes),
    })
}

fn validate_receipt_launch_identity(
    receipt: &Receipt,
    coordinate: &Coordinate,
    identity: &serde_json::Value,
) -> Result<(), String> {
    let slug = coordinate_slug(coordinate);
    let p = &receipt.provenance;
    let field = |key| identity.get(key).and_then(serde_json::Value::as_str);
    let inventory = |key: &str| {
        identity
            .get(key)
            .and_then(|v| v.get("sha256"))
            .and_then(serde_json::Value::as_str)
    };
    let inventory_bytes = |key: &str| {
        identity
            .get(key)
            .and_then(|v| v.get("bytes"))
            .and_then(serde_json::Value::as_u64)
    };
    let (candidate, reference) = if coordinate.family == "llama" {
        ("llamaCandidate", "llamaReference")
    } else {
        ("qwenCandidate", "qwenReference")
    };
    let method = resume_identity_mode(identity)?;
    if receipt.mode
        != if method.is_some() {
            "compressed"
        } else {
            "dense"
        }
        || receipt.compression.as_ref().map(|c| c.method.as_str()) != method.map(|m| m.id())
    {
        return Err(format!(
            "resume row {slug} has a stale dense/compressed mode"
        ));
    }
    if p.inference_revision != field("inferenceRevision").unwrap_or("")
        || p.scene_works_revision != field("sceneWorksRevision").unwrap_or("")
        || p.model_file_sha256 != inventory(candidate).unwrap_or("")
        || Some(p.model_file_bytes) != inventory_bytes(candidate)
        || p.reference_model_sha256 != inventory(reference).unwrap_or("")
        || Some(p.reference_model_bytes) != inventory_bytes(reference)
        || receipt.matrix.family != coordinate.family
        || receipt.matrix.context_band != coordinate.context_band
        || receipt.matrix.request_mode != coordinate.request_mode
        || receipt.matrix.prefill_mode != coordinate.prefill_mode
        || receipt.matrix.process_temperature != coordinate.process_temperature
    {
        return Err(format!(
            "resume row {slug} has stale source, model, or coordinate identity"
        ));
    }
    Ok(())
}

fn validate_resume_row(
    root: &Path,
    coordinate: &Coordinate,
    identity: &serde_json::Value,
    identity_sha256: &str,
) -> Result<PreparedCoordinateReceipt, String> {
    let slug = coordinate_slug(coordinate);
    let item = load_prepared_coordinate_receipt(&root.join(&slug), coordinate.clone())?;
    let receipt: Receipt =
        serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
    validate_sealed_receipt(&receipt)?;
    validate_receipt_launch_identity(&receipt, coordinate, identity)?;
    let expected = seal_json(&row_binding(
        identity_sha256,
        &receipt,
        &item.bundle.receipt,
    ))?
    .0;
    if fs::read(row_binding_path(root, &slug)).map_err(|e| e.to_string())? != expected {
        return Err(format!(
            "resume row {slug} has missing or stale identity binding"
        ));
    }
    Ok(item)
}

/// Run one row at a time under a mandatory bounded policy. Valid sealed rows stay in an
/// identity-bound resume directory after a failure; the destination appears only after all eight
/// scheduled rows have been accepted. Between rows — never during a worker — the parent honours
/// the operator stop file: it records a durable stopped-before-row status and returns
/// [`CampaignOutcome::StoppedByOperator`]; the same command later resumes at that row.
pub fn launch_complete_campaign(launch: &CampaignLaunch) -> Result<CampaignOutcome, String> {
    if launch.destination.exists() {
        return Err("campaign destination already exists".into());
    }
    if !launch.resume_dir.is_absolute() || !launch.destination.is_absolute() {
        return Err("campaign destination and resume directory must be absolute paths".into());
    }
    if launch.resume_dir == launch.destination {
        return Err("campaign resume directory must differ from destination".into());
    }
    for path in [
        &launch.executable,
        &launch.llama_snapshot,
        &launch.qwen_snapshot,
        &launch.llama_fp32_reference_snapshot,
        &launch.qwen_fp32_reference_snapshot,
        &launch.prompt_file,
        &launch.safety_policy,
    ] {
        if !path.exists() {
            return Err(format!(
                "required campaign input is absent: {}",
                path.display()
            ));
        }
    }
    let policy = load_campaign_safety_policy(&launch.safety_policy)?;
    let policy_sha256 = policy.seal()?;
    let inventories = [
        validate_benchmark_snapshot(&launch.llama_snapshot, &LLAMA_CANDIDATE)?,
        validate_benchmark_snapshot(&launch.qwen_snapshot, &QWEN_CANDIDATE)?,
        validate_benchmark_snapshot(&launch.llama_fp32_reference_snapshot, &LLAMA_REFERENCE)?,
        validate_benchmark_snapshot(&launch.qwen_fp32_reference_snapshot, &QWEN_REFERENCE)?,
    ];
    let schedule = required_schedule();
    let selected = selected_schedule_indices(launch.only_coordinate.as_deref())?;
    let prompt = fs::read_to_string(&launch.prompt_file).map_err(|e| e.to_string())?;
    if prompt.trim().is_empty() {
        return Err("campaign prompt must not be empty".into());
    }
    let mut identity = resume_identity(launch, &policy_sha256, &inventories)?;
    let identity_sha256 = prepare_resume_root(&launch.resume_dir, &mut identity)?;
    let slugs = selected
        .iter()
        .map(|index| coordinate_slug(&schedule[*index].coordinate))
        .collect::<Vec<_>>();
    validate_resume_entries(&launch.resume_dir, &launch.stop_files, &slugs)?;
    let mut prepared = Vec::with_capacity(schedule.len());
    let stopped = drive_rows(
        &slugs,
        &launch.stop_files,
        &launch.resume_dir.join("logs"),
        "sc-20671-operator-stop",
        |index, step| {
            let schedule_index = selected[index];
            let row = &schedule[schedule_index];
            let slug = coordinate_slug(&row.coordinate);
            let child_dir = launch.resume_dir.join(&slug);
            let binding_path = row_binding_path(&launch.resume_dir, &slug);
            if step == RowStep::Resume {
                if !child_dir.exists() && !binding_path.exists() {
                    return Ok(false);
                }
                if !child_dir.exists() || !binding_path.exists() {
                    return Err(format!(
                        "partial resume row {slug} must be repaired explicitly"
                    ));
                }
                prepared.push(validate_resume_row(
                    &launch.resume_dir,
                    &row.coordinate,
                    &identity,
                    &identity_sha256,
                )?);
                return Ok(true);
            }
            let snapshot = if row.coordinate.family == "llama" {
                &launch.llama_snapshot
            } else {
                &launch.qwen_snapshot
            };
            let reference_snapshot = if row.coordinate.family == "llama" {
                &launch.llama_fp32_reference_snapshot
            } else {
                &launch.qwen_fp32_reference_snapshot
            };
            let (total_tokens, request_tokens, static_footprint_floor) = static_row_requirements(
                &row.coordinate,
                snapshot,
                reference_snapshot,
                &prompt,
                &policy,
                launch.compressed.is_some(),
            )?;
            let admission = runtime_guarded_admission(
                &policy,
                static_footprint_floor,
                SC20671_ESTIMATE_SOURCE,
            )?;
            let mut command = Command::new(&launch.executable);
            command
                .arg("worker")
                .arg("--coordinate-index")
                .arg(schedule_index.to_string())
                .arg("--snapshot")
                .arg(snapshot)
                .arg("--prompt-file")
                .arg(&launch.prompt_file)
                .arg("--fp32-reference-snapshot")
                .arg(reference_snapshot)
                .arg("--safety-policy")
                .arg(&launch.safety_policy)
                .arg("--policy-sha256")
                .arg(&policy_sha256)
                .arg("--resume-identity")
                .arg(launch.resume_dir.join("identity.json"))
                .arg("--resume-identity-sha256")
                .arg(&identity_sha256)
                .arg("--out")
                .arg(&child_dir);
            if let Some(method) = launch.compressed {
                command
                    .arg("--mode")
                    .arg("compressed")
                    .arg("--kv-method")
                    .arg(method.id());
            }
            let logs = launch.resume_dir.join("logs");
            let log_prefix =
                unused_attempt_prefix(&logs, &slug).ok_or("no unused bounded worker log path")?;
            let request = RunRequest {
                context_tokens: total_tokens,
                request_tokens,
                estimate: admission.estimate(),
                stdout_path: logs.join(format!("{log_prefix}.stdout.log")),
                stderr_path: logs.join(format!("{log_prefix}.stderr.log")),
            };
            let unaccepted = logs.join(format!("{log_prefix}.unaccepted.json"));
            // Pre-row duration estimate from completed same-family rows. A row whose estimate
            // exceeds the deadline by the margin is refused before it spawns; the campaign then
            // continues with the next row and ends incomplete-with-refusals.
            let basis = prepared
                .iter()
                .filter_map(|item| serde_json::from_slice::<Receipt>(&item.bundle.receipt).ok())
                .filter_map(|receipt| row_duration_basis(&receipt))
                .collect::<Vec<_>>();
            let estimate = estimate_row_seconds(
                row.coordinate.family,
                row.coordinate.context_band,
                row_work(
                    row.coordinate.process_temperature,
                    row.coordinate.request_mode,
                    row.coordinate.prefill_mode,
                ),
                &basis,
            );
            let deadline_seconds = policy.row_deadline_seconds;
            let estimate_record = serde_json::json!({
                "schemaVersion": 2,
                "kind": "sc-20671-row-duration-estimate",
                "coordinate": slug,
                "estimatedSeconds": estimate.as_ref().map(|e| e.seconds),
                "fixedSeconds": estimate.as_ref().map(|e| e.fixed_seconds),
                "prefillSeconds": estimate.as_ref().map(|e| e.prefill_seconds),
                "basisCoordinate": estimate.as_ref().map(|e| e.basis_coordinate.clone()),
                "rowDeadlineSeconds": deadline_seconds,
                "refusalMargin": DURATION_REFUSAL_MARGIN,
                "method": "basis wall time split into measured full-context prefills and a fixed remainder; fixed scaled by measurement units (timing runs + quality halves), prefill by count x T(1+T/Tattn) ratio",
            });
            fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
            fs::write(
                logs.join(format!("{log_prefix}.estimate.json")),
                seal_json(&estimate_record)?.0,
            )
            .map_err(|e| e.to_string())?;
            match &estimate {
                Some(estimate) => eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} estimated at {:.0}s (fixed {:.0}s + prefill {:.0}s, from {}); row deadline {deadline_seconds}s",
                    estimate.seconds, estimate.fixed_seconds, estimate.prefill_seconds, estimate.basis_coordinate
                ),
                None => eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} has no completed full-prefill {} row to estimate from; row deadline {deadline_seconds}s",
                    row.coordinate.family
                ),
            }
            if let Some(estimate) = estimate
                .as_ref()
                .filter(|estimate| duration_estimate_refuses(estimate, deadline_seconds))
            {
                let detail = format!(
                    "estimated duration {:.0}s (from {}) exceeds the {deadline_seconds}s row deadline by more than {DURATION_REFUSAL_MARGIN}x",
                    estimate.seconds, estimate.basis_coordinate
                );
                let message = unaccepted_row_error(
                    &unaccepted,
                    "sc-20671-unaccepted-row",
                    &slug,
                    &admission,
                    ("DurationEstimate", &detail, None, None),
                    format!("coordinate {slug} refused before spawn: {detail}"),
                );
                eprintln!("sc20671-kv-baseline: {message}; continuing with the next row");
                return Ok(false);
            }
            let status = match campaign_supervisor::run_guarded(
                &mut command,
                &request,
                &policy.supervisor(),
                &mut SystemProbe,
            ) {
                Ok(status) => status,
                Err(failure) => {
                    let reason = format!("{:?}", failure.reason);
                    return Err(unaccepted_row_error(
                        &unaccepted,
                        "sc-20671-unaccepted-row",
                        &slug,
                        &admission.with_host_memory(failure.host_memory.as_deref().cloned()),
                        (
                &reason,
                &failure.detail,
                failure.pid,
                failure.watchdog_host_memory.as_deref(),
            ),
                        format!(
                            "coordinate {slug} stopped ({reason}): {}; child {:?} reaped; valid earlier rows remain in {}",
                            failure.detail, failure.pid, launch.resume_dir.display(),
                        ),
                    ));
                }
            };
            if !status.success() {
                return Err(unaccepted_row_error(
                    &unaccepted,
                    "sc-20671-unaccepted-row",
                    &slug,
                    &admission,
                    ("ChildExit", &status.to_string(), None, None),
                    format!(
                        "coordinate {slug} failed with {status}; stderr: {}; valid earlier rows remain in {}",
                        request.stderr_path.display(), launch.resume_dir.display(),
                    ),
                ));
            }
            // Child publication itself is atomic. Bind the row to this exact immutable launch before
            // permitting a future resume to accept it.
            let item = load_prepared_coordinate_receipt(&child_dir, row.coordinate.clone())?;
            let receipt: Receipt =
                serde_json::from_slice(&item.bundle.receipt).map_err(|e| e.to_string())?;
            if let Some(gate) = &receipt.quality.quality_gate {
                eprintln!(
                    "sc20671-kv-baseline: coordinate {slug} accepted as measured; quality gate {}",
                    quality_gate_summary(gate)
                );
            }
            let binding = seal_json(&row_binding(
                &identity_sha256,
                &receipt,
                &item.bundle.receipt,
            ))?
            .0;
            fs::write(&binding_path, binding).map_err(|e| e.to_string())?;
            prepared.push(validate_resume_row(
                &launch.resume_dir,
                &row.coordinate,
                &identity,
                &identity_sha256,
            )?);
            Ok(true)
        },
    )?;
    if let Some(stop) = stopped.stop {
        return Ok(CampaignOutcome::StoppedByOperator(stop));
    }
    if !stopped.refused.is_empty() {
        // Every other row ran; the campaign is not complete and is never published.
        eprintln!(
            "sc20671-kv-baseline: campaign incomplete: {} row(s) refused before spawn ({}); accepted rows remain in {}",
            stopped.refused.len(),
            stopped.refused.join(", "),
            launch.resume_dir.display()
        );
        return Ok(CampaignOutcome::IncompleteWithRefusals(stopped.refused));
    }
    match &launch.only_coordinate {
        Some(slug) => {
            publish_campaign(
                &launch.destination,
                &prepared,
                &identity,
                &policy,
                Some(slug),
            )?;
            eprintln!(
                "sc20671-kv-baseline: PARTIAL run of {slug} only, published as a non-publishable {SC20671_PARTIAL_RUN_KIND} at {}; it is not a campaign",
                launch.destination.display()
            );
        }
        None => publish_complete_campaign(&launch.destination, &prepared, &identity, &policy)?,
    }
    report_campaign_quality_gates(
        &prepared
            .iter()
            .map(|item| serde_json::from_slice::<Receipt>(&item.bundle.receipt))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?,
    )?;
    Ok(CampaignOutcome::Completed)
}

/// Every resume-directory entry must be campaign state for this schedule or the operator stop file;
/// the stop file is tolerated here and never enters the resume identity.
fn validate_resume_entries(
    resume_dir: &Path,
    stop_files: &[PathBuf],
    slugs: &[String],
) -> Result<(), String> {
    for entry in fs::read_dir(resume_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        if !["identity.json", "identity.json.sha256", "logs"].contains(&name.as_str())
            && !is_operator_stop_entry(resume_dir, stop_files, &name)
            && !slugs.contains(&name)
            && !slugs
                .iter()
                .any(|slug| name == format!("{slug}.binding.json"))
        {
            return Err(format!("unexpected or partial resume artifact: {name}"));
        }
    }
    Ok(())
}

/// `name <value>` when present (a flag without a value is refused).
fn optional_flag(args: &[String], name: &str) -> Result<Option<String>, String> {
    match args.iter().position(|arg| arg == name) {
        None => Ok(None),
        Some(index) => args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .cloned()
            .map(Some)
            .ok_or_else(|| format!("{name} requires a value")),
    }
}

fn required_flag(args: &[String], name: &str) -> Result<String, String> {
    args.windows(2)
        .find_map(|window| (window[0] == name).then(|| window[1].clone()))
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("missing {name}"))
}

/// `--mode dense|compressed` (default dense) and, for compressed, the required `--kv-method`.
fn compressed_mode_flags(args: &[String]) -> Result<Option<CompressedKvMethod>, String> {
    let flag = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .map(|_| required_flag(args, name))
            .transpose()
    };
    match (flag("--mode")?.as_deref(), flag("--kv-method")?) {
        (None | Some("dense"), None) => Ok(None),
        (Some("compressed"), Some(method)) => CompressedKvMethod::parse(&method).map(Some),
        (Some("compressed"), None) => Err("--mode compressed requires --kv-method".into()),
        (None | Some("dense"), Some(_)) => Err("--kv-method requires --mode compressed".into()),
        (Some(mode), _) => Err(format!("--mode must be dense or compressed, not {mode:?}")),
    }
}

/// CLI entrypoint used by the standalone `sc20671-kv-baseline` binary.  The child accepts no
/// caller-authored evidence: identity, geometry, timing, quality, cache, and fixtures are bound
/// from the loaded product session before receipt publication.
///
/// `--mode compressed --kv-method <method>` (parent, preflight, and worker) runs every scheduled
/// row with its KV held in that method's compressed cache and fused decode attention, gated
/// against a dense-KV reference arm on the same candidate weights (SC-20676). Methods:
/// `group-affine` (2-bit codes), `group-affine-4` (4-bit) and `group-affine-8` (8-bit), all group
/// 32. `--only-coordinate <coordinate>` runs one scheduled row and publishes a partial,
/// non-publishable `sc-20671-partial-coordinate-run` manifest (never a campaign). The GPU-window
/// launch of the compressed campaign is one command from the inference checkout:
///
/// ```text
/// eval "$(scripts/fetch-prebuilt-mlx.sh --build-type Release)" && export PMETAL_MLX_PREBUILT_DIR PMETAL_METALLIB_PATH && \
/// SCENEWORKS_ROOT=/abs/SceneWorks cargo run --locked --release -p mlx-llm --bin sc20671_kv_baseline -- \
///   parent --mode compressed --kv-method group-affine \
///   --llama-snapshot <llama-4bit> --qwen-snapshot <qwen-4bit> \
///   --llama-fp32-reference-snapshot <llama-bf16> --qwen-fp32-reference-snapshot <qwen-bf16> \
///   --prompt-file <prompt.txt> --safety-policy <policy.json> \
///   --resume-dir /abs/sc20676-compressed-resume --out /abs/sc20676-compressed-campaign
/// ```
///
/// Safe operator stop: `touch /abs/sc20676-compressed-resume/STOP` (always honoured) or the
/// `--stop-file <path>` given to `parent` (honoured as well). The parent finishes the row whose
/// worker is running — it never signals a worker — then, before spawning the next row, writes
/// `<resume-dir>/logs/operator-stop.attempt-<n>.json` ("stopped-by-operator", `beforeRow`) with its
/// `.sha256` seal and exits with status [`OPERATOR_STOP_EXIT_CODE`] (75). Remove the stop file(s)
/// and rerun the same command: accepted rows resume and the campaign continues at that row.
pub fn sc20671_cli(args: &[String]) -> Result<CampaignOutcome, String> {
    let Some(mode) = args.first().map(String::as_str) else {
        return Err("usage: sc20671-kv-baseline parent|worker [options]".into());
    };
    match mode {
        "parent" | "preflight" => {
            let resume_dir = PathBuf::from(required_flag(args, "--resume-dir")?);
            let launch = CampaignLaunch {
                executable: std::env::current_exe().map_err(|e| e.to_string())?,
                llama_snapshot: PathBuf::from(required_flag(args, "--llama-snapshot")?),
                qwen_snapshot: PathBuf::from(required_flag(args, "--qwen-snapshot")?),
                llama_fp32_reference_snapshot: PathBuf::from(required_flag(
                    args,
                    "--llama-fp32-reference-snapshot",
                )?),
                qwen_fp32_reference_snapshot: PathBuf::from(required_flag(
                    args,
                    "--qwen-fp32-reference-snapshot",
                )?),
                prompt_file: PathBuf::from(required_flag(args, "--prompt-file")?),
                destination: PathBuf::from(required_flag(args, "--out")?),
                stop_files: operator_stop_files(args, &resume_dir)?,
                resume_dir,
                safety_policy: PathBuf::from(required_flag(args, "--safety-policy")?),
                compressed: compressed_mode_flags(args)?,
                only_coordinate: optional_flag(args, "--only-coordinate")?,
            };
            if mode == "preflight" {
                let value = preflight_complete_campaign(&launch)?;
                println!(
                    "{}",
                    String::from_utf8(canonical_json_bytes(&value).map_err(|e| e.to_string())?)
                        .map_err(|e| e.to_string())?
                );
                Ok(CampaignOutcome::Completed)
            } else {
                launch_complete_campaign(&launch)
            }
        }
        "worker" => {
            let policy = load_campaign_safety_policy(&PathBuf::from(required_flag(
                args,
                "--safety-policy",
            )?))?;
            if policy.seal()? != required_flag(args, "--policy-sha256")? {
                return Err("worker safety policy seal differs from parent's frozen policy".into());
            }
            let identity_bytes = fs::read(required_flag(args, "--resume-identity")?)
                .map_err(|e| format!("read parent resume identity: {e}"))?;
            let identity_sha256 = seal_bytes(&identity_bytes);
            if identity_sha256 != required_flag(args, "--resume-identity-sha256")? {
                return Err(
                    "worker resume identity seal differs from parent's frozen identity".into(),
                );
            }
            let identity: serde_json::Value =
                serde_json::from_slice(&identity_bytes).map_err(|e| e.to_string())?;
            let identity_field = |key| {
                identity
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| format!("resume identity lacks {key}"))
            };
            let compressed = compressed_mode_flags(args)?;
            if resume_identity_mode(&identity)? != compressed {
                return Err("worker dense/compressed mode differs from resume identity".into());
            }
            if identity_field("policySha256")? != policy.seal()?
                || identity_field("executableSha256")?
                    != file_seal(&std::env::current_exe().map_err(|e| e.to_string())?)?
                || identity
                    .get("scheduleVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(SC20671_SCHEDULE_VERSION)
            {
                return Err(
                    "worker executable, policy, or schedule differs from resume identity".into(),
                );
            }
            let captured_source = (
                identity_field("sceneWorksRevision")?,
                identity_field("inferenceRevision")?,
            );
            let index = required_flag(args, "--coordinate-index")?
                .parse::<usize>()
                .map_err(|_| "--coordinate-index must be an integer".to_string())?;
            let row = required_schedule()
                .get(index)
                .cloned()
                .ok_or("--coordinate-index is outside the frozen eight-coordinate schedule")?;
            if let Some(only) = identity.get("onlyCoordinate") {
                if only.as_str() != Some(coordinate_slug(&row.coordinate).as_str()) {
                    return Err(
                        "worker coordinate differs from the resume identity's coordinate filter"
                            .into(),
                    );
                }
            }
            let snapshot = PathBuf::from(required_flag(args, "--snapshot")?);
            let reference_snapshot =
                PathBuf::from(required_flag(args, "--fp32-reference-snapshot")?);
            let prompt_path = PathBuf::from(required_flag(args, "--prompt-file")?);
            if file_seal(&prompt_path)? != identity_field("promptSha256")? {
                return Err(
                    "worker prompt bytes differ from the parent-sealed resume identity".into(),
                );
            }
            let prompt =
                fs::read_to_string(&prompt_path).map_err(|e| format!("read prompt file: {e}"))?;
            if prompt.trim().is_empty() {
                return Err("campaign prompt must not be empty".into());
            }
            let (candidate_inventory, reference_inventory) = validate_coordinate_model_contract(
                &row.coordinate,
                &snapshot,
                &reference_snapshot,
            )?;
            let (candidate_key, reference_key) = if row.coordinate.family == "llama" {
                ("llamaCandidate", "llamaReference")
            } else {
                ("qwenCandidate", "qwenReference")
            };
            for (key, inventory) in [
                (candidate_key, &candidate_inventory),
                (reference_key, &reference_inventory),
            ] {
                if identity
                    .get(key)
                    .and_then(|model| model.get("sha256"))
                    .and_then(serde_json::Value::as_str)
                    != Some(inventory.sha256.as_str())
                    || identity
                        .get(key)
                        .and_then(|model| model.get("bytes"))
                        .and_then(serde_json::Value::as_u64)
                        != Some(inventory.bytes)
                {
                    return Err(format!(
                        "worker {key} inventory differs from parent-sealed resume identity"
                    ));
                }
            }
            let (_, _, static_footprint_floor) = static_row_requirements(
                &row.coordinate,
                &snapshot,
                &reference_snapshot,
                &prompt,
                &policy,
                compressed.is_some(),
            )?;
            let admission =
                supervised_admission(&policy, static_footprint_floor, SC20671_ESTIMATE_SOURCE)?;
            // Observed before any model loads: a throttled host refuses the row before it runs.
            let row_start = capture_host_state("row-start")?;
            refuse_throttled_row_start(&row_start)?;
            // Compressed rows: the measured arm holds KV in the method's compressed cache, and its
            // quality denominator is the dense-KV run on the SAME candidate weights.
            let candidate_session = match compressed {
                Some(method) => CampaignSession::load_compressed(&snapshot, method),
                None => CampaignSession::load(&snapshot),
            }
            .map_err(|e| format!("load candidate campaign session: {e}"))?;
            let candidate_baseline = candidate_session.load_start_sample.mlx_active_bytes;
            // Candidate lifecycle evidence is collected before the reference model exists in the
            // process. This keeps every global MLX phase sample attributable to the candidate.
            // Contract v5: the timing measurements (a warm row's warmups, then the five repeats)
            // run first, so a cold row's repeat 0 is the process's first product dispatch; quality
            // is then measured once per arm, because every fixture is deterministic in-process.
            let timing_run = |kind: &str, index: usize| {
                run_timing_on_session(
                    &candidate_session,
                    &prompt,
                    &row.coordinate,
                    policy.max_request_tokens,
                )
                .map_err(|e| {
                    format!(
                        "timing {kind} {index} for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })
            };
            let mut timing_warmups = Vec::new();
            if row.coordinate.process_temperature == "warm" {
                for warmup_index in 0..TIMING_WARMUPS {
                    timing_warmups.push(timing_run("warmup", warmup_index)?);
                }
            }
            let warmup_cache_state_version =
                (!timing_warmups.is_empty()).then(|| candidate_session.cache_state_version());
            let mut timing_repeats = Vec::with_capacity(TIMING_REPEATS);
            let mut repeat_host_states = Vec::with_capacity(TIMING_REPEATS);
            for repeat in 0..TIMING_REPEATS {
                timing_repeats.push(timing_run("repeat", repeat)?);
                // Outside every timed window: recorded so throughput drift is attributable.
                repeat_host_states.push(capture_host_state(TIMING_SAMPLE_HOST_BOUNDARY)?);
            }
            // The kernel fixture's coordinate operation is timing repeat 0's: the same prompt,
            // operation and session, so it is reused rather than dispatched again.
            let candidate_quality = run_product_fixture_half_on_session_bounded(
                &candidate_session,
                &prompt,
                &row.coordinate,
                policy.max_request_tokens,
                Some(timing_repeats[0].coordinate.clone()),
            )
            .map_err(|e| {
                format!(
                    "candidate quality measurement for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;
            eprintln!(
                "{}",
                fixture_half_attempt_diagnostic(
                    &candidate_quality,
                    &row.coordinate,
                    &identity_sha256,
                    "candidate",
                    "quality",
                    0,
                )
            );
            let candidate_repeats = vec![candidate_quality];
            let compressed_parity = candidate_session
                .compressed()
                .map(CompressedKvArm::kernel_parity_errors)
                .transpose()
                .map_err(|e| format!("compressed kernel parity: {e}"))?;
            drop(candidate_session);
            quiesce_campaign_active_memory(candidate_baseline).map_err(|e| {
                format!(
                    "candidate session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let (reference_session_snapshot, reference_kind) = if compressed.is_some() {
                (&snapshot, QualityReference::DenseKvSameWeights)
            } else {
                (&reference_snapshot, QualityReference::Bf16Characterization)
            };
            let reference_session = CampaignSession::load(reference_session_snapshot)
                .map_err(|e| format!("load reference campaign session: {e}"))?;
            let reference_baseline = reference_session.load_start_sample.mlx_active_bytes;
            // The reference arm is never timed: its one quality measurement is all it runs.
            let reference_quality = run_product_fixture_half_on_session_bounded(
                &reference_session,
                &prompt,
                &row.coordinate,
                policy.max_request_tokens,
                None,
            )
            .map_err(|e| {
                format!(
                    "reference quality measurement for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;
            eprintln!(
                "{}",
                fixture_half_attempt_diagnostic(
                    &reference_quality,
                    &row.coordinate,
                    &identity_sha256,
                    "reference",
                    "quality",
                    0,
                )
            );
            let reference_repeats = vec![reference_quality];
            // Compressed rows: the same-weights dense-KV session greedily continues the kernel
            // fixture prompt through every stop token, once per row (the stream is deterministic),
            // scoring its own likelihood of every token of that stream.
            let forced_reference = compressed
                .map(|_| {
                    let kernel_prompt =
                        kernel_fixture_prompt(&reference_session, &prompt, &row.coordinate)?;
                    reference_session.provider.campaign_forced_continuation(
                        &kernel_prompt,
                        FORCED_CONTINUATION_TOKENS as usize,
                        None,
                        None,
                    )
                })
                .transpose()
                .map_err(|e| {
                    format!(
                        "dense forced continuation for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
            // Contract v4: compressed rows' multi-turn prompt-cache agreement is measured on the
            // same-weights dense-KV session's turn-2 forced continuation, after its own turn-2
            // prompt-cache hit, once per row.
            let multi_turn_reference = compressed
                .map(|_| {
                    let turn1_prompt =
                        multi_turn_fixture_prompt(&reference_session, &prompt, &row.coordinate)?;
                    reference_session
                        .provider
                        .campaign_multi_turn_forced_continuation(
                            &fixture_request(turn1_prompt, Vec::new()),
                            MULTI_TURN_FIXTURE_FOLLOW_UP,
                            FORCED_CONTINUATION_TOKENS as usize,
                            None,
                            None,
                        )
                })
                .transpose()
                .map_err(|e| {
                    format!(
                        "dense turn-2 forced continuation for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
            drop(reference_session);
            quiesce_campaign_active_memory(reference_baseline).map_err(|e| {
                format!(
                    "reference session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            // Greedy agreement (quality contract v3) is teacher-forced: the candidate, reloaded in
            // its own representation, is evaluated on each reference repeat's kernel stream. This
            // runs after every timed dispatch, so it never touches timing or compile attribution.
            let forcing_session = match compressed {
                Some(method) => CampaignSession::load_compressed(&snapshot, method),
                None => CampaignSession::load(&snapshot),
            }
            .map_err(|e| format!("load teacher-forcing candidate session: {e}"))?;
            let forcing_baseline = forcing_session.load_start_sample.mlx_active_bytes;
            // A compressed row is teacher-forced once on its dense forced continuation (every
            // position compared by argmax); a dense row keeps one forced pass per distinct natural
            // reference stream (greedy reference repeats are usually identical, and each pass is
            // a full-context prefill). Each pass also scores the candidate's likelihood of the
            // reference tokens, so the perplexity delta compares both arms on one stream.
            let (teacher_forced, forced_continuation, stream_likelihoods) = match &forced_reference
            {
                Some(reference) => {
                    let reference_stream = &reference.choices;
                    let candidate = kernel_fixture_prompt(&forcing_session, &prompt, &row.coordinate)
                        .and_then(|kernel_prompt| {
                            forcing_session.provider.campaign_forced_continuation(
                                &kernel_prompt,
                                reference_stream.len(),
                                forcing_session.compressed(),
                                Some(reference_stream),
                            )
                        })
                        .map_err(|e| {
                            format!(
                                "teacher-forced continuation for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })?;
                    let continuation =
                        forced_continuation_evidence(reference_stream, &candidate.choices)?;
                    let likelihood = StreamLikelihood::from_probabilities(
                        &reference.stream_probabilities,
                        &candidate.stream_probabilities,
                    )?;
                    eprintln!(
                        "sc20671-kv-baseline: coordinate {} forced continuation agreement {} ({}/{}; first flips {:?})",
                        coordinate_slug(&row.coordinate),
                        continuation.agreement,
                        continuation.matches,
                        continuation.tokens,
                        continuation.first_flip_positions
                    );
                    (
                        vec![None; reference_repeats.len()],
                        Some(continuation),
                        vec![Some(likelihood); reference_repeats.len()],
                    )
                }
                None => {
                    let stop_tokens = forcing_session.provider.campaign_stop_tokens().to_vec();
                    let streams = reference_repeats
                        .iter()
                        .map(|reference| {
                            stream_tokens(
                                &reference.kernel.quality_observation.token_probabilities,
                            )
                        })
                        .collect::<Vec<_>>();
                    let passes = force_distinct_streams(&streams, |stream| {
                        teacher_forced_kernel_choices(
                            &forcing_session,
                            &prompt,
                            &row.coordinate,
                            stream,
                        )
                        .map_err(|e| {
                            format!(
                                "teacher-forced pass for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })
                    })?;
                    // The reference's natural stream is its own greedy output, so its recorded
                    // probabilities are already its likelihood of that stream.
                    let likelihoods = reference_repeats
                        .iter()
                        .zip(&passes)
                        .map(|(reference, pass)| {
                            StreamLikelihood::from_probabilities(
                                &reference
                                    .kernel
                                    .quality_observation
                                    .token_probabilities
                                    .iter()
                                    .map(|(_, probability)| *probability)
                                    .collect::<Vec<_>>(),
                                &pass.stream_probabilities,
                            )
                            .map(Some)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let forced = passes
                        .into_iter()
                        .map(|pass| {
                            Some(TeacherForcedChoices {
                                choices: pass.choices,
                                stop_tokens: stop_tokens.clone(),
                            })
                        })
                        .collect::<Vec<_>>();
                    (forced, None, likelihoods)
                }
            };
            // Multi-turn prompt-cache agreement (contract v4) is teacher-forced the same way, on
            // turn 2 after the candidate's own turn-2 prompt-cache hit: the row's turn-2 forced
            // continuation (compressed) or each reference repeat's natural turn-2 stream (dense).
            let (teacher_forced_cache, multi_turn_continuation, multi_turn_pass) =
                match &multi_turn_reference {
                Some((reference_stream, reference_turns)) => {
                    let (choices, candidate_turns) =
                        multi_turn_fixture_prompt(&forcing_session, &prompt, &row.coordinate)
                        .and_then(|turn1_prompt| {
                            forcing_session
                                .provider
                                .campaign_multi_turn_forced_continuation(
                                    &fixture_request(turn1_prompt, Vec::new()),
                                    MULTI_TURN_FIXTURE_FOLLOW_UP,
                                    reference_stream.len(),
                                    forcing_session.compressed(),
                                    Some(reference_stream),
                                )
                        })
                        .map_err(|e| {
                            format!(
                                "teacher-forced turn-2 continuation for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })?;
                    // The compressed arm is forced on the reference stream of the same turn-2
                    // prompt token ids, or the row is refused.
                    let pass = MultiTurnForcedPass {
                        reference: reference_turns.clone(),
                        candidate: candidate_turns,
                    };
                    pass.validate().map_err(|e| {
                        format!(
                            "turn-2 forced continuation for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        )
                    })?;
                    let continuation =
                        multi_turn_forced_continuation_evidence(reference_stream, &choices)?;
                    eprintln!(
                        "sc20671-kv-baseline: coordinate {} multi-turn turn-2 forced continuation agreement {} ({}/{}; first flips {:?})",
                        coordinate_slug(&row.coordinate),
                        continuation.agreement,
                        continuation.matches,
                        continuation.tokens,
                        continuation.first_flip_positions
                    );
                    (
                        vec![None; reference_repeats.len()],
                        Some(continuation),
                        Some(pass),
                    )
                }
                None => {
                    let stop_tokens = forcing_session.provider.campaign_stop_tokens().to_vec();
                    let streams = reference_repeats
                        .iter()
                        .map(|reference| {
                            stream_tokens(&reference.cache.quality_observation.token_probabilities)
                        })
                        .collect::<Vec<_>>();
                    let forced = force_distinct_streams(&streams, |stream| {
                        teacher_forced_multi_turn_choices(
                            &forcing_session,
                            &prompt,
                            &row.coordinate,
                            stream,
                        )
                        .map_err(|e| {
                            format!(
                                "teacher-forced turn-2 pass for {}: {e}",
                                coordinate_slug(&row.coordinate)
                            )
                        })
                    })?
                    .into_iter()
                    .map(|choices| {
                        Some(TeacherForcedChoices {
                            choices,
                            stop_tokens: stop_tokens.clone(),
                        })
                    })
                    .collect::<Vec<_>>();
                    (forced, None, None)
                }
            };
            drop(forcing_session);
            quiesce_campaign_active_memory(forcing_baseline).map_err(|e| {
                format!(
                    "teacher-forcing session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let suites = candidate_repeats
                .into_iter()
                .zip(reference_repeats)
                .zip(teacher_forced)
                .zip(teacher_forced_cache)
                .zip(stream_likelihoods)
                .enumerate()
                .map(|(repeat, ((((candidate, reference), forced), forced_cache), likelihood))| {
                    let mut suite =
                        pair_product_fixture_halves(candidate, reference, reference_kind)?;
                    suite.kernel_parity_errors = compressed_parity.clone();
                    suite.teacher_forced_candidate = forced;
                    suite.forced_continuation = forced_continuation.clone();
                    suite.stream_likelihood = likelihood;
                    suite.teacher_forced_cache = forced_cache;
                    suite.multi_turn_forced_continuation = multi_turn_continuation.clone();
                    suite.multi_turn_forced_pass = multi_turn_pass.clone();
                    suite.quality().map_err(|e| {
                        core_llm::Error::InvalidRequest(format!(
                            "product quality measurement {repeat} for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        ))
                    })?;
                    Ok(suite)
                })
                .collect::<core_llm::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            let kernel_runs = timing_repeats
                .iter()
                .map(|run| &run.coordinate)
                .collect::<Vec<_>>();
            let warmup_runs = timing_warmups
                .iter()
                .map(|run| &run.coordinate)
                .collect::<Vec<_>>();
            let steady_decodes = timing_repeats
                .iter()
                .map(|run| &run.steady_decode)
                .collect::<Vec<_>>();
            let mut timings = timing_samples_from_product_repeats(
                &kernel_runs,
                &steady_decodes,
                &row.coordinate,
                row.coordinate.process_temperature,
                &warmup_runs,
            )?;
            for (sample, state) in timings.samples.iter_mut().zip(repeat_host_states) {
                sample.host_state = Some(state);
            }
            let executable = std::env::current_exe().map_err(|e| e.to_string())?;
            let fixtures = sealed_product_fixture_artifacts(&suites, &row.coordinate)?;
            let receipt = product_receipt(
                &row.coordinate,
                &suites,
                timings,
                &executable,
                &fixtures,
                warmup_cache_state_version,
                &captured_source,
                admission,
                compressed.map(|method| {
                    (
                        method,
                        timing_warmups
                            .iter()
                            .chain(&timing_repeats)
                            .map(|run| &run.coordinate)
                            .collect::<Vec<_>>(),
                    )
                }),
                &reference_inventory,
                row_start,
            )?;
            let bundle = assemble_artifacts_with_fixtures(receipt, fixtures)?;
            write_artifacts(
                &PathBuf::from(required_flag(args, "--out")?),
                "receipt.json",
                "receipt.md",
                &bundle,
            )
            .map_err(|e| e.to_string())?;
            Ok(CampaignOutcome::Completed)
        }
        "noise-floor-parent" => noise_floor_parent(args),
        "noise-floor" => {
            let value = dense_noise_floor(
                Path::new(&required_flag(args, "--snapshot")?),
                Path::new(&required_flag(args, "--prompt-file")?),
                &required_flag(args, "--coordinate")?,
                optional_flag(args, "--prefill-chunk")?
                    .map(|chunk| chunk.parse::<usize>().map_err(|e| e.to_string()))
                    .transpose()?
                    .unwrap_or(crate::primitives::attention::SDPA_PREFILL_BLOCK_QLEN as usize),
                load_campaign_safety_policy(Path::new(&required_flag(args, "--safety-policy")?))?
                    .max_request_tokens,
            )?;
            let bytes = canonical_json_bytes(&value).map_err(|e| e.to_string())?;
            fs::write(required_flag(args, "--out")?, &bytes).map_err(|e| e.to_string())?;
            println!("{}", String::from_utf8(bytes).map_err(|e| e.to_string())?);
            Ok(CampaignOutcome::Completed)
        }
        _ => Err("usage: sc20671-kv-baseline parent|preflight|worker [--mode dense|compressed --kv-method <method>] [--only-coordinate <coordinate>] [--stop-file <path>] [options] | noise-floor --snapshot <dir> --prompt-file <file> --coordinate <coordinate> --safety-policy <json> --out <json> [--prefill-chunk <tokens>] | noise-floor-parent --llama-snapshot <dir> --qwen-snapshot <dir> --llama-fp32-reference-snapshot <dir> --qwen-fp32-reference-snapshot <dir> --prompt-file <file> --safety-policy <json> --resume-dir <dir> --out <dir> [--prefill-chunk <tokens>] [--stop-file <path>]".into()),
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

/// The frozen covering set: each family runs every context band, while cold/warm,
/// chunked/single-shot, and single/batch each have observable rows. The material baseline for
/// SC-20676 remains warm, single, and single-shot in both families.
/// SC-20669 dense-vs-dense noise floor of the frozen compressed quality thresholds, for one
/// scheduled coordinate's kernel fixture (dense session only; no compressed arm, no receipt). The
/// dense session's forced continuation (exactly the compressed rows' reference, scored) is
/// compared, by the compressed rows' own metrics — teacher-forced greedy agreement and the
/// stream-scored perplexity delta — with two exact dense recomputations of the same stream:
/// `one-shot-repeat` (the identical path again: determinism, expected 1.0 / 0) and
/// `chunked-prefill` (the prompt prefilled in `prefill_chunk`-token steps: exact bf16 dense
/// attention in another reduction order). A compressed row cannot be held to agreement or a
/// perplexity delta tighter than `chunked-prefill` shows the dense arm achieves against itself.
pub fn dense_noise_floor(
    snapshot: &Path,
    prompt_file: &Path,
    coordinate_slug_value: &str,
    prefill_chunk: usize,
    max_request_tokens: u64,
) -> Result<serde_json::Value, String> {
    let coordinate = required_coordinates()
        .into_iter()
        .find(|coordinate| coordinate_slug(coordinate) == coordinate_slug_value)
        .ok_or_else(|| format!("{coordinate_slug_value:?} is not a scheduled coordinate"))?;
    let prompt = fs::read_to_string(prompt_file).map_err(|e| format!("read prompt: {e}"))?;
    let session = CampaignSession::load(snapshot).map_err(|e| e.to_string())?;
    session
        .validate_coordinate_family(&coordinate)
        .map_err(|e| e.to_string())?;
    // Same-process dense timing first, with A2's exact definitions and order (a warm row's
    // warmups, then the timed repeats; a cold row's repeat 0 is the process's first dispatch),
    // and before any scored pass, whose per-token host softmax would distort a timed window.
    let warmups = if coordinate.process_temperature == "warm" {
        TIMING_WARMUPS
    } else {
        0
    };
    for index in 0..warmups {
        run_timing_on_session(&session, &prompt, &coordinate, max_request_tokens)
            .map_err(|e| format!("dense timing warmup {index}: {e}"))?;
    }
    let mut operation = None;
    let samples = (0..TIMING_REPEATS)
        .map(|index| {
            let run = run_timing_on_session(&session, &prompt, &coordinate, max_request_tokens)
                .map_err(|e| format!("dense timing repeat {index}: {e}"))?;
            operation.get_or_insert_with(|| run.coordinate.primary.operation.clone());
            timing_from_product_observation(
                &run.coordinate.primary.observation,
                &run.steady_decode,
                coordinate.process_temperature,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    let dense_timing = noise_floor_dense_timing(
        &coordinate,
        operation.as_deref().unwrap_or_default(),
        warmups,
        &samples,
    )?;
    let kernel_prompt =
        kernel_fixture_prompt(&session, &prompt, &coordinate).map_err(|e| e.to_string())?;
    let provider = &session.provider;
    let reference = provider
        .campaign_forced_continuation(
            &kernel_prompt,
            FORCED_CONTINUATION_TOKENS as usize,
            None,
            None,
        )
        .map_err(|e| format!("dense forced continuation: {e}"))?;
    let stream = &reference.choices;
    let repeat = provider
        .campaign_forced_continuation(&kernel_prompt, stream.len(), None, Some(stream))
        .map_err(|e| format!("dense one-shot repeat: {e}"))?;
    let chunked = provider
        .campaign_chunked_prefill_continuation(&kernel_prompt, stream, prefill_chunk)
        .map_err(|e| format!("dense chunked-prefill control: {e}"))?;
    let snapshot_sha256 = session.inventory.sha256.clone();
    let prompt_tokens = provider
        .campaign_prompt_tokens(&kernel_prompt)
        .map_err(|e| e.to_string())?;
    // Dense multi-turn control: A2's multi-turn fixture exactly (turn 1 stored in the product
    // prompt cache, turn 2 served by its cache hit). The reference turn-2 stream comes from this
    // session, as A2's comes from its dense reference session; the control is teacher-forced on it
    // from a fresh dense session, as A2's compressed arm is from its fresh forcing session. The
    // first session is released before the second loads, so one session is ever resident.
    let turn1 = fixture_request(
        multi_turn_fixture_prompt(&session, &prompt, &coordinate).map_err(|e| e.to_string())?,
        Vec::new(),
    );
    let (multi_turn_reference, reference_turns) = provider
        .campaign_multi_turn_scored_continuation(
            &turn1,
            MULTI_TURN_FIXTURE_FOLLOW_UP,
            FORCED_CONTINUATION_TOKENS as usize,
            None,
        )
        .map_err(|e| format!("dense multi-turn reference: {e}"))?;
    let baseline = session.load_start_sample.mlx_active_bytes;
    drop(session);
    quiesce_campaign_active_memory(baseline)
        .map_err(|e| format!("noise-floor session release: {e}"))?;
    let forcing = CampaignSession::load(snapshot).map_err(|e| e.to_string())?;
    let (multi_turn_control, control_turns) = forcing
        .provider
        .campaign_multi_turn_scored_continuation(
            &turn1,
            MULTI_TURN_FIXTURE_FOLLOW_UP,
            multi_turn_reference.choices.len(),
            Some(&multi_turn_reference.choices),
        )
        .map_err(|e| format!("dense multi-turn control: {e}"))?;
    drop(forcing);
    let mut multi_turn = noise_floor_control(&multi_turn_reference, &multi_turn_control, true)?;
    multi_turn["tokens"] = multi_turn_reference.choices.len().into();
    multi_turn["referenceStreamSha256"] = token_stream_sha256(&multi_turn_reference.choices).into();
    multi_turn["turns"] = serde_json::json!({
        "reference": reference_turns,
        "control": control_turns,
    });
    Ok(serde_json::json!({
        "kind": "sc20669-dense-noise-floor",
        "coordinate": coordinate_slug_value,
        "snapshotSha256": snapshot_sha256,
        "promptTokens": prompt_tokens,
        "tokens": stream.len(),
        "referenceStreamSha256": token_stream_sha256(stream),
        "prefillChunk": prefill_chunk,
        "thresholds": {
            "greedyTokenAgreement": COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN,
            "perplexityDelta": COMPRESSED_PERPLEXITY_DELTA_MAX,
            "multiTurnPromptCache": COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
        },
        "controls": {
            "one-shot-repeat": noise_floor_control(&reference, &repeat, false)?,
            "chunked-prefill": noise_floor_control(&reference, &chunked, false)?,
            "multi-turn-repeat": multi_turn,
        },
        "denseTiming": dense_timing,
    }))
}

/// One noise-floor control scored on its reference stream with the compressed rows' own metrics:
/// teacher-forced agreement (A2's kernel or, with `multi_turn`, turn-2 forced-continuation
/// evidence) and the stream-scored perplexity delta.
pub fn noise_floor_control(
    reference: &ScoredContinuation,
    control: &ScoredContinuation,
    multi_turn: bool,
) -> Result<serde_json::Value, String> {
    let agreement = if multi_turn {
        multi_turn_forced_continuation_evidence(&reference.choices, &control.choices)?
    } else {
        forced_continuation_evidence(&reference.choices, &control.choices)?
    };
    let likelihood = StreamLikelihood::from_probabilities(
        &reference.stream_probabilities,
        &control.stream_probabilities,
    )?;
    Ok(serde_json::json!({
        "method": agreement.method,
        "agreement": agreement.agreement,
        "matches": agreement.matches,
        "flipCount": agreement.flip_count,
        "firstFlipPositions": agreement.first_flip_positions,
        "referenceNegativeLogLikelihood": likelihood.reference,
        "controlNegativeLogLikelihood": likelihood.candidate,
        "perplexityDelta": likelihood.candidate - likelihood.reference,
    }))
}

/// The noise floor's same-process dense timing of one coordinate (`denseTiming`): A2's timing
/// definitions (`run_timing_on_session`, [`timing_from_product_observation`]) and aggregates
/// (`prefillMs`, `ttftMs`, `firstTokenMs` and `decodeTokensPerSecond` are the repeats' means, as
/// a receipt's `timings`), keyed by the coordinate and matrix so a reader pairs it with that A2
/// row. It never shares a window with the scored passes.
pub fn noise_floor_dense_timing(
    coordinate: &Coordinate,
    operation: &str,
    warmups: usize,
    samples: &[RawTiming],
) -> Result<serde_json::Value, String> {
    let expected_warmups = if coordinate.process_temperature == "warm" {
        TIMING_WARMUPS
    } else {
        0
    };
    if samples.len() != TIMING_REPEATS || warmups != expected_warmups || operation.is_empty() {
        return Err(format!(
            "dense timing needs {TIMING_REPEATS} repeats after {expected_warmups} warmups of the coordinate operation, got {} after {warmups}",
            samples.len()
        ));
    }
    let mean = |f: fn(&RawTiming) -> f64| samples.iter().map(f).sum::<f64>() / samples.len() as f64;
    let decode = mean(|t| t.decode_tokens_per_second);
    let variance = samples
        .iter()
        .map(|t| (t.decode_tokens_per_second - decode).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    Ok(serde_json::json!({
        "coordinate": coordinate_slug(coordinate),
        "matrix": {
            "family": coordinate.family,
            "contextBand": coordinate.context_band,
            "requestMode": coordinate.request_mode,
            "prefillMode": coordinate.prefill_mode,
            "processTemperature": coordinate.process_temperature,
        },
        "kvPath": "dense",
        "coordinateOperation": operation,
        "method": "same-process dense session: the kernel-prompt coordinate operation and a fixed-length steady decode per repeat (A2 run_timing_on_session), before and apart from the scored passes",
        "warmups": warmups,
        "repeats": samples.len(),
        "prefillMs": mean(|t| t.prefill_ms),
        "ttftMs": mean(|t| t.ttft_ms),
        "firstTokenMs": mean(|t| t.first_token_ms),
        "decodeTokensPerSecond": decode,
        "decodeTokensPerSecondCoefficientOfVariation": variance.sqrt() / decode,
        "samples": samples.iter().map(|t| serde_json::json!({
            "prefillMs": t.prefill_ms,
            "ttftMs": t.ttft_ms,
            "firstTokenMs": t.first_token_ms,
            "decodeTokensPerSecond": t.decode_tokens_per_second,
            "steadyDecodeTimedTokens": t.steady_decode.timed_tokens,
            "steadyDecodeMs": t.steady_decode.decode_ms,
        })).collect::<Vec<_>>(),
    }))
}

/// Kind of the SC-20669 dense noise-floor summary (`summary.json` of a `noise-floor-parent` run).
pub const NOISE_FLOOR_SUMMARY_KIND: &str = "sc20669-dense-noise-floor-summary";

/// The worst case of each dense noise-floor control over every row (`rows` are
/// [`dense_noise_floor`] outputs): minimum agreement, maximum flip count and maximum
/// `|perplexityDelta|`, beside the frozen thresholds and whether the dense arm's own floor already
/// misses them (a compressed row cannot be held tighter than the dense arm holds itself).
pub fn noise_floor_summary(rows: &[serde_json::Value]) -> Result<serde_json::Value, String> {
    if rows.is_empty() {
        return Err("a noise-floor summary needs at least one row".into());
    }
    let mut controls = serde_json::Map::new();
    for control in ["one-shot-repeat", "chunked-prefill", "multi-turn-repeat"] {
        let (mut agreement, mut flips, mut delta) = (f64::INFINITY, 0_u64, 0.0_f64);
        let mut worst_agreement_row = String::new();
        let mut worst_delta_row = String::new();
        for row in rows {
            let coordinate = row
                .get("coordinate")
                .and_then(serde_json::Value::as_str)
                .ok_or("noise-floor row has no coordinate")?;
            let measured = row
                .get("controls")
                .and_then(|controls| controls.get(control))
                .ok_or_else(|| format!("noise-floor row {coordinate} has no {control} control"))?;
            let number = |key: &str| {
                measured
                    .get(key)
                    .and_then(serde_json::Value::as_f64)
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| format!("noise-floor row {coordinate} {control} lacks {key}"))
            };
            let row_agreement = number("agreement")?;
            let row_delta = number("perplexityDelta")?.abs();
            if row_agreement < agreement {
                agreement = row_agreement;
                worst_agreement_row = coordinate.into();
            }
            if row_delta > delta || worst_delta_row.is_empty() {
                delta = delta.max(row_delta);
                worst_delta_row = coordinate.into();
            }
            flips = flips.max(number("flipCount")? as u64);
        }
        controls.insert(
            control.into(),
            serde_json::json!({
                "minAgreement": agreement,
                "minAgreementRow": worst_agreement_row,
                "maxFlipCount": flips,
                "maxAbsPerplexityDelta": delta,
                "maxAbsPerplexityDeltaRow": worst_delta_row,
                "greedyThresholdWithinDenseFloor": agreement < COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN,
                "perplexityThresholdWithinDenseFloor": delta > COMPRESSED_PERPLEXITY_DELTA_MAX,
                "multiTurnThresholdWithinDenseFloor": control == "multi-turn-repeat"
                    && agreement < COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
            }),
        );
    }
    // Each row's same-process dense timing, keyed by coordinate (the A2 receipt row it pairs with).
    let dense_timing = rows
        .iter()
        .filter_map(|row| {
            let timing = row.get("denseTiming")?;
            Some((
                timing.get("coordinate")?.as_str()?.to_owned(),
                serde_json::json!({
                    "processTemperature": timing.get("matrix")?.get("processTemperature")?,
                    "prefillMs": timing.get("prefillMs")?,
                    "firstTokenMs": timing.get("firstTokenMs")?,
                    "decodeTokensPerSecond": timing.get("decodeTokensPerSecond")?,
                    "decodeTokensPerSecondCoefficientOfVariation": timing
                        .get("decodeTokensPerSecondCoefficientOfVariation")?,
                }),
            ))
        })
        .collect::<serde_json::Map<_, _>>();
    Ok(serde_json::json!({
        "kind": NOISE_FLOOR_SUMMARY_KIND,
        "rows": rows.len(),
        "thresholds": {
            "greedyTokenAgreement": COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN,
            "perplexityDelta": COMPRESSED_PERPLEXITY_DELTA_MAX,
            "multiTurnPromptCache": COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN,
        },
        "controls": controls,
        "denseTiming": dense_timing,
    }))
}

/// Every pinned file of a row's candidate and reference snapshots exists (a cheap check before the
/// header reads of [`static_row_requirements`]); the error names the first missing path.
fn pinned_snapshot_files_present(coordinate: &Coordinate, snapshot: &Path) -> Result<(), String> {
    for (path, spec) in [(snapshot, benchmark_model(coordinate.family, false)?)] {
        for required in spec.required_files {
            let file = path.join(required.path);
            if !file.is_file() {
                return Err(format!(
                    "{}: pinned {} file {} is missing",
                    coordinate_slug(coordinate),
                    spec.repository,
                    file.display()
                ));
            }
        }
    }
    Ok(())
}

/// `noise-floor-parent`: the dense noise floor ([`dense_noise_floor`]) of every scheduled
/// coordinate, one guarded `noise-floor` worker per row under the campaign safety policy (the
/// same estimate-plus-reserve admission, child footprint cap, host reserve and row deadline as an
/// SC-20671 row). Each finished row is sealed in `--resume-dir` (a rerun skips it); a stop file
/// halts between rows with exit status 75. When every row is present, `summary.json`
/// ([`noise_floor_summary`]) is sealed beside them and the directory is renamed to `--out`.
fn noise_floor_parent(args: &[String]) -> Result<CampaignOutcome, String> {
    let resume_dir = PathBuf::from(required_flag(args, "--resume-dir")?);
    let destination = PathBuf::from(required_flag(args, "--out")?);
    if destination.exists() {
        return Err("noise-floor destination must be absent".into());
    }
    let safety_policy = PathBuf::from(required_flag(args, "--safety-policy")?);
    let policy = load_campaign_safety_policy(&safety_policy)?;
    let prompt_file = PathBuf::from(required_flag(args, "--prompt-file")?);
    let prompt = fs::read_to_string(&prompt_file).map_err(|e| format!("read prompt file: {e}"))?;
    // (family, measured candidate snapshot, its pinned bf16 reference). The worker loads only the
    // candidate; the reference is named so the row is admitted on the same static estimate as the
    // SC-20671 dense row (candidate plus reference), an upper bound for one dense session.
    // The measured 4-bit candidate of each family: the worker's only session (the bf16 reference
    // flags phase.sh shares with A1/A2 are accepted and unused).
    let snapshots = [
        (
            "llama",
            PathBuf::from(required_flag(args, "--llama-snapshot")?),
        ),
        (
            "qwen",
            PathBuf::from(required_flag(args, "--qwen-snapshot")?),
        ),
    ];
    let prefill_chunk = optional_flag(args, "--prefill-chunk")?;
    let stop_files = operator_stop_files(args, &resume_dir)?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let logs = resume_dir.join("logs");
    fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
    // `--only-coordinate`: one scheduled row (kv-poc `nf_only_coordinate`); else all eight.
    let only_coordinate = optional_flag(args, "--only-coordinate")?;
    let coordinates = required_coordinates()
        .into_iter()
        .filter(|coordinate| {
            only_coordinate
                .as_deref()
                .is_none_or(|only| coordinate_slug(coordinate) == only)
        })
        .collect::<Vec<_>>();
    if coordinates.is_empty() {
        return Err(format!(
            "--only-coordinate {:?} is not a scheduled coordinate",
            only_coordinate.unwrap_or_default()
        ));
    }
    let mut rows = Vec::with_capacity(coordinates.len());
    for (index, coordinate) in coordinates.iter().enumerate() {
        let slug = coordinate_slug(coordinate);
        let row_path = resume_dir.join(format!("{slug}.json"));
        if row_path.exists() {
            let bytes = fs::read(&row_path).map_err(|e| e.to_string())?;
            let sidecar = fs::read_to_string(resume_dir.join(format!("{slug}.json.sha256")))
                .map_err(|e| e.to_string())?;
            if sidecar != format!("{}  {slug}.json\n", seal_bytes(&bytes)) {
                return Err(format!("noise-floor row {slug} does not match its seal"));
            }
            rows.push(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?);
            continue;
        }
        if let Some(stop) = operator_stop_before_row(
            &stop_files,
            &logs,
            "sc-20669-noise-floor-operator-stop",
            index,
            &slug,
            coordinates.len(),
        )? {
            return Ok(CampaignOutcome::StoppedByOperator(stop));
        }
        let (_, snapshot) = snapshots
            .iter()
            .find(|(family, _)| *family == coordinate.family)
            .ok_or("noise-floor coordinate family has no snapshot")?;
        pinned_snapshot_files_present(coordinate, snapshot)?;
        let (total_tokens, request_tokens, static_footprint_floor) =
            noise_floor_row_requirements(coordinate, snapshot, &prompt, &policy)?;
        let admission = runtime_guarded_admission(
            &policy,
            static_footprint_floor,
            NOISE_FLOOR_ESTIMATE_SOURCE,
        )?;
        let staged = resume_dir.join(format!("{slug}.json.partial"));
        let _ = fs::remove_file(&staged);
        let mut command = Command::new(&executable);
        command
            .arg("noise-floor")
            .arg("--snapshot")
            .arg(snapshot)
            .arg("--prompt-file")
            .arg(&prompt_file)
            .arg("--coordinate")
            .arg(&slug)
            .arg("--safety-policy")
            .arg(&safety_policy)
            .arg("--out")
            .arg(&staged);
        if let Some(chunk) = &prefill_chunk {
            command.arg("--prefill-chunk").arg(chunk);
        }
        let prefix =
            unused_attempt_prefix(&logs, &slug).ok_or("no unused bounded worker log path")?;
        let request = RunRequest {
            context_tokens: total_tokens,
            request_tokens,
            estimate: admission.estimate(),
            stdout_path: logs.join(format!("{prefix}.stdout.log")),
            stderr_path: logs.join(format!("{prefix}.stderr.log")),
        };
        eprintln!(
            "sc20671-kv-baseline: noise floor {}/{} {slug}",
            index + 1,
            coordinates.len()
        );
        let status = campaign_supervisor::run_guarded(
            &mut command,
            &request,
            &policy.supervisor(),
            &mut SystemProbe,
        )
        .map_err(|failure| {
            format!(
                "noise-floor row {slug} stopped ({:?}): {}; earlier rows remain in {}",
                failure.reason,
                failure.detail,
                resume_dir.display()
            )
        })?;
        if !status.success() {
            return Err(format!(
                "noise-floor row {slug} failed with {status}; stderr: {}",
                request.stderr_path.display()
            ));
        }
        let bytes = fs::read(&staged).map_err(|e| e.to_string())?;
        let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if value.get("coordinate").and_then(serde_json::Value::as_str) != Some(slug.as_str()) {
            return Err(format!(
                "noise-floor row {slug} reported another coordinate"
            ));
        }
        let (sealed, sha256) = seal_json(&value)?;
        fs::write(&row_path, &sealed).map_err(|e| e.to_string())?;
        fs::write(
            resume_dir.join(format!("{slug}.json.sha256")),
            format!("{sha256}  {slug}.json\n"),
        )
        .map_err(|e| e.to_string())?;
        let _ = fs::remove_file(&staged);
        rows.push(value);
    }
    let mut summary = noise_floor_summary(&rows)?;
    summary["onlyCoordinate"] = only_coordinate.into();
    let (bytes, sha256) = seal_json(&summary)?;
    fs::write(resume_dir.join("summary.json"), &bytes).map_err(|e| e.to_string())?;
    fs::write(
        resume_dir.join("summary.json.sha256"),
        format!("{sha256}  summary.json\n"),
    )
    .map_err(|e| e.to_string())?;
    println!("{}", String::from_utf8(bytes).map_err(|e| e.to_string())?);
    fs::rename(&resume_dir, &destination).map_err(|e| {
        format!(
            "publish noise floor {} -> {}: {e}",
            resume_dir.display(),
            destination.display()
        )
    })?;
    Ok(CampaignOutcome::Completed)
}

pub fn required_coordinates() -> Vec<Coordinate> {
    [
        ("llama", "short", "single", "chunked", "cold"),
        ("llama", "medium", "supported-batch", "single-shot", "warm"),
        ("llama", "memory-material", "single", "single-shot", "warm"),
        ("llama", "fit-boundary", "single", "chunked", "cold"),
        ("qwen", "short", "single", "single-shot", "cold"),
        ("qwen", "medium", "supported-batch", "chunked", "warm"),
        ("qwen", "memory-material", "single", "single-shot", "warm"),
        ("qwen", "fit-boundary", "single", "chunked", "cold"),
    ]
    .into_iter()
    .map(
        |(family, context_band, request_mode, prefill_mode, process_temperature)| Coordinate {
            family,
            context_band,
            request_mode,
            prefill_mode,
            process_temperature,
        },
    )
    .collect()
}

/// Historical v1 publications remain readable as their original complete 64-row matrix.
fn legacy_required_coordinates() -> Vec<Coordinate> {
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
/// list, which prevents a selectively successful campaign from being published as the baseline;
/// `--only-coordinate` runs one of these rows and publishes it only as a partial run.
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

fn legacy_required_schedule() -> Vec<ScheduledCoordinate> {
    legacy_required_coordinates()
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
    if ![8, 64].contains(&scheduled.len()) || outcomes.len() != scheduled.len() {
        return Err("campaign schedule must contain exactly 8 or 64 outcomes".into());
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
    if expected.len() != scheduled.len() {
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
#[cfg(test)]
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

/// Validate a materialized snapshot against its source-owned model contract before the product
/// loader sees it.  A filesystem path is a storage location only: it cannot select a repository,
/// revision, family, architecture, context window, precision arm, or weight payload.
pub fn validate_benchmark_snapshot(
    root: impl AsRef<Path>,
    spec: &BenchmarkModelSpec,
) -> Result<SnapshotInventory, String> {
    if spec.native_context_tokens < SC20671_MIN_NATIVE_CONTEXT_TOKENS {
        return Err(format!(
            "{} {} is below SC-20671's 32k native-context minimum",
            spec.repository, spec.revision
        ));
    }
    let inventory = inventory_snapshot(root).map_err(|error| error.to_string())?;
    for expected in spec.required_files {
        let actual = inventory
            .files
            .iter()
            .find(|file| file.path == expected.path)
            .ok_or_else(|| {
                format!(
                    "{}@{} lacks required {}",
                    spec.repository, spec.revision, expected.path
                )
            })?;
        if actual.bytes != expected.bytes || actual.sha256 != expected.sha256 {
            return Err(format!(
                "{}@{} has an unexpected {} inventory entry",
                spec.repository, spec.revision, expected.path
            ));
        }
    }
    let config_bytes = fs::read(inventory.root.join("config.json"))
        .map_err(|error| format!("read {} config: {error}", spec.repository))?;
    let config: serde_json::Value = serde_json::from_slice(&config_bytes)
        .map_err(|error| format!("parse {} config: {error}", spec.repository))?;
    let architecture = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str);
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let context = config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64);
    let quantized = config
        .get("quantization_config")
        .is_some_and(|value| !value.is_null());
    if architecture != Some(spec.architecture)
        || model_type != Some(spec.model_type)
        || context != Some(spec.native_context_tokens)
        || quantized != spec.quantized
    {
        return Err(format!(
            "{}@{} configuration does not match its frozen {} {} contract",
            spec.repository, spec.revision, spec.family, spec.role
        ));
    }
    Ok(inventory)
}

fn validate_coordinate_model_contract(
    coordinate: &Coordinate,
    snapshot: &Path,
    reference_snapshot: &Path,
) -> Result<(SnapshotInventory, SnapshotInventory), String> {
    let candidate = benchmark_model(coordinate.family, false)?;
    let reference = benchmark_model(coordinate.family, true)?;
    if snapshot == reference_snapshot {
        return Err("candidate and higher-precision reference paths must be distinct".into());
    }
    Ok((
        validate_benchmark_snapshot(snapshot, candidate)?,
        validate_benchmark_snapshot(reference_snapshot, reference)?,
    ))
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

/// Which run is the denominator of a fixture comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityReference {
    /// Dense rows: raw characterization of weight quantization against the bf16 model. Recorded,
    /// never gated.
    Bf16Characterization,
    /// Compressed rows: the dense-KV run on the same weights, so every gated metric isolates the
    /// effect of KV compression.
    DenseKvSameWeights,
}

/// Raw per-fixture behaviour of both arms, recorded as observations in the sealed artifacts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixtureOutcomes {
    pub candidate_tool_valid: bool,
    pub reference_tool_valid: bool,
    pub tool_outputs_match: bool,
    pub candidate_needle_recovered: bool,
    pub reference_needle_recovered: bool,
    pub needle_outputs_match: bool,
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
    pub outcomes: FixtureOutcomes,
    /// Whether the needle check can detect KV-induced retrieval loss: the same-weights dense-KV
    /// run recovered the exact needle.
    pub needle_discriminating: bool,
    /// Whether tool agreement can detect KV-induced tool-call loss: the same-weights dense-KV run
    /// emitted the valid structured call.
    pub tool_discriminating: bool,
    /// Observation only: the first position at which the free-running candidate and reference
    /// kernel streams differ (`None` when identical).
    pub free_running_first_divergence: Option<u64>,
    /// Teacher-forced greedy agreement of every measured repeat, in order (empty for one suite).
    pub greedy_agreement_by_repeat: Vec<f64>,
    /// Every measured repeat's quality metrics, in order (empty for one suite). A compressed
    /// receipt's quality gate evaluates each.
    pub repeat_metrics: Vec<QualityMetrics>,
    /// Compressed rows: the forced continuation whose agreement is the greedy agreement.
    pub forced_continuation: Option<ReceiptForcedContinuation>,
    /// Observation only: the first position at which the free-running candidate and reference
    /// turn-2 streams of the multi-turn fixture differ (`None` when identical).
    pub cache_free_running_first_divergence: Option<u64>,
    /// Observation only: leading turn-2 positions on which those free-running streams agree.
    pub cache_matched_prefix_tokens: u64,
    /// Both arms' prompt-cache records of the multi-turn fixture (turn 2 must hit in each).
    pub cache_candidate_turns: MultiTurnPromptCacheTurns,
    pub cache_reference_turns: MultiTurnPromptCacheTurns,
    /// Compressed rows: the turn-2 forced continuation whose agreement is `multiTurnPromptCache`.
    pub multi_turn_forced_continuation: Option<ReceiptForcedContinuation>,
    /// Compressed rows: both sessions' turn records of that forced continuation.
    pub multi_turn_forced_pass: Option<MultiTurnForcedPass>,
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
    coordinate: &Coordinate,
    repeat: usize,
) -> Result<(Vec<u8>, String), String> {
    let quality = suite.quality()?;
    let (evidence, candidate, reference) = match name {
        "kernel-fp32-reference" => (
            serde_json::json!({
                "candidatePerplexity": quality.candidate_perplexity,
                "referencePerplexity": quality.reference_perplexity,
                "parityErrors": quality.parity_errors,
                "greedyMatches": quality.greedy_matches,
                "greedyTotal": quality.greedy_total,
                "freeRunningFirstDivergence": quality.free_running_first_divergence,
            }),
            &suite.kernel_candidate,
            &suite.kernel_reference,
        ),
        "structured-tool-call" => (
            serde_json::json!({
                "matches": quality.tool_matches,
                "total": quality.tool_total,
                "candidateValid": quality.outcomes.candidate_tool_valid,
                "referenceValid": quality.outcomes.reference_tool_valid,
                "outputsMatch": quality.outcomes.tool_outputs_match,
                "discriminating": quality.tool_discriminating,
            }),
            &suite.tool_candidate,
            &suite.tool_reference,
        ),
        "long-context-needle" => (
            serde_json::json!({
                "matches": quality.needle_matches,
                "total": quality.needle_total,
                "candidateRecovered": quality.outcomes.candidate_needle_recovered,
                "referenceRecovered": quality.outcomes.reference_needle_recovered,
                "outputsMatch": quality.outcomes.needle_outputs_match,
                "discriminating": quality.needle_discriminating,
            }),
            &suite.needle_candidate,
            &suite.needle_reference,
        ),
        "multi-turn-prompt-cache" => (
            serde_json::json!({
                "matches": quality.cache_matches,
                "total": quality.cache_total,
                "method": MULTI_TURN_PROMPT_CACHE_METHOD,
                "freeRunningFirstDivergence": quality.cache_free_running_first_divergence,
                "matchedPrefixTokens": quality.cache_matched_prefix_tokens,
                "turns": {
                    "candidate": quality.cache_candidate_turns,
                    "reference": quality.cache_reference_turns,
                },
            }),
            &suite.cache_candidate,
            &suite.cache_reference,
        ),
        _ => return Err(format!("unknown fixture {name}")),
    };
    let mut evidence = evidence;
    if let (Some(continuation), "kernel-fp32-reference") = (&quality.forced_continuation, name) {
        evidence["forcedContinuation"] =
            serde_json::to_value(continuation).map_err(|e| e.to_string())?;
    }
    if let (Some(continuation), "multi-turn-prompt-cache") =
        (&quality.multi_turn_forced_continuation, name)
    {
        evidence["forcedContinuation"] =
            serde_json::to_value(continuation).map_err(|e| e.to_string())?;
    }
    if let (Some(pass), "multi-turn-prompt-cache") = (&quality.multi_turn_forced_pass, name) {
        evidence["forcedPass"] = serde_json::to_value(pass).map_err(|e| e.to_string())?;
    }
    let metrics = compute_quality(&quality)?;
    let value = serde_json::json!({
        "fixture": name,
        "independentReference": fixture_independent_reference(name, suite.reference, &reference.quality_observation.snapshot.sha256),
        "binding": {
            "coordinate": coordinate_slug(coordinate),
            "repeat": repeat,
            "candidate": {
                "coordinateInventorySha256": candidate.observation.snapshot.sha256.as_str(),
                "coordinateSessionId": candidate.observation.session_id.as_str(),
                "qualityInventorySha256": candidate.quality_observation.snapshot.sha256.as_str(),
                "qualitySessionId": candidate.quality_observation.session_id.as_str(),
                "operation": candidate.coordinate_operation.as_str(),
                "operationOutputSha256": candidate.coordinate_output_sha256.as_str(),
                "operationEvidenceSha256": primary_operation_evidence_digest(candidate),
                "coordinateEvidenceSha256": coordinate_operation_digest(candidate),
                "operationGeneratedTokens": candidate.coordinate_generated_tokens,
                "operationPromptTokens": candidate.coordinate_prompt_tokens,
                "compileSetupMs": candidate.compile_setup_ms,
                "compileDispatchMs": candidate.compile_dispatch_ms,
                "qualityTranscriptSha256": product_output_digest(&candidate.output),
                "secondaryOperation": candidate.secondary_coordinate_operation.as_ref().map(|secondary| serde_json::json!({
                    "operation": secondary.operation.as_str(),
                    "outputSha256": secondary.output_sha256.as_str(),
                    "evidenceSha256": operation_evidence_digest(secondary),
                    "sessionId": secondary.observation.session_id.as_str(),
                    "cacheStateVersion": secondary.observation.cache_state_version,
                    "generatedTokens": secondary.generated_tokens,
                    "promptTokens": secondary.prompt_tokens,
                    "compileSetupMs": secondary.compile_setup_ms,
                    "compileDispatchMs": secondary.compile_dispatch_ms,
                })),
            },
            "reference": {
                "coordinateInventorySha256": reference.observation.snapshot.sha256.as_str(),
                "coordinateSessionId": reference.observation.session_id.as_str(),
                "qualityInventorySha256": reference.quality_observation.snapshot.sha256.as_str(),
                "qualitySessionId": reference.quality_observation.session_id.as_str(),
                "operation": reference.coordinate_operation.as_str(),
                "operationOutputSha256": reference.coordinate_output_sha256.as_str(),
                "operationEvidenceSha256": primary_operation_evidence_digest(reference),
                "coordinateEvidenceSha256": coordinate_operation_digest(reference),
                "operationGeneratedTokens": reference.coordinate_generated_tokens,
                "operationPromptTokens": reference.coordinate_prompt_tokens,
                "compileSetupMs": reference.compile_setup_ms,
                "compileDispatchMs": reference.compile_dispatch_ms,
                "qualityTranscriptSha256": product_output_digest(&reference.output),
                "secondaryOperation": reference.secondary_coordinate_operation.as_ref().map(|secondary| serde_json::json!({
                    "operation": secondary.operation.as_str(),
                    "outputSha256": secondary.output_sha256.as_str(),
                    "evidenceSha256": operation_evidence_digest(secondary),
                    "sessionId": secondary.observation.session_id.as_str(),
                    "cacheStateVersion": secondary.observation.cache_state_version,
                    "generatedTokens": secondary.generated_tokens,
                    "promptTokens": secondary.prompt_tokens,
                    "compileSetupMs": secondary.compile_setup_ms,
                    "compileDispatchMs": secondary.compile_dispatch_ms,
                })),
            },
        },
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

fn fixture_independent_reference(
    name: &str,
    reference: QualityReference,
    reference_inventory: &str,
) -> String {
    let model = match reference {
        QualityReference::Bf16Characterization => format!("bf16-model:{reference_inventory}"),
        QualityReference::DenseKvSameWeights => {
            format!("{COMPRESSED_QUALITY_REFERENCE}:{reference_inventory}")
        }
    };
    // Kernel parity keeps its independent host-fp32 dequantize-then-attend reference either way.
    if name == "kernel-fp32-reference" {
        format!("host-fp32-dense-attention-v1;{model}")
    } else {
        model
    }
}

fn fixture_artifact_name(fixture: &str, repeat: usize) -> String {
    if repeat == 0 {
        format!("fixtures/{fixture}.json")
    } else {
        format!("fixtures/repeat-{repeat}/{fixture}.json")
    }
}

fn sealed_product_fixture_artifacts(
    suites: &[ProductFixtureSuite],
    coordinate: &Coordinate,
) -> Result<Vec<SealedFixtureArtifact>, String> {
    if suites.len() != QUALITY_MEASUREMENTS {
        return Err(
            "complete product evidence requires exactly the row's one quality measurement".into(),
        );
    }
    let mut artifacts = Vec::with_capacity(REQUIRED_FIXTURES.len() * suites.len());
    for (repeat, suite) in suites.iter().enumerate() {
        for fixture in REQUIRED_FIXTURES {
            let (bytes, hash) = product_fixture_artifact(fixture, suite, coordinate, repeat)?;
            let name = fixture_artifact_name(fixture, repeat);
            artifacts.push(SealedFixtureArtifact {
                sidecar: format!("{hash}  {name}\n"),
                name,
                bytes,
            });
        }
    }
    Ok(artifacts)
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
/// Teacher forcing for the campaign's greedy-agreement measurement: the decode loop feeds the
/// forced token at each step instead of its own choice, which it still reports through
/// [`Observer::token_probability`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeacherForcing {
    /// Ordinary generation.
    Off,
    /// Feed this token at the current step.
    Token(i32),
    /// The forced stream is exhausted: stop before sampling another position.
    Exhausted,
}

pub trait Observer {
    fn phase(&mut self, name: &'static str);
    /// Teacher forcing for decode `step` of the observed generation; off unless a campaign quality
    /// observer supplies a forced stream.
    fn teacher_forced_token(&mut self, _step: usize) -> TeacherForcing {
        TeacherForcing::Off
    }
    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64);
    fn allocation_event(
        &mut self,
        _kind: &'static str,
        role: &'static str,
        lifetime: &'static str,
        bytes: u64,
    ) {
        self.allocation(role, lifetime, bytes);
    }
    /// Observe a cache-ownership release. This is deliberately separate from allocation_event:
    /// released bytes are lifecycle evidence, never positive workspace or retained allocation.
    fn release_event(&mut self, _kind: &'static str, _role: &'static str, _bytes: u64) {}
    /// Wall-clock duration of the real snapshot inventory plus provider/model load that created
    /// the bound campaign session.
    fn load_duration(&mut self, _milliseconds: f64) {}
    /// Candidate-only samples bracketing explicit model-parameter materialization.  The campaign
    /// observer replays these as the first two receipt phases so later reference allocations cannot
    /// be mistaken for candidate weights.
    fn load_boundary(&mut self, _process_start: &MemorySample, _weights_loaded: &MemorySample) {}
    /// Reset MLX's process-global high-water immediately before the measured prefill and seal the
    /// active-memory baseline independently from the historical model-load boundary.
    fn begin_prefill_memory_window(&mut self) {}
    /// Current cumulative cache ownership at a decoder boundary. The product cache supplies bytes,
    /// live length, allocated capacity, and actual MLX array scalar width so append events are not
    /// summed and the receipt cannot confuse model-weight dtype with cache dtype.
    fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _capacity: u64, _element_bytes: u64) {
        self.allocation("cache", "persistent", bytes);
    }
    /// Bind a product-owned provider-load identity before the first phase.  The default preserves
    /// existing non-campaign observers without permitting receipt assembly to invent an identity.
    fn bind_session(&mut self, _session_id: &str) {}
    /// Cache state is versioned by the provider-load session.  It prevents a same-PID reload from
    /// being mislabeled as a warmed provider/cache.
    fn cache_state(&mut self, _version: u64) {}
    /// Record a coordinate operation that actually executed on this observed product session.
    fn operation(&mut self, _operation: &'static str) {}
    /// Exact fp32 logits materialized from the production decode seam only while a campaign
    /// observer is attached.  Ordinary serving never takes this host-read path.
    fn logits(&mut self, _stage: &'static str, _values: &[f32]) {}
    /// Exact probability of the sampled production token, derived at the same decode seam as the
    /// sampler.  It is intentionally not a caller-provided quality number.
    fn token_probability(&mut self, _stage: &'static str, _token: i32, _probability: f64) {}
    /// Under teacher forcing, the model's exact probability of the forced token at the same decode
    /// seam (beside [`Self::token_probability`] of its own choice), so the candidate's likelihood
    /// is measured on the reference's tokens.
    fn forced_token_probability(&mut self, _token: i32, _probability: f64) {}
    /// Model identity is derived from the exact files resolved by the product loader, never a
    /// caller-authored digest string.
    fn snapshot_inventory(&mut self, _inventory: &SnapshotInventory) {}
    fn geometry(&mut self, _geometry: ProductGeometry) {}
    /// Immutable model-boundary evidence of one compressed (packed) decoder cache, forwarded once
    /// after decode. Dense caches never call it.
    fn packed_cache_evidence(&mut self, _evidence: &crate::primitives::PackedCacheEvidence) {}
    /// A compressed-arm operation that ran on the explicit dense path, with its reason.
    fn dense_fallback(&mut self, _operation: &str, _reason: &str) {}
    /// A packed-to-dense transition rebuilt the whole cached history as dense K/V.
    fn dense_reconstruction(&mut self, _bytes: u64) {}
    /// Measured physical storage of a live compressed cache at a decoder boundary.
    fn compressed_storage(&mut self, _storage: &crate::primitives::CompressedCacheStorage) {}
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
    session_id: Option<String>,
    cache_state_version: Option<u64>,
    operations: Vec<String>,
    load_elapsed_ms: Option<f64>,
    load_boundary: Option<(MemorySample, MemorySample)>,
    prefill_peak_window: Option<ReceiptPeakWindow>,
    cache_live_tokens: u64,
    cache_capacity_tokens: u64,
    sampling_elapsed_ms: f64,
    packed_evidence: Vec<crate::primitives::PackedCacheEvidence>,
    dense_fallbacks: Vec<(String, String)>,
    compressed_storage_peak: Option<crate::primitives::CompressedCacheStorage>,
    /// `(packed evidence, dense fallbacks)` recorded before the coordinate's measured dispatch
    /// opened; set once by [`ProductObserver::begin_coordinate_operation`].
    coordinate_start: Option<(usize, usize)>,
    coordinate_scope: Option<CoordinateCompressionScope>,
    /// The reference token stream fed back at each decode step (teacher forcing), if any.
    forced_tokens: Option<Vec<i32>>,
    /// The model's probability of each forced token, in decode order (teacher forcing only).
    forced_token_probabilities: Vec<f64>,
    /// Teacher forcing covers only the first observed generation; a second step 0 ends it.
    forcing_started: bool,
    forcing_done: bool,
}

/// Compressed-arm evidence of one coordinate operation's measured dispatch only (opened by
/// [`ProductObserver::begin_coordinate_operation`] after its setup, or from observer creation when
/// never opened), closed by [`ProductObserver::end_coordinate_operation`] before the lifecycle probes
/// (prompt-cache reuse, cancellation) run on the same observer.
#[derive(Clone, Debug, Default)]
pub struct CoordinateCompressionScope {
    pub packed_evidence: Vec<crate::primitives::PackedCacheEvidence>,
    pub dense_fallbacks: Vec<(String, String)>,
    pub storage_peak: Option<crate::primitives::CompressedCacheStorage>,
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
            session_id: None,
            cache_state_version: None,
            operations: Vec::new(),
            load_elapsed_ms: None,
            load_boundary: None,
            prefill_peak_window: None,
            cache_live_tokens: 0,
            cache_capacity_tokens: 0,
            sampling_elapsed_ms: 0.0,
            packed_evidence: Vec::new(),
            dense_fallbacks: Vec::new(),
            compressed_storage_peak: None,
            coordinate_start: None,
            coordinate_scope: None,
            forced_tokens: None,
            forced_token_probabilities: Vec::new(),
            forcing_started: false,
            forcing_done: false,
        }
    }

    /// An observer whose first observed generation is teacher-forced on `tokens`: at every
    /// position the model's own greedy choice is recorded and the forced token is fed back.
    pub fn teacher_forced(tokens: Vec<i32>) -> Self {
        Self {
            forced_tokens: Some(tokens),
            ..Self::new()
        }
    }

    /// Open the coordinate operation's measured dispatch. Evidence recorded before it — setup
    /// such as seeding the provider's dense prefix store — stays counted arm-wide but never
    /// classifies the coordinate's representation or supplies its storage.
    pub fn begin_coordinate_operation(&mut self) {
        if self.coordinate_start.is_some() || self.coordinate_scope.is_some() {
            self.error = Some("coordinate operation opened twice or after it closed".into());
            return;
        }
        self.coordinate_start = Some((self.packed_evidence.len(), self.dense_fallbacks.len()));
        self.compressed_storage_peak = None;
    }

    /// The coordinate operation's storage at its persistent-KV peak (test inspection).
    #[cfg(test)]
    pub(crate) fn coordinate_storage_peak(
        &self,
    ) -> Option<crate::primitives::CompressedCacheStorage> {
        self.coordinate_scope
            .as_ref()
            .and_then(|scope| scope.storage_peak)
    }

    /// Close the coordinate operation: compressed evidence recorded so far describes the measured
    /// operation, and anything recorded afterwards is lifecycle evidence of the same observer.
    pub fn end_coordinate_operation(&mut self) {
        if self.coordinate_scope.is_some() {
            self.error = Some("duplicate coordinate-operation scope".into());
            return;
        }
        let (evidence, fallbacks) = self.coordinate_start.unwrap_or((0, 0));
        self.coordinate_scope = Some(CoordinateCompressionScope {
            packed_evidence: self.packed_evidence[evidence..].to_vec(),
            dense_fallbacks: self.dense_fallbacks[fallbacks..].to_vec(),
            storage_peak: self.compressed_storage_peak,
        });
    }

    pub fn bind_session(&mut self, session_id: impl Into<String>) {
        self.session_id = Some(session_id.into());
    }

    pub fn cache_state(&mut self, version: u64) {
        if version == 0 {
            self.error = Some("campaign cache state version must be positive".into());
        } else {
            self.cache_state_version = Some(version);
        }
    }

    pub fn operation(&mut self, operation: &'static str) {
        self.operations.push(operation.into());
    }

    fn sampling_elapsed_ms(&self) -> f64 {
        self.sampling_elapsed_ms
    }

    /// The observer's product clock: wall time since creation minus every synchronous memory
    /// sample the observer itself took. Phase timings are deltas of this clock, so a memory sample
    /// at a phase boundary is never relabeled as prefill, first-token, or decode time.
    fn product_elapsed_ms(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1_000.0 - self.sampling_elapsed_ms
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
            || self
                .forced_token_probabilities
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || (self.forced_tokens.is_none() && !self.forced_token_probabilities.is_empty())
        {
            return Err("product observer has invalid numeric decode evidence".into());
        }
        if self.phases.len() != REQUIRED_PHASES.len() {
            return Err("product observer did not capture the exact phase set".into());
        }
        let prefill_peak_window = self
            .prefill_peak_window
            .ok_or("product observer did not reset the MLX prefill peak window")?;
        if self.cache_live_tokens == 0 || self.cache_capacity_tokens == 0 {
            return Err("product observer did not capture a cumulative cache snapshot".into());
        }
        let mut live_cache = None;
        let mut releases = 0_u64;
        for event in &self.allocations {
            if event.role == "cache" && event.lifetime == "persistent" {
                live_cache = Some(event.bytes);
            } else if event.lifetime == "released" {
                if event.kind != "product-cache_release"
                    || event.role != "cache"
                    || live_cache != Some(event.bytes)
                {
                    return Err("product observer received an invalid cache release".into());
                }
                live_cache = None;
                releases = releases
                    .checked_add(1)
                    .ok_or("product cache release event count overflow")?;
            }
        }
        if releases == 0 || live_cache.is_some() {
            return Err("product observer finalized before releasing cache ownership".into());
        }
        Ok(ProductObservations {
            snapshot: self.snapshot.expect("checked above"),
            geometry: self.geometry.expect("checked above"),
            phases: self.phases,
            phase_elapsed_ms: self.phase_elapsed_ms,
            allocations: self.allocations,
            prefill_logits,
            token_probabilities: self.token_probabilities,
            forced_token_probabilities: self.forced_token_probabilities,
            session_id: self
                .session_id
                .ok_or("product observer is missing campaign session")?,
            cache_state_version: self
                .cache_state_version
                .ok_or("product observer is missing provider cache-state evidence")?,
            operations: self.operations,
            load_elapsed_ms: self
                .load_elapsed_ms
                .ok_or("product observer is missing measured snapshot load duration")?,
            prefill_peak_window,
            cache_live_tokens: self.cache_live_tokens,
            cache_capacity_tokens: self.cache_capacity_tokens,
            packed_evidence: self.packed_evidence,
            dense_fallbacks: self.dense_fallbacks,
            coordinate_scope: self.coordinate_scope,
        })
    }
}

impl Default for ProductObserver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct ProductObservations {
    pub snapshot: SnapshotInventory,
    pub geometry: ProductGeometry,
    pub phases: Vec<ReceiptPhase>,
    pub phase_elapsed_ms: Vec<f64>,
    pub allocations: Vec<ReceiptAllocation>,
    pub prefill_logits: Vec<f32>,
    pub token_probabilities: Vec<(i32, f64)>,
    /// Teacher-forced observations only: the model's probability of each forced token.
    pub forced_token_probabilities: Vec<f64>,
    pub session_id: String,
    pub cache_state_version: u64,
    pub operations: Vec<String>,
    pub load_elapsed_ms: f64,
    pub prefill_peak_window: ReceiptPeakWindow,
    pub cache_live_tokens: u64,
    pub cache_capacity_tokens: u64,
    /// Compressed-arm packed cache evidence, one entry per packed decoder cache (empty for dense).
    pub packed_evidence: Vec<crate::primitives::PackedCacheEvidence>,
    /// Compressed-arm operations that ran on the explicit dense path, with their reasons.
    pub dense_fallbacks: Vec<(String, String)>,
    /// The coordinate operation's own compressed evidence; `None` for observers that never ran a
    /// coordinate operation (fixture quality/lifecycle runs).
    pub coordinate_scope: Option<CoordinateCompressionScope>,
}

impl Observer for ProductObserver {
    fn load_duration(&mut self, milliseconds: f64) {
        if self.load_elapsed_ms.is_some() || !milliseconds.is_finite() || milliseconds <= 0.0 {
            self.error = Some("invalid or duplicate product snapshot load duration".into());
        } else {
            self.load_elapsed_ms = Some(milliseconds);
        }
    }

    fn load_boundary(&mut self, process_start: &MemorySample, weights_loaded: &MemorySample) {
        if self.load_boundary.is_some()
            || measured_model_weight_bytes(process_start, weights_loaded).is_err()
        {
            self.error = Some("invalid or duplicate product load boundary".into());
        } else {
            self.load_boundary = Some((process_start.clone(), weights_loaded.clone()));
        }
    }

    fn begin_prefill_memory_window(&mut self) {
        if self.prefill_peak_window.is_some()
            || self.phase != Some("process-start")
            || self.phases.len() != 1
        {
            self.error = Some("invalid or duplicate MLX prefill peak-window reset".into());
            return;
        }
        if self.load_boundary.is_none() {
            self.error = Some("MLX prefill peak-window reset has no load boundary".into());
            return;
        }
        mlx_rs::memory::reset_peak_memory();
        let sampling_started = std::time::Instant::now();
        let sampled = sample_memory(self.pid);
        self.sampling_elapsed_ms += sampling_started.elapsed().as_secs_f64() * 1_000.0;
        match sampled {
            Ok(sample) if sample.mlx_active_bytes > 0 && sample.mlx_peak_bytes == 0 => {
                self.prefill_peak_window = Some(ReceiptPeakWindow {
                    started_at: sample.captured_at,
                    baseline_active_bytes: sample.mlx_active_bytes,
                    reset_peak_bytes: sample.mlx_peak_bytes,
                });
            }
            Ok(_) => {
                self.error =
                    Some("MLX prefill peak-window reset did not produce a positive active baseline and zero peak".into())
            }
            Err(error) => {
                self.error = Some(format!("MLX prefill peak-window baseline sample: {error}"))
            }
        }
    }

    fn cache_snapshot(&mut self, bytes: u64, tokens: u64, capacity: u64, element_bytes: u64) {
        if bytes == 0 || tokens == 0 || capacity < tokens || element_bytes == 0 {
            self.error = Some(
                "product cache snapshot must have positive bytes, valid live/capacity tokens, and element width".into(),
            );
            return;
        }
        let Some(geometry) = self.geometry.as_mut() else {
            self.error = Some("product cache snapshot arrived before loaded geometry".into());
            return;
        };
        if geometry.element_bytes == 0 {
            geometry.element_bytes = element_bytes;
        } else if geometry.element_bytes != element_bytes {
            self.error = Some("product cache element width changed within one coordinate".into());
            return;
        }
        if element_bytes != DENSE_KV_COMPUTE_ELEMENT_BYTES {
            self.error = Some(dense_kv_width_refusal(element_bytes));
            return;
        }
        self.cache_live_tokens = self.cache_live_tokens.max(tokens);
        self.cache_capacity_tokens = self.cache_capacity_tokens.max(capacity);
        self.allocation("cache", "persistent", bytes);
    }

    fn bind_session(&mut self, session_id: &str) {
        self.bind_session(session_id);
    }

    fn cache_state(&mut self, version: u64) {
        self.cache_state(version);
    }

    fn operation(&mut self, operation: &'static str) {
        self.operation(operation);
    }

    fn phase(&mut self, name: &'static str) {
        let expected = REQUIRED_PHASES.get(self.phases.len()).copied();
        if self.error.is_some() || expected != Some(name) {
            self.error = Some(format!("duplicate or out-of-order product phase {name}"));
            return;
        }
        let sampling_started = std::time::Instant::now();
        let sampled = match name {
            "process-start" => self
                .load_boundary
                .as_ref()
                .map(|(sample, _)| sample.clone())
                .ok_or_else(|| std::io::Error::other("missing product load-start sample")),
            "weights-loaded" => self
                .load_boundary
                .as_ref()
                .map(|(_, sample)| sample.clone())
                .ok_or_else(|| std::io::Error::other("missing product weights-loaded sample")),
            _ => sample_memory(self.pid),
        };
        self.sampling_elapsed_ms += sampling_started.elapsed().as_secs_f64() * 1_000.0;
        match sampled {
            Ok(sample) => {
                self.phases.push(ReceiptPhase {
                    phase: name.into(),
                    pid: sample.pid,
                    source: PHYS_FOOTPRINT_SOURCE.into(),
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
                self.phase_elapsed_ms.push(self.product_elapsed_ms());
            }
            Err(error) => self.error = Some(format!("product memory sample at {name}: {error}")),
        }
        self.phase = Some(name);
    }

    fn allocation(&mut self, role: &'static str, lifetime: &'static str, bytes: u64) {
        self.allocation_event(role, role, lifetime, bytes);
    }

    fn allocation_event(
        &mut self,
        kind: &'static str,
        role: &'static str,
        lifetime: &'static str,
        bytes: u64,
    ) {
        let Some(phase) = self.phase else {
            self.error = Some("allocation observed before a product phase".into());
            return;
        };
        if bytes == 0 {
            return;
        }
        self.allocations.push(ReceiptAllocation {
            kind: format!("product-{kind}"),
            role: role.into(),
            lifetime: lifetime.into(),
            phase: phase.into(),
            timestamp: timestamp_now(),
            bytes,
        });
    }

    fn release_event(&mut self, kind: &'static str, role: &'static str, bytes: u64) {
        let Some(phase) = self.phase else {
            self.error = Some("release observed before a product phase".into());
            return;
        };
        if bytes == 0 {
            return;
        }
        self.allocations.push(ReceiptAllocation {
            kind: format!("product-{kind}"),
            role: role.into(),
            lifetime: "released".into(),
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

    fn teacher_forced_token(&mut self, step: usize) -> TeacherForcing {
        let Some(forced) = &self.forced_tokens else {
            return TeacherForcing::Off;
        };
        if step == 0 {
            if self.forcing_started {
                self.forcing_done = true;
            }
            self.forcing_started = true;
        }
        if self.forcing_done {
            return TeacherForcing::Off;
        }
        forced.get(step).map_or(TeacherForcing::Exhausted, |token| {
            TeacherForcing::Token(*token)
        })
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

    fn forced_token_probability(&mut self, token: i32, probability: f64) {
        if self.forced_tokens.is_none()
            || token < 0
            || !probability.is_finite()
            || !(0.0..=1.0).contains(&probability)
        {
            self.error = Some("invalid teacher-forced token probability".into());
            return;
        }
        self.forced_token_probabilities.push(probability);
    }

    fn snapshot_inventory(&mut self, inventory: &SnapshotInventory) {
        self.snapshot = Some(inventory.clone());
    }
    fn geometry(&mut self, geometry: ProductGeometry) {
        self.geometry = Some(geometry);
    }

    fn packed_cache_evidence(&mut self, evidence: &crate::primitives::PackedCacheEvidence) {
        self.packed_evidence.push(evidence.clone());
    }

    fn dense_fallback(&mut self, operation: &str, reason: &str) {
        if operation.trim().is_empty() || reason.trim().is_empty() {
            self.error = Some("compressed dense fallback lacks an operation or reason".into());
            return;
        }
        self.dense_fallbacks.push((operation.into(), reason.into()));
    }

    /// Keep the storage at the persistent-KV peak: the largest device share (which is what
    /// `cache_snapshot` reports as persistent KV), then the largest whole physical footprint, so
    /// the receipt's physical bytes describe the same instant as `memory.persistentKvBytes`.
    /// Device arrays grow only by whole blocks and the dense residual is a fixed group, so the
    /// peak is a plateau across every append inside the last block: the latest (most live tokens)
    /// instant of it is the one whose length is the receipt's `kvLength`.
    fn compressed_storage(&mut self, storage: &crate::primitives::CompressedCacheStorage) {
        let key = |s: &crate::primitives::CompressedCacheStorage| {
            (
                s.device_bytes(),
                s.device_bytes().saturating_add(s.host_payload_bytes),
                s.tokens,
            )
        };
        if self
            .compressed_storage_peak
            .is_none_or(|peak| key(storage) > key(&peak))
        {
            self.compressed_storage_peak = Some(*storage);
        }
    }

    fn dense_reconstruction(&mut self, bytes: u64) {
        let Some(phase) = self.phase else {
            self.error = Some("dense reconstruction observed before a product phase".into());
            return;
        };
        // Deliberately unprefixed: this is the shared full-cache temporary witness kind that both
        // receipt validators reject for compressed rows.
        self.allocations.push(ReceiptAllocation {
            kind: FULL_CACHE_MATERIALIZATION_KIND.into(),
            role: "cache".into(),
            lifetime: "transient".into(),
            phase: phase.into(),
            timestamp: timestamp_now(),
            bytes: bytes.max(1),
        });
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

/// Derive live candidate parameter residency from the exact MLX active-memory boundary around
/// explicit weight evaluation. Snapshot file bytes remain provenance only: tokenizer/config bytes,
/// safetensors headers, and allocator cache reservation are deliberately excluded.
fn measured_model_weight_bytes(
    process_start: &MemorySample,
    weights_loaded: &MemorySample,
) -> Result<u64, String> {
    if process_start.pid == 0
        || process_start.pid != weights_loaded.pid
        || process_start.captured_at >= weights_loaded.captured_at
    {
        return Err("campaign weight samples do not form one ordered process boundary".into());
    }
    if weights_loaded.mlx_active_bytes > weights_loaded.current_bytes {
        return Err("campaign weight sample exceeds Darwin physical footprint".into());
    }
    weights_loaded
        .mlx_active_bytes
        .checked_sub(process_start.mlx_active_bytes)
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| {
            "campaign model materialization produced no positive MLX active delta".into()
        })
}

/// Dense KV as a share of the prefill-peak footprint (basis points, floor) and whether a
/// memory-material row falls below [`MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS`]: recorded, not refused.
fn dense_kv_share(dense: u64, prefill_footprint: u64, band: &str) -> Result<(u64, bool), String> {
    if prefill_footprint == 0 {
        return Err("prefill-peak footprint is zero".into());
    }
    let bps = u128::from(dense).saturating_mul(10_000) / u128::from(prefill_footprint);
    let bps = u64::try_from(bps).map_err(|_| "dense KV share overflows u64".to_string())?;
    Ok((
        bps,
        band == "memory-material"
            && u128::from(dense).saturating_mul(10_000)
                < u128::from(prefill_footprint)
                    .saturating_mul(u128::from(MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS)),
    ))
}

/// Release evidence from the weights-loaded and post-run-release phase samples: the frozen
/// footprint allowance, the MLX slack over each weights-loaded counter, and the recorded residuals.
fn release_evidence(weights_loaded: &ReceiptPhase, release: &ReceiptPhase) -> ReceiptRelease {
    let active_tolerance = post_release_mlx_slack_bytes(weights_loaded.mlx.active_bytes);
    let cache_tolerance = post_release_mlx_slack_bytes(weights_loaded.mlx.cache_bytes);
    ReceiptRelease {
        verified: release.phys_footprint_bytes
            <= weights_loaded
                .phys_footprint_bytes
                .saturating_add(POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES)
            && release.mlx.active_bytes
                <= weights_loaded
                    .mlx
                    .active_bytes
                    .saturating_add(active_tolerance)
            && release.mlx.cache_bytes
                <= weights_loaded
                    .mlx
                    .cache_bytes
                    .saturating_add(cache_tolerance),
        phys_footprint_tolerance_bytes: POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES,
        mlx_active_tolerance_bytes: active_tolerance,
        mlx_cache_tolerance_bytes: cache_tolerance,
        mlx_active_residual_bytes: release
            .mlx
            .active_bytes
            .saturating_sub(weights_loaded.mlx.active_bytes),
        mlx_cache_residual_bytes: release
            .mlx
            .cache_bytes
            .saturating_sub(weights_loaded.mlx.cache_bytes),
    }
}

/// Between sessions: MLX active memory returns to `expected_active_bytes` and the cache empties,
/// each within [`post_release_mlx_slack_bytes`]. A residual inside the slack is logged, not fatal.
fn quiesce_campaign_active_memory(expected_active_bytes: u64) -> core_llm::Result<()> {
    mlx_rs::memory::clear_cache();
    let active = mlx_rs::memory::get_active_memory() as u64;
    let cache = mlx_rs::memory::get_cache_memory() as u64;
    quiesced_within_slack(active, cache, expected_active_bytes).map_err(core_llm::Error::Load)
}

fn quiesced_within_slack(active: u64, cache: u64, expected_active: u64) -> Result<(), String> {
    let active_residual = active.saturating_sub(expected_active);
    if active_residual > post_release_mlx_slack_bytes(expected_active)
        || cache > post_release_mlx_slack_bytes(0)
    {
        return Err(format!(
            "campaign session did not quiesce before the next model load: active={active}, expected={expected_active}, cache={cache}"
        ));
    }
    if active_residual > 0 || cache > 0 {
        eprintln!(
            "sc20671-kv-baseline: session quiesced with a recorded residual within slack: active residual={active_residual} bytes, cache={cache} bytes"
        );
    }
    Ok(())
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
            source: PHYS_FOOTPRINT_SOURCE.into(),
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
            .any(|w| !utc_timestamp_before(&w[0].timestamp, &w[1].timestamp))
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

/// Provenance recorded on every phase sample: one `proc_pid_rusage(RUSAGE_INFO_V4)` read of the
/// Darwin `phys_footprint` ledger (`ri_phys_footprint`, `ri_lifetime_max_phys_footprint`).
pub const PHYS_FOOTPRINT_SOURCE: &str = "proc_pid_rusage";
/// Receipts written before sc-20671 moved sampling off the `/usr/bin/footprint` subprocess. The
/// tool prints the same ledger, so those receipts stay valid.
const LEGACY_PHYS_FOOTPRINT_SOURCE: &str = "footprint -p";

fn valid_phys_footprint_source(source: &str) -> bool {
    source == PHYS_FOOTPRINT_SOURCE || source == LEGACY_PHYS_FOOTPRINT_SOURCE
}

#[cfg(target_os = "macos")]
pub fn sample_memory(pid: u32) -> std::io::Result<MemorySample> {
    // Settled: freed Metal memory leaves `phys_footprint` asynchronously (see
    // `settled_phys_footprint`), and the release check compares against the weights-loaded read.
    let footprint = crate::campaign_supervisor::settled_phys_footprint(pid)?;
    Ok(MemorySample {
        captured_at: timestamp_now(),
        pid,
        current_bytes: footprint.current_bytes,
        peak_bytes: footprint.lifetime_peak_bytes,
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
    let session = CampaignSession::load(snapshot)?;
    observer.bind_session(session.session_id());
    observer.load_duration(session.load_elapsed_ms);
    observer.load_boundary(&session.load_start_sample, &session.weights_loaded_sample);
    observer.phase("process-start");
    let output = {
        let provider = &session.provider;
        observer.snapshot_inventory(session.inventory());
        observer.geometry(provider.campaign_geometry());
        observer.begin_prefill_memory_window();
        observer.phase("weights-loaded");
        observer.allocation("weights", "persistent", session.model_weights_bytes());
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
            session.compressed(),
        )?;
        if !output_has_observed_generation(&output, saw_token) {
            return Err(core_llm::Error::InvalidRequest(
                "dense campaign produced neither a streamed token nor a parsed tool call".into(),
            ));
        }
        output
    };
    drop(session); // Provider/model ownership is released before the final sample.
    mlx_rs::memory::clear_cache();
    observer.phase("post-run-release");
    Ok(output)
}

/// Full one-process product lifecycle used by a worker in the covering-set runner. The parent
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
    let session = CampaignSession::load(snapshot)?;
    run_dense_lifecycle_request_on_session(&session, prefix_prompt, request, coordinate, observer)
}

pub fn run_dense_lifecycle_request_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: TextLlmRequest,
    coordinate: Option<&Coordinate>,
    observer: &mut dyn Observer,
) -> core_llm::Result<TextLlmOutput> {
    run_lifecycle_request_on_session(session, prefix_prompt, request, None, coordinate, observer)
        .map(|(output, _)| output)
}

/// [`run_dense_lifecycle_request_on_session`], optionally as the multi-turn prompt-cache fixture:
/// with `follow_up`, the observed generation is turn 2 of a two-turn conversation served by the
/// product prompt cache ([`crate::provider::LlamaProvider::campaign_multi_turn_observed`]), and
/// both turns' cache records are returned.
fn run_lifecycle_request_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: TextLlmRequest,
    follow_up: Option<&str>,
    coordinate: Option<&Coordinate>,
    observer: &mut dyn Observer,
) -> core_llm::Result<(TextLlmOutput, Option<MultiTurnPromptCacheTurns>)> {
    if let Some(coordinate) = coordinate {
        session.validate_coordinate_family(coordinate)?;
    }
    observer.bind_session(session.session_id());
    observer.load_duration(session.load_elapsed_ms);
    observer.load_boundary(&session.load_start_sample, &session.weights_loaded_sample);
    observer.phase("process-start");
    let inventory = session.inventory();
    let provider = &session.provider;
    let output = {
        observer.snapshot_inventory(inventory);
        observer.geometry(provider.campaign_geometry());
        if let Some(coordinate) = coordinate {
            if coordinate.request_mode == "supported-batch" {
                provider.campaign_supported_batch(prefix_prompt, 2)?;
                session.dense_fallback(
                    observer,
                    "supported-batch",
                    COMPRESSED_BATCH_FALLBACK_REASON,
                );
                observer.operation("supported-batch");
            }
            if coordinate.prefill_mode == "chunked" {
                provider.campaign_prefix_reuse(prefix_prompt)?;
                session.dense_fallback(
                    observer,
                    "chunked-prefix-reuse",
                    COMPRESSED_PREFIX_REUSE_FALLBACK_REASON,
                );
                observer.operation("chunked-prefix-reuse");
            }
        }
        observer.begin_prefill_memory_window();
        observer.phase("weights-loaded");
        observer.allocation("weights", "persistent", session.model_weights_bytes());
        let mut saw_token = false;
        let mut on_event = |event| saw_token |= matches!(event, StreamEvent::Token { .. });
        let (output, turns) = match follow_up {
            Some(follow_up) => provider
                .campaign_multi_turn_observed(
                    &request,
                    follow_up,
                    &mut on_event,
                    observer,
                    session.compressed(),
                )
                .map(|(output, turns)| (output, Some(turns)))?,
            None => (
                provider.generate_observed(
                    &request,
                    &mut on_event,
                    observer,
                    session.compressed(),
                )?,
                None,
            ),
        };
        if !output_has_observed_generation(&output, saw_token) {
            return Err(core_llm::Error::InvalidRequest(
                "dense campaign produced neither a streamed token nor a parsed tool call".into(),
            ));
        }
        observer.operation("single-shot-generation");
        if coordinate.is_some_and(|c| c.prefill_mode == "chunked") {
            observer.operation("prompt-cache-reuse");
        }
        // A successful prefix reuse is the product-owned cache proof for this exact provider
        // load.  The version is monotonically scoped to `CampaignSession`, not this PID.
        observer.cache_state(session.advance_cache_state());
        observer.phase("prompt-cache-reuse");
        provider.campaign_cancel_after_first_token(observer, session.compressed())?;
        (output, turns)
    };
    provider.campaign_release_cache_state();
    mlx_rs::memory::clear_cache();
    observer.phase("post-run-release");
    Ok(output)
}

/// Product-owned result of one independently executed frozen quality fixture.  The artifact name
/// is selected by the worker, but every numeric value and emitted tool/token fact comes from the
/// loaded provider through the observer and `TextLlmOutput`.
pub struct ProductFixtureResult {
    /// Product observations for the coordinate's actual dispatch (single, batch, or prefix reuse).
    /// Receipt memory/timing/geometry consume this field exclusively.
    pub observation: ProductObservations,
    /// Independently observed single-request fixture output used only for the frozen quality checks.
    /// It is sealed alongside the coordinate observation and is never used as coordinate telemetry.
    pub quality_observation: ProductObservations,
    pub output: TextLlmOutput,
    /// The multi-turn prompt-cache fixture only: both turns' prompt-cache records (the quality
    /// output is turn 2's). `None` for every single-turn fixture.
    pub multi_turn: Option<MultiTurnPromptCacheTurns>,
    pub coordinate_operation: String,
    pub coordinate_generated_tokens: u64,
    pub coordinate_prompt_tokens: u64,
    pub coordinate_output_sha256: String,
    /// Product-call time consumed before the observed prefill boundary. Prefix reuse uses this for
    /// its cache-seeding dispatch; all other coordinate operations report zero.
    pub compile_setup_ms: f64,
    /// Complete compile-attribution probe for the coordinate's product calls. It excludes the
    /// campaign's pre-dispatch footprint and allocator-reset instrumentation.
    pub compile_dispatch_ms: f64,
    /// Mixed batch/chunked coordinates seal a second independently observed product dispatch.
    /// It is absent for the ordinary single-operation rows.
    pub secondary_coordinate_operation: Option<CoordinateOperationEvidence>,
}

#[derive(Clone)]
pub struct CoordinateOperationEvidence {
    pub observation: ProductObservations,
    pub operation: String,
    pub generated_tokens: u64,
    pub prompt_tokens: u64,
    pub output_sha256: String,
    pub compile_setup_ms: f64,
    pub compile_dispatch_ms: f64,
}

/// The coordinate operation(s) one fixture prompt runs: the coordinate's primary operation and,
/// on a supported-batch chunked row, its secondary chunked-prefix-reuse dispatch.
#[derive(Clone)]
pub struct CoordinateRun {
    pub primary: CoordinateOperationEvidence,
    pub secondary: Option<CoordinateOperationEvidence>,
}

impl CoordinateRun {
    /// The coordinate fields of a fixture result (tests and timing over fixture results).
    pub fn of(result: &ProductFixtureResult) -> Self {
        Self {
            primary: CoordinateOperationEvidence {
                observation: result.observation.clone(),
                operation: result.coordinate_operation.clone(),
                generated_tokens: result.coordinate_generated_tokens,
                prompt_tokens: result.coordinate_prompt_tokens,
                output_sha256: result.coordinate_output_sha256.clone(),
                compile_setup_ms: result.compile_setup_ms,
                compile_dispatch_ms: result.compile_dispatch_ms,
            },
            secondary: result.secondary_coordinate_operation.clone(),
        }
    }
}

/// One timing measurement (contract v5 `statistics.repeats` / `warmups`): the kernel fixture
/// prompt's coordinate operation — prefill, time to first token and the compile probe — and the
/// fixed-length steady decode. Quality is never re-measured here.
pub struct TimingRun {
    pub coordinate: CoordinateRun,
    pub steady_decode: SteadyDecodeMeasurement,
}

fn output_has_observed_generation(output: &TextLlmOutput, saw_streamed_token: bool) -> bool {
    output.usage.generated_tokens > 0 && (saw_streamed_token || !output.tool_calls.is_empty())
}

/// Bind every externally meaningful output channel. Tool-only responses deliberately have empty
/// `text`, so hashing text alone would make distinct structured calls indistinguishable.
fn product_output_digest(output: &TextLlmOutput) -> String {
    let tool_calls = output
        .tool_calls
        .iter()
        .map(|call| {
            serde_json::json!({
                "name": call.name,
                "arguments": call.arguments,
            })
        })
        .collect::<Vec<_>>();
    let finish_reason = match output.finish_reason {
        Some(core_llm::FinishReason::Stop) => "stop",
        Some(core_llm::FinishReason::Length) => "length",
        Some(core_llm::FinishReason::Cancelled) => "cancelled",
        Some(core_llm::FinishReason::ContentFilter) => "content-filter",
        None => "none",
    };
    let (bytes, _) = sealed_json(&serde_json::json!({
        "text": output.text,
        "thinking": output.thinking,
        "toolCalls": tool_calls,
        "usage": {
            "promptTokens": output.usage.prompt_tokens,
            "generatedTokens": output.usage.generated_tokens,
        },
        "finishReason": finish_reason,
    }));
    seal_bytes(&bytes)
}

fn canonical_operation_evidence_sha256(value: &serde_json::Value) -> String {
    let bytes = canonical_json_bytes(value).expect("operation evidence must serialize");
    seal_bytes(&bytes)
}

fn operation_evidence_digest(evidence: &CoordinateOperationEvidence) -> String {
    let value = serde_json::json!({
        "operation": evidence.operation,
        "outputSha256": evidence.output_sha256,
        "generatedTokens": evidence.generated_tokens,
        "promptTokens": evidence.prompt_tokens,
        "cacheStateVersion": evidence.observation.cache_state_version,
        "phaseElapsedMs": evidence.observation.phase_elapsed_ms,
        "tokenProbabilities": evidence.observation.token_probabilities,
        "allocations": evidence.observation.allocations,
        "compileSetupMs": evidence.compile_setup_ms,
        "compileDispatchMs": evidence.compile_dispatch_ms,
    });
    canonical_operation_evidence_sha256(&value)
}

fn primary_operation_evidence_digest(result: &ProductFixtureResult) -> String {
    let value = serde_json::json!({
        "operation": result.coordinate_operation,
        "outputSha256": result.coordinate_output_sha256,
        "generatedTokens": result.coordinate_generated_tokens,
        "promptTokens": result.coordinate_prompt_tokens,
        "cacheStateVersion": result.observation.cache_state_version,
        "phaseElapsedMs": result.observation.phase_elapsed_ms,
        "tokenProbabilities": result.observation.token_probabilities,
        "allocations": result.observation.allocations,
        "compileSetupMs": result.compile_setup_ms,
        "compileDispatchMs": result.compile_dispatch_ms,
    });
    canonical_operation_evidence_sha256(&value)
}

/// [`primary_operation_evidence_digest`] of a timing run's primary coordinate operation (the same
/// evidence fields, so a reused kernel coordinate operation keeps one digest).
fn coordinate_run_evidence_digest(run: &CoordinateRun) -> String {
    operation_evidence_digest(&run.primary)
}

/// Stable contract binding shared by every repeat for one loaded session. Volatile timings,
/// outputs, and monotonically advancing cache versions remain bound by each repeat's separate
/// operation-evidence digest and therefore cannot make the shared coordinate digest incoherent.
fn coordinate_operation_digest(result: &ProductFixtureResult) -> String {
    let secondary = result
        .secondary_coordinate_operation
        .as_ref()
        .map(|evidence| {
            format!(
                "{}:{}:{}",
                evidence.operation, evidence.prompt_tokens, evidence.observation.session_id
            )
        })
        .unwrap_or_default();
    seal_bytes(
        format!(
            "inventory={};session={};primary={}:{};secondary={secondary}",
            result.observation.snapshot.sha256,
            result.observation.session_id,
            result.coordinate_operation,
            result.coordinate_prompt_tokens,
        )
        .as_bytes(),
    )
}

pub fn run_product_fixture(
    snapshot: impl AsRef<Path>,
    prefix_prompt: &str,
    request: TextLlmRequest,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureResult> {
    let session = CampaignSession::load(snapshot)?;
    run_product_fixture_on_session(&session, prefix_prompt, request, coordinate)
}

pub fn run_product_fixture_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: TextLlmRequest,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureResult> {
    run_fixture_on_session(session, prefix_prompt, request, None, coordinate)
}

/// One product fixture: the coordinate operation on `prefix_prompt` (with `request`), then the
/// quality request — with `multi_turn = (follow_up, turn 1)`, the multi-turn prompt-cache fixture
/// (contract v4), whose turn-1 request may carry a shorter, fit-sized prompt than the coordinate
/// operation's full band prompt.
fn run_fixture_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: TextLlmRequest,
    multi_turn: Option<(&str, TextLlmRequest)>,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureResult> {
    let run = run_coordinate_on_session(session, prefix_prompt, &request, coordinate)?;
    run_fixture_quality_on_session(session, run, prefix_prompt, request, multi_turn)
}

/// The coordinate operation(s) `coordinate` requires over one fixture prompt, each a separately
/// observed product dispatch.
fn run_coordinate_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: &TextLlmRequest,
    coordinate: &Coordinate,
) -> core_llm::Result<CoordinateRun> {
    session.validate_coordinate_family(coordinate)?;
    let primary_operation = if coordinate.request_mode == "supported-batch" {
        "supported-batch"
    } else if coordinate.prefill_mode == "chunked" {
        "chunked-prefix-reuse"
    } else {
        "single-shot-generation"
    };
    let primary = run_coordinate_operation_on_session(
        session,
        prefix_prompt,
        request.clone(),
        primary_operation,
    )?;
    let secondary_coordinate_operation =
        if coordinate.request_mode == "supported-batch" && coordinate.prefill_mode == "chunked" {
            Some(run_coordinate_operation_on_session(
                session,
                prefix_prompt,
                request.clone(),
                "chunked-prefix-reuse",
            )?)
        } else {
            None
        };
    Ok(CoordinateRun {
        primary,
        secondary: secondary_coordinate_operation,
    })
}

/// The fixture's quality request over `run`'s prompt: a separately bound product operation after
/// the coordinate dispatch(es) `run` already measured on this session.
fn run_fixture_quality_on_session(
    session: &CampaignSession,
    run: CoordinateRun,
    prefix_prompt: &str,
    request: TextLlmRequest,
    multi_turn: Option<(&str, TextLlmRequest)>,
) -> core_llm::Result<ProductFixtureResult> {
    let CoordinateRun {
        primary,
        secondary: secondary_coordinate_operation,
    } = run;
    let mut observer = ProductObserver::new();
    let (follow_up, quality_request) = match multi_turn {
        Some((follow_up, turn1)) => (Some(follow_up), turn1),
        None => (None, request),
    };
    let (output, multi_turn) = run_lifecycle_request_on_session(
        session,
        prefix_prompt,
        quality_request,
        follow_up,
        None,
        &mut observer,
    )?;
    let quality_observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    Ok(ProductFixtureResult {
        observation: primary.observation,
        quality_observation,
        output,
        multi_turn,
        coordinate_operation: primary.operation,
        coordinate_generated_tokens: primary.generated_tokens,
        coordinate_prompt_tokens: primary.prompt_tokens,
        coordinate_output_sha256: primary.output_sha256,
        compile_setup_ms: primary.compile_setup_ms,
        compile_dispatch_ms: primary.compile_dispatch_ms,
        secondary_coordinate_operation,
    })
}

/// Time one product dispatch while removing only the synchronous memory-sampling work performed
/// by the campaign observer. The provider remains responsible for invoking the observer at its
/// real phase boundaries; this wrapper prevents memory-sampling latency from being relabeled
/// as model compilation or execution time.
fn measure_product_dispatch<T>(
    observer: &mut ProductObserver,
    dispatch: impl FnOnce(&mut ProductObserver) -> core_llm::Result<T>,
) -> core_llm::Result<(T, f64)> {
    let sampling_before_ms = observer.sampling_elapsed_ms();
    let dispatch_started = std::time::Instant::now();
    let value = dispatch(observer)?;
    let wall_ms = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
    let sampling_after_ms = observer.sampling_elapsed_ms();
    let sampled_ms = sampling_after_ms - sampling_before_ms;
    let product_ms = wall_ms - sampled_ms;
    if !wall_ms.is_finite()
        || !sampling_before_ms.is_finite()
        || !sampling_after_ms.is_finite()
        || sampled_ms < 0.0
        || !product_ms.is_finite()
        || product_ms <= 0.0
    {
        return Err(core_llm::Error::InvalidRequest(format!(
            "invalid sampling-adjusted product duration: wall={wall_ms:.6}ms sampled={sampled_ms:.6}ms product={product_ms:.6}ms"
        )));
    }
    Ok((value, product_ms))
}

/// Execute the exact requested coordinate while a product observer is attached.  Quality fixtures
/// are intentionally a second, separately-bound product operation: they may not relabel a normal
/// single request as a batch or prefix-reuse measurement.
fn run_coordinate_operation_on_session(
    session: &CampaignSession,
    prefix_prompt: &str,
    request: TextLlmRequest,
    operation: &str,
) -> core_llm::Result<CoordinateOperationEvidence> {
    let provider = &session.provider;
    let mut observer = ProductObserver::new();
    observer.bind_session(session.session_id());
    observer.load_duration(session.load_elapsed_ms);
    observer.load_boundary(&session.load_start_sample, &session.weights_loaded_sample);
    observer.phase("process-start");
    observer.snapshot_inventory(session.inventory());
    observer.geometry(provider.campaign_geometry());
    let compile_setup_ms = if operation == "chunked-prefix-reuse" {
        let seeded = provider.campaign_seed_prefix_reuse(prefix_prompt)?;
        session.dense_fallback(
            &mut observer,
            "chunked-prefix-seed",
            COMPRESSED_PREFIX_REUSE_FALLBACK_REASON,
        );
        seeded
    } else {
        0.0
    };
    // The dense prefix-store seed above is setup: counted, but it never classifies the
    // coordinate's own (compressed or reasoned-dense) persistent KV.
    observer.begin_coordinate_operation();
    observer.begin_prefill_memory_window();
    observer.phase("weights-loaded");
    observer.allocation("weights", "persistent", session.model_weights_bytes());

    let (operation, generated_tokens, prompt_tokens, output_sha256, observed_dispatch_ms) =
        if operation == "supported-batch" {
            let ((outputs, prompt_tokens), dispatch_elapsed_ms) =
                measure_product_dispatch(&mut observer, |observer| {
                    provider.campaign_supported_batch_observed(prefix_prompt, 2, observer)
                })?;
            session.dense_fallback(
                &mut observer,
                "supported-batch",
                COMPRESSED_BATCH_FALLBACK_REASON,
            );
            let mut bytes = Vec::new();
            let generated = outputs
                .iter()
                .map(|output| output.tokens.len() as u64)
                .sum();
            for output in outputs {
                for token in output.tokens {
                    bytes.extend_from_slice(&token.to_le_bytes());
                }
            }
            observer.operation("supported-batch");
            (
                "supported-batch".to_string(),
                generated,
                prompt_tokens,
                seal_bytes(&bytes),
                dispatch_elapsed_ms,
            )
        } else if operation == "chunked-prefix-reuse" {
            // A compressed arm imports the reused prefix into its compressed cache; a declined
            // import is recorded by the prefix path as a reasoned fallback.
            let ((output, hits, prompt_tokens), dispatch_elapsed_ms) =
                measure_product_dispatch(&mut observer, |observer| {
                    provider.campaign_prefix_reuse_observed(
                        prefix_prompt,
                        observer,
                        session.compressed(),
                    )
                })?;
            if hits == 0 {
                return Err(core_llm::Error::InvalidRequest(
                    "observed prefix coordinate has no cache hit".into(),
                ));
            }
            let mut bytes = Vec::new();
            for token in &output.tokens {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
            observer.operation("chunked-prefix-reuse");
            (
                "chunked-prefix-reuse".to_string(),
                output.tokens.len() as u64,
                prompt_tokens,
                seal_bytes(&bytes),
                dispatch_elapsed_ms,
            )
        } else if operation == "single-shot-generation" {
            let mut saw_token = false;
            let (output, dispatch_elapsed_ms) =
                measure_product_dispatch(&mut observer, |observer| {
                    provider.generate_observed(
                        &request,
                        &mut |event| saw_token |= matches!(event, StreamEvent::Token { .. }),
                        observer,
                        session.compressed(),
                    )
                })?;
            if !output_has_observed_generation(&output, saw_token) {
                return Err(core_llm::Error::InvalidRequest(
                    "single coordinate produced neither a streamed token nor a parsed tool call"
                        .into(),
                ));
            }
            observer.operation("single-shot-generation");
            (
                "single-shot-generation".to_string(),
                output.usage.generated_tokens as u64,
                output.usage.prompt_tokens as u64,
                product_output_digest(&output),
                dispatch_elapsed_ms,
            )
        } else {
            return Err(core_llm::Error::InvalidRequest(
                "unknown campaign coordinate operation".into(),
            ));
        };
    let compile_dispatch_ms = compile_setup_ms + observed_dispatch_ms;
    if !compile_dispatch_ms.is_finite() || compile_dispatch_ms <= 0.0 {
        return Err(core_llm::Error::InvalidRequest(
            "coordinate produced no positive compile-attribution dispatch duration".into(),
        ));
    }
    finish_coordinate_lifecycle(
        &mut observer,
        session.compressed().is_some(),
        || provider.campaign_prefix_reuse(prefix_prompt).map(drop),
        || session.advance_cache_state(),
        |observer| provider.campaign_cancel_after_first_token(observer, session.compressed()),
        || {
            provider.campaign_release_cache_state();
            mlx_rs::memory::clear_cache();
        },
    )?;
    let observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    Ok(CoordinateOperationEvidence {
        observation,
        operation,
        generated_tokens,
        prompt_tokens,
        output_sha256,
        compile_setup_ms,
        compile_dispatch_ms,
    })
}

/// Prompt-cache reuse and deliberate cancellation are additional lifecycle facts of every
/// coordinate observer. They do not replace the coordinate operation measured before them: the
/// coordinate scope is closed first, so their (reasoned dense or compressed) evidence can never
/// classify the measured representation.
fn finish_coordinate_lifecycle(
    observer: &mut ProductObserver,
    compressed: bool,
    prefix_reuse: impl FnOnce() -> core_llm::Result<()>,
    advance_cache_state: impl FnOnce() -> u64,
    cancel_after_first_token: impl FnOnce(&mut ProductObserver) -> core_llm::Result<()>,
    release_cache_state: impl FnOnce(),
) -> core_llm::Result<()> {
    observer.end_coordinate_operation();
    prefix_reuse()?;
    if compressed {
        Observer::dense_fallback(
            observer,
            "prompt-cache-reuse",
            COMPRESSED_PREFIX_REUSE_FALLBACK_REASON,
        );
    }
    observer.cache_state(advance_cache_state());
    observer.phase("prompt-cache-reuse");
    cancel_after_first_token(observer)?;
    release_cache_state();
    observer.phase("post-run-release");
    Ok(())
}

fn negative_log_likelihood(probabilities: &[(i32, f64)]) -> Result<f64, String> {
    mean_negative_log_likelihood(
        &probabilities
            .iter()
            .map(|(_, probability)| *probability)
            .collect::<Vec<_>>(),
    )
}

fn mean_negative_log_likelihood(probabilities: &[f64]) -> Result<f64, String> {
    if probabilities.is_empty()
        || probabilities
            .iter()
            .any(|probability| !probability.is_finite() || *probability <= 0.0)
    {
        return Err("missing finite product token probabilities".into());
    }
    Ok(-probabilities
        .iter()
        .map(|probability| probability.ln())
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

/// Run a fixed real MLX dense-attention dispatch against an independently accumulated fp64 host
/// reference. Model-level candidate/bf16 differences must never be relabeled as kernel parity.
fn dense_kernel_fp32_parity_errors() -> Result<Vec<f64>, String> {
    const QUERY_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const TOKENS: usize = 16;
    const WIDTH: usize = 128;
    let query = (0..QUERY_HEADS * WIDTH)
        .map(|index| (index as i32 % 17 - 8) as f32 * 0.002)
        .collect::<Vec<_>>();
    let keys = (0..KV_HEADS * TOKENS * WIDTH)
        .map(|index| (index as i32 % 29 - 14) as f32 * 0.001)
        .collect::<Vec<_>>();
    let values = (0..KV_HEADS * TOKENS * WIDTH)
        .map(|index| (index as i32 % 23 - 11) as f32 * 0.003)
        .collect::<Vec<_>>();
    let q = mlx_rs::Array::from_slice(&query, &[1, QUERY_HEADS as i32, 1, WIDTH as i32]);
    let k = mlx_rs::Array::from_slice(&keys, &[1, KV_HEADS as i32, TOKENS as i32, WIDTH as i32]);
    let v = mlx_rs::Array::from_slice(&values, &[1, KV_HEADS as i32, TOKENS as i32, WIDTH as i32]);
    let scale = 1.0 / (WIDTH as f32).sqrt();
    let output = crate::primitives::attention::sdpa(
        &q,
        &k,
        &v,
        scale,
        crate::primitives::attention::AttnMask::None,
    )
    .map_err(|error| format!("dense kernel parity dispatch: {error}"))?;
    output
        .eval()
        .map_err(|error| format!("dense kernel parity evaluation: {error}"))?;
    // Stride-safe host read: a multi-row fused SDPA output is a transposed view (sc-20676).
    let actual = crate::primitives::nn::to_f32_host(&output)
        .map_err(|error| format!("dense kernel parity readback: {error}"))?;
    let groups = QUERY_HEADS / KV_HEADS;
    let mut expected = Vec::with_capacity(QUERY_HEADS * WIDTH);
    for query_head in 0..QUERY_HEADS {
        let kv_head = query_head / groups;
        let query_base = query_head * WIDTH;
        let kv_base = kv_head * TOKENS * WIDTH;
        let mut scores = Vec::with_capacity(TOKENS);
        let mut maximum = f32::NEG_INFINITY;
        for token in 0..TOKENS {
            let key_base = kv_base + token * WIDTH;
            let dot = (0..WIDTH)
                .map(|channel| query[query_base + channel] * keys[key_base + channel])
                .sum::<f32>()
                * scale;
            maximum = maximum.max(dot);
            scores.push(dot);
        }
        let denominator = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .sum::<f32>();
        for channel in 0..WIDTH {
            expected.push(
                scores
                    .iter()
                    .enumerate()
                    .map(|(token, score)| {
                        ((*score - maximum).exp() / denominator)
                            * values[kv_base + token * WIDTH + channel]
                    })
                    .sum::<f32>(),
            );
        }
    }
    let errors = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| f64::from((*actual - expected).abs()))
        .collect::<Vec<_>>();
    drop(output);
    drop((q, k, v));
    mlx_rs::memory::clear_cache();
    Ok(errors)
}

/// Convert four actual candidate/reference fixture pairs into the raw quality input consumed by
/// [`ReceiptBuilder`]. Tool validity, needle recovery, and output agreement are recorded for both
/// arms as observations; nothing here rejects a row on model behaviour. Dense rows are raw
/// characterization against the bf16 model, and compressed rows are gated by the receipt
/// validator against the dense-KV run on the same weights (quality contract v3).
#[allow(clippy::too_many_arguments)]
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
    reference: QualityReference,
    kernel_parity_errors: Option<&[f64]>,
    teacher_forced_candidate: Option<&TeacherForcedChoices>,
    forced_continuation: Option<&ReceiptForcedContinuation>,
    teacher_forced_cache: Option<&TeacherForcedChoices>,
    multi_turn_forced_continuation: Option<&ReceiptForcedContinuation>,
    multi_turn_forced_pass: Option<&MultiTurnForcedPass>,
    stream_likelihood: Option<&StreamLikelihood>,
) -> Result<QualityObservation, String> {
    if reference == QualityReference::DenseKvSameWeights
        && [
            (kernel_candidate, kernel_reference),
            (tool_candidate, tool_reference),
            (needle_candidate, needle_reference),
            (cache_candidate, cache_reference),
        ]
        .iter()
        .any(|(candidate, reference)| {
            candidate.quality_observation.snapshot.sha256
                != reference.quality_observation.snapshot.sha256
        })
    {
        return Err(
            "compressed quality must be measured against the dense-KV run on the same weights"
                .into(),
        );
    }
    let candidate_stream = stream_tokens(&kernel_candidate.quality_observation.token_probabilities);
    let reference_stream = stream_tokens(&kernel_reference.quality_observation.token_probabilities);
    // Greedy agreement is teacher-forced: the candidate's own greedy choice at every position of
    // the reference stream. A free-running cascade after one argmax flip is recorded separately.
    let (greedy_matches, greedy_total) = match (forced_continuation, teacher_forced_candidate) {
        (Some(continuation), _) => (continuation.matches, continuation.tokens),
        (None, Some(forced)) => {
            teacher_forced_agreement(&forced.choices, &reference_stream, &forced.stop_tokens)
        }
        (None, None) => token_agreement(
            &kernel_candidate.quality_observation.token_probabilities,
            &kernel_reference.quality_observation.token_probabilities,
        ),
    };
    // Perplexity is scored like greedy agreement: both arms' likelihood of the reference stream,
    // the candidate teacher-forced on it. Free-running streams diverge after one argmax flip and
    // would then score two different texts.
    let (reference_perplexity, candidate_perplexity) = match stream_likelihood {
        Some(likelihood) => (likelihood.reference, likelihood.candidate),
        None if forced_continuation.is_some() || teacher_forced_candidate.is_some() => {
            return Err(
                "a teacher-forced repeat is scored on the reference stream's likelihood".into(),
            )
        }
        // Untimed standalone suites only (no teacher-forced pass), like the free-running agreement
        // above: a receipt's quality measurement always carries its scored stream.
        None => (
            negative_log_likelihood(&kernel_reference.quality_observation.token_probabilities)?,
            negative_log_likelihood(&kernel_candidate.quality_observation.token_probabilities)?,
        ),
    };
    // Contract v4: the multi-turn fixture's quality output is turn 2, served in both arms by a
    // prompt-cache hit over turn 1 (fail closed otherwise), and its agreement is teacher-forced
    // like greedy agreement: on the row's turn-2 forced continuation (compressed rows) or on the
    // reference's natural turn-2 stream (dense rows). Free-running agreement is only observed.
    let turns = |result: &ProductFixtureResult, arm: &str| {
        let turns = result.multi_turn.clone().ok_or_else(|| {
            format!("{arm} multi-turn prompt-cache fixture has no per-turn cache record")
        })?;
        turns.validate(arm)?;
        Ok::<_, String>(turns)
    };
    let cache_candidate_turns = turns(cache_candidate, "candidate")?;
    let cache_reference_turns = turns(cache_reference, "reference")?;
    // Same weights, dense turn 1 in both arms: a compressed row's two arms must render the same
    // turn-2 prompt, or their turn-2 outputs are not comparable.
    if reference == QualityReference::DenseKvSameWeights
        && cache_candidate_turns.turn2.prompt_sha256 != cache_reference_turns.turn2.prompt_sha256
    {
        return Err("the compressed and same-weights dense-KV turn-2 prompts differ".into());
    }
    if let Some(pass) = multi_turn_forced_pass {
        pass.validate()?;
    }
    if multi_turn_forced_continuation.is_some() != multi_turn_forced_pass.is_some() {
        return Err(
            "a turn-2 forced continuation is sealed with both sessions' turn records".into(),
        );
    }
    let cache_candidate_stream =
        stream_tokens(&cache_candidate.quality_observation.token_probabilities);
    let cache_reference_stream =
        stream_tokens(&cache_reference.quality_observation.token_probabilities);
    let (cache_matches, cache_total) = match (multi_turn_forced_continuation, teacher_forced_cache)
    {
        (Some(continuation), _) => (continuation.matches, continuation.tokens),
        (None, Some(forced)) => teacher_forced_agreement(
            &forced.choices,
            &cache_reference_stream,
            &forced.stop_tokens,
        ),
        // Untimed standalone suites only: a receipt's quality measurement requires its
        // teacher-forced turn-2 pass.
        (None, None) => token_agreement(
            &cache_candidate.quality_observation.token_probabilities,
            &cache_reference.quality_observation.token_probabilities,
        ),
    };
    let outcomes = FixtureOutcomes {
        candidate_tool_valid: structured_fixture_tool_valid(&tool_candidate.output),
        reference_tool_valid: structured_fixture_tool_valid(&tool_reference.output),
        tool_outputs_match: tool_candidate.output.tool_calls == tool_reference.output.tool_calls,
        candidate_needle_recovered: needle_candidate.output.text.contains(expected_needle),
        reference_needle_recovered: needle_reference.output.text.contains(expected_needle),
        needle_outputs_match: needle_candidate.output.text == needle_reference.output.text,
    };
    let (needle_recovered, needle_discriminating) = needle_observation(&outcomes, reference);
    // Tool agreement is exact tool-call equality with the reference. It discriminates only when
    // the same-weights dense-KV run (a dense row's own candidate) emitted the valid call.
    let tool_discriminating = match reference {
        QualityReference::Bf16Characterization => outcomes.candidate_tool_valid,
        QualityReference::DenseKvSameWeights => outcomes.reference_tool_valid,
    };
    // A compressed arm supplies its own fused-reader parity; dense rows measure the dense kernel.
    let parity_errors = match kernel_parity_errors {
        Some(errors) if !errors.is_empty() => errors.to_vec(),
        Some(_) => return Err("compressed kernel parity produced no errors to reduce".into()),
        None => dense_kernel_fp32_parity_errors()?,
    };
    Ok(QualityObservation {
        parity_errors,
        reference_perplexity,
        candidate_perplexity,
        greedy_matches,
        greedy_total,
        tool_matches: u64::from(outcomes.tool_outputs_match),
        tool_total: 1,
        needle_matches: u64::from(needle_recovered),
        needle_total: 1,
        cache_matches,
        cache_total,
        outcomes,
        needle_discriminating,
        tool_discriminating,
        free_running_first_divergence: first_divergence(&candidate_stream, &reference_stream),
        greedy_agreement_by_repeat: Vec::new(),
        repeat_metrics: Vec::new(),
        forced_continuation: forced_continuation.cloned(),
        cache_free_running_first_divergence: first_divergence(
            &cache_candidate_stream,
            &cache_reference_stream,
        ),
        cache_matched_prefix_tokens: matched_prefix_tokens(
            &cache_candidate_stream,
            &cache_reference_stream,
        ),
        cache_candidate_turns,
        cache_reference_turns,
        multi_turn_forced_continuation: multi_turn_forced_continuation.cloned(),
        multi_turn_forced_pass: multi_turn_forced_pass.cloned(),
    })
}

fn stream_tokens(probabilities: &[(i32, f64)]) -> Vec<i32> {
    probabilities.iter().map(|(token, _)| *token).collect()
}

/// Teacher-forced agreement: at each position of the `reference` stream, whether the candidate's
/// greedy choice (given the reference prefix) is the reference token. The reference stream keeps
/// its stop token, so its stop position counts: there the candidate agrees when it also chooses a
/// stop token. A position the candidate never reached counts as a mismatch; the denominator is
/// always the reference length.
pub fn teacher_forced_agreement(
    candidate_choices: &[i32],
    reference: &[i32],
    stop_tokens: &[i32],
) -> (u64, u64) {
    let matches = candidate_choices
        .iter()
        .zip(reference)
        .filter(|(candidate, reference)| {
            candidate == reference
                || (stop_tokens.contains(reference) && stop_tokens.contains(candidate))
        })
        .count() as u64;
    (matches, reference.len() as u64)
}

/// First position at which two free-running streams differ; a strict prefix diverges at its end.
pub fn first_divergence(candidate: &[i32], reference: &[i32]) -> Option<u64> {
    candidate
        .iter()
        .zip(reference)
        .position(|(candidate, reference)| candidate != reference)
        .or((candidate.len() != reference.len()).then(|| candidate.len().min(reference.len())))
        .map(|position| position as u64)
}

/// Needle outcome and whether it can discriminate KV-induced retrieval loss.
///
/// A dense row's dense-KV run on its own weights is the candidate itself, so its observation is
/// the candidate's exact recovery. A compressed row is compared with the same-weights dense-KV run:
/// when that run recovered the needle the compressed run must recover it too; when it did not, the
/// only measurable property is exact agreement with the dense output, and the row is flagged
/// non-discriminating instead of silently counting a shared miss as retrieval.
fn needle_observation(outcomes: &FixtureOutcomes, reference: QualityReference) -> (bool, bool) {
    match reference {
        QualityReference::Bf16Characterization => (
            outcomes.candidate_needle_recovered,
            outcomes.candidate_needle_recovered,
        ),
        QualityReference::DenseKvSameWeights if outcomes.reference_needle_recovered => {
            (outcomes.candidate_needle_recovered, true)
        }
        // Contract v5: the dense run missed, so the candidate's own recovery is recorded as an
        // observation (both outputs stay in the artifact) and the gate never evaluates it.
        QualityReference::DenseKvSameWeights => (outcomes.candidate_needle_recovered, false),
    }
}

fn structured_fixture_tool_valid(output: &TextLlmOutput) -> bool {
    matches!(output.tool_calls.as_slice(), [call]
        if call.name == "record_baseline_fact"
            && call.arguments.get("fact").and_then(serde_json::Value::as_str)
                == Some("SC20671 structured fixture"))
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
    pub context_window_tokens: u64,
    pub context_target_tokens: u64,
    pub context_payload_tokens: u64,
    /// The fixture denominator: bf16 characterization for dense rows, the same-weights dense-KV
    /// run for compressed rows.
    pub reference: QualityReference,
    /// Compressed rows: the fused reader's parity against its host-fp32 dequantize-then-attend
    /// reference. `None` measures the dense kernel.
    pub kernel_parity_errors: Option<Vec<f64>>,
    /// The candidate's greedy choice at every position of this repeat's reference kernel stream
    /// (teacher-forced). Required for a dense row's measured repeats; warmups do not carry it.
    pub teacher_forced_candidate: Option<TeacherForcedChoices>,
    /// Compressed rows' measured repeats: the row's forced continuation, which replaces the
    /// natural-stream teacher-forced agreement.
    pub forced_continuation: Option<ReceiptForcedContinuation>,
    /// Dense rows' measured repeats: the candidate's greedy choice at every position of this
    /// repeat's reference turn-2 stream, after its own turn-2 prompt-cache hit.
    pub teacher_forced_cache: Option<TeacherForcedChoices>,
    /// Compressed rows' measured repeats: the row's turn-2 forced continuation (contract v4).
    pub multi_turn_forced_continuation: Option<ReceiptForcedContinuation>,
    /// Compressed rows' measured repeats: both sessions' turn records of that continuation.
    pub multi_turn_forced_pass: Option<MultiTurnForcedPass>,
    /// Measured repeats: both arms' likelihood of the reference stream the candidate was
    /// teacher-forced on (the row's dense forced continuation, or a dense row's reference kernel
    /// stream) — the perplexity-delta inputs.
    pub stream_likelihood: Option<StreamLikelihood>,
}

const FIXTURE_MAX_NEW_TOKENS: u32 = 64;

fn fixture_request(prompt: String, tools: Vec<ToolSpec>) -> TextLlmRequest {
    TextLlmRequest {
        messages: vec![Message::text(Role::User, prompt)],
        tools,
        // The frozen fixture measures bounded answer/tool behavior, not unbounded reasoning.
        // Qwen3 defaults to thinking and can consume the entire 64-token fixture budget before it
        // reaches the answer or tool-call block; disabling it keeps candidate and reference on the
        // same deterministic product path. Providers without thinking support ignore Disabled.
        thinking: ThinkingMode::Disabled,
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        },
        max_new_tokens: FIXTURE_MAX_NEW_TOKENS,
        seed: Some(0),
        ..Default::default()
    }
}

fn needle_fixture_prompt(prompt: &str, band_payload: &str, needle: &str) -> String {
    format!(
        "{prompt}\n{NEEDLE_FIXTURE_STATEMENT_PREFIX}{needle}.\n\
         BEGIN LONG CONTEXT\n{band_payload}\nEND LONG CONTEXT\n{NEEDLE_FIXTURE_QUESTION}"
    )
}

pub fn run_product_fixture_suite(
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureSuite> {
    let candidate = CampaignSession::load(candidate_snapshot)?;
    let candidate_baseline = candidate.load_start_sample.mlx_active_bytes;
    let candidate_half = run_product_fixture_half_on_session(&candidate, prompt, coordinate)?;
    drop(candidate);
    quiesce_campaign_active_memory(candidate_baseline)?;

    let reference = CampaignSession::load(reference_snapshot)?;
    let reference_baseline = reference.load_start_sample.mlx_active_bytes;
    let reference_half = run_product_fixture_half_on_session(&reference, prompt, coordinate)?;
    drop(reference);
    quiesce_campaign_active_memory(reference_baseline)?;
    pair_product_fixture_halves(
        candidate_half,
        reference_half,
        QualityReference::Bf16Characterization,
    )
}

struct ProductFixtureHalf {
    kernel: ProductFixtureResult,
    tool: ProductFixtureResult,
    needle_result: ProductFixtureResult,
    cache: ProductFixtureResult,
    needle: String,
    context_window_tokens: u64,
    context_target_tokens: u64,
    context_payload_tokens: u64,
    context_payload_sha256: String,
    fixture_prompt_tokens: [u64; 4],
    fixture_prompt_sha256: [String; 4],
}

/// A bounded, explicitly unaccepted worker-log record. It survives quality rejection but is never
/// read by the resume validator or published as a receipt. Every record binds its role and fixture
/// observations to the sealed launch identity and the model inventory actually loaded by the child.
fn fixture_half_attempt_diagnostic(
    half: &ProductFixtureHalf,
    coordinate: &Coordinate,
    resume_identity_sha256: &str,
    role: &str,
    run_kind: &str,
    index: usize,
) -> String {
    const OUTPUT_CHARS: usize = 1024;
    let bounded_text = |text: &str| {
        let prefix = text.chars().take(OUTPUT_CHARS).collect::<String>();
        serde_json::json!({
            "prefix": prefix,
            "bytes": text.len(),
            "sha256": seal_bytes(text.as_bytes()),
            "truncated": prefix.len() < text.len(),
        })
    };
    let observation = |observed: &ProductObservations| {
        serde_json::json!({
            "inventorySha256": observed.snapshot.sha256,
            "sessionId": observed.session_id,
            "geometry": {
                "queryHeads": observed.geometry.query_heads,
                "kvHeads": observed.geometry.kv_heads,
                "headDimension": observed.geometry.head_dimension,
                "layers": observed.geometry.layers,
                "elementBytes": observed.geometry.element_bytes,
                "kvLength": observed.cache_live_tokens,
                "capacity": observed.cache_capacity_tokens,
            },
            "phasePeaks": observed.phases.iter().map(|phase| serde_json::json!({
                "phase": phase.phase,
                "physFootprintBytes": phase.phys_footprint_bytes,
                "physFootprintPeakBytes": phase.phys_footprint_peak_bytes,
                "mlxActiveBytes": phase.mlx.active_bytes,
                "mlxCacheBytes": phase.mlx.cache_bytes,
                "mlxPeakBytes": phase.mlx.peak_bytes,
            })).collect::<Vec<_>>(),
        })
    };
    let fixtures = [
        ("kernel-fp32-reference", &half.kernel),
        ("structured-tool-call", &half.tool),
        ("long-context-needle", &half.needle_result),
        ("multi-turn-prompt-cache", &half.cache),
    ];
    let fixtures = fixtures
        .iter()
        .enumerate()
        .map(|(fixture_index, (name, result))| {
            let output = &result.output;
            (
                name.to_string(),
                serde_json::json!({
                    "promptSha256": half.fixture_prompt_sha256[fixture_index],
                    "promptTokens": half.fixture_prompt_tokens[fixture_index],
                    "maxNewTokens": FIXTURE_MAX_NEW_TOKENS,
                    "output": {
                        "text": bounded_text(&output.text),
                        "thinking": output.thinking.as_deref().map(&bounded_text),
                        "toolCalls": output.tool_calls.iter().map(|call| serde_json::json!({
                            "name": bounded_text(&call.name),
                            "arguments": bounded_text(&serde_json::Value::Object(call.arguments.clone()).to_string()),
                        })).collect::<Vec<_>>(),
                        "promptTokens": output.usage.prompt_tokens,
                        "generatedTokens": output.usage.generated_tokens,
                        "finishReason": format!("{:?}", output.finish_reason),
                        "sha256": product_output_digest(output),
                    },
                    "coordinateOperation": result.coordinate_operation,
                    "coordinateOutputSha256": result.coordinate_output_sha256,
                    "multiTurn": result.multi_turn,
                    "coordinateObservation": observation(&result.observation),
                    "qualityObservation": observation(&result.quality_observation),
                }),
            )
        })
        .collect::<serde_json::Map<String, serde_json::Value>>();
    let record = serde_json::json!({
        "schemaVersion": 1,
        "kind": "sc-20671-unaccepted-attempt-diagnostic",
        "resumeIdentitySha256": resume_identity_sha256,
        "coordinate": coordinate_slug(coordinate),
        "role": role,
        "runKind": run_kind,
        "index": index,
        "structuredToolValid": structured_fixture_tool_valid(&half.tool.output),
        "needleRecovered": half.needle_result.output.text.contains(&half.needle),
        "fixtures": fixtures,
    });
    format!("SC20671_DIAGNOSTIC_UNACCEPTED {record}")
}

fn run_product_fixture_half_on_session(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureHalf> {
    run_product_fixture_half_on_session_bounded(session, prompt, coordinate, u64::MAX, None)
}

/// The kernel fixture prompt of `coordinate`'s context band, exactly as the fixture half builds it.
fn kernel_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (band_payload, ..) = session
        .provider
        .campaign_context_band_measurement(coordinate.context_band)?;
    Ok(format!(
        "{prompt}\n{band_payload}\nReturn a concise deterministic answer."
    ))
}

/// The multi-turn fixture's turn-1 prompt of `coordinate`'s context band, exactly as the fixture
/// half builds it.
fn cache_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (band_payload, ..) = session
        .provider
        .campaign_context_band_measurement(coordinate.context_band)?;
    Ok(format!(
        "{prompt}\n{band_payload}\nRepeat the stable baseline fact."
    ))
}

/// The multi-turn fixture's turn-1 prompt (contract v4): the cache-fixture framing over the
/// band's multi-turn payload ([`multi_turn_payload_target`]).
fn multi_turn_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (payload, ..) = session
        .provider
        .campaign_multi_turn_payload(coordinate.context_band)?;
    Ok(format!(
        "{prompt}\n{payload}\nRepeat the stable baseline fact."
    ))
}

/// Teacher-forced multi-turn fixture (dense rows): the candidate session runs both turns on the
/// product prompt cache and decodes turn 2 forced on `forced` (the reference's natural turn-2
/// stream); its greedy choice at every position is returned.
fn teacher_forced_multi_turn_choices(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
    forced: &[i32],
) -> core_llm::Result<Vec<i32>> {
    let turn1_prompt = multi_turn_fixture_prompt(session, prompt, coordinate)?;
    if forced.is_empty() {
        return Err(core_llm::Error::InvalidRequest(
            "reference turn-2 stream is empty; nothing to teacher-force".into(),
        ));
    }
    let mut observer = ProductObserver::teacher_forced(forced.to_vec());
    run_lifecycle_request_on_session(
        session,
        &turn1_prompt,
        fixture_request(turn1_prompt.clone(), Vec::new()),
        Some(MULTI_TURN_FIXTURE_FOLLOW_UP),
        None,
        &mut observer,
    )?;
    let observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    Ok(stream_tokens(&observation.token_probabilities))
}

/// Teacher-forced kernel fixture: the candidate session decodes `reference`'s own kernel token
/// stream, and at every position the candidate's greedy choice and its probability of the forced
/// token are returned.
fn teacher_forced_kernel_choices(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
    forced: &[i32],
) -> core_llm::Result<ScoredContinuation> {
    let kernel_prompt = kernel_fixture_prompt(session, prompt, coordinate)?;
    if forced.is_empty() {
        return Err(core_llm::Error::InvalidRequest(
            "reference kernel stream is empty; nothing to teacher-force".into(),
        ));
    }
    let mut observer = ProductObserver::teacher_forced(forced.to_vec());
    run_dense_lifecycle_request_on_session(
        session,
        &kernel_prompt,
        fixture_request(kernel_prompt.clone(), Vec::new()),
        None,
        &mut observer,
    )?;
    let observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    if session.compressed().is_some() {
        forced_pass_stayed_compressed(&observation.packed_evidence, &observation.dense_fallbacks)
            .map_err(core_llm::Error::Load)?;
    }
    Ok(ScoredContinuation {
        choices: stream_tokens(&observation.token_probabilities),
        stream_probabilities: observation.forced_token_probabilities,
    })
}

/// Run `force` once per distinct stream and return its result for every stream, in order.
pub fn force_distinct_streams<T: Clone, E>(
    streams: &[Vec<i32>],
    mut force: impl FnMut(&[i32]) -> Result<T, E>,
) -> Result<Vec<T>, E> {
    let mut forced = std::collections::BTreeMap::<&[i32], T>::new();
    streams
        .iter()
        .map(|stream| {
            if let Some(choices) = forced.get(stream.as_slice()) {
                return Ok(choices.clone());
            }
            let choices = force(stream)?;
            forced.insert(stream, choices.clone());
            Ok(choices)
        })
        .collect()
}

/// A compressed row's teacher-forced pass must decode wholly on the fused compressed reader, like
/// its steady decode: a pass that fell back to dense would measure dense agreement under a
/// compressed label. Refused (fail closed) otherwise.
pub fn forced_pass_stayed_compressed(
    evidence: &[crate::primitives::PackedCacheEvidence],
    dense_fallbacks: &[(String, String)],
) -> Result<(), String> {
    let fused = evidence
        .first()
        .is_some_and(|first| first.accepted_direct_calls > 0)
        && evidence.iter().all(|cache| {
            cache.fallback_reasons.is_empty()
                && !cache.dense_active
                && cache.full_cache_dequantizations == 0
                && cache.failed_dispatches == 0
        })
        && dense_fallbacks.is_empty();
    if !fused {
        return Err(
            "teacher-forced pass did not decode wholly on the fused compressed reader; the row is refused"
                .into(),
        );
    }
    Ok(())
}

/// A greedy pass over one token stream, scored: the session's choice at every position and its
/// probability of the stream token there (the forced token when teacher-forced, its own choice when
/// it produced the stream).
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredContinuation {
    pub choices: Vec<i32>,
    pub stream_probabilities: Vec<f64>,
}

/// Both arms' mean per-token negative log-likelihood of ONE token stream — the reference's greedy
/// stream, which the reference produced and the candidate is teacher-forced on — the inputs of
/// `perplexityDelta`. Scoring each arm on its own free-running stream instead compares two
/// different texts once the streams diverge (after a single argmax flip), which measures the
/// continuation, not the KV representation.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamLikelihood {
    pub reference: f64,
    pub candidate: f64,
}

impl StreamLikelihood {
    /// Per-position probabilities of the same stream from each arm.
    pub fn from_probabilities(reference: &[f64], candidate: &[f64]) -> Result<Self, String> {
        if reference.len() != candidate.len() {
            return Err(format!(
                "stream likelihood needs both arms scored on one stream: {} reference and {} candidate positions",
                reference.len(),
                candidate.len()
            ));
        }
        Ok(Self {
            reference: mean_negative_log_likelihood(reference)?,
            candidate: mean_negative_log_likelihood(candidate)?,
        })
    }
}

/// The candidate's greedy choice at every position of a reference stream, and the stop tokens the
/// generation ends on (any stop token agrees with a reference stop token).
#[derive(Clone, Debug, PartialEq)]
pub struct TeacherForcedChoices {
    pub choices: Vec<i32>,
    pub stop_tokens: Vec<i32>,
}

/// One arm's quality measurement (contract v5: once per arm per process): the four frozen
/// fixtures, each with its coordinate operation. `kernel_coordinate`, when given, is this
/// session's already measured coordinate operation over the kernel fixture prompt (the candidate's
/// timing repeat 0) and is reused as the kernel fixture's coordinate evidence instead of being
/// dispatched again: the same prompt, operation and session, so the evidence is the same
/// measurement.
fn run_product_fixture_half_on_session_bounded(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
    max_request_tokens: u64,
    kernel_coordinate: Option<CoordinateRun>,
) -> core_llm::Result<ProductFixtureHalf> {
    session.validate_coordinate_family(coordinate)?;
    let context_window_tokens = session.provider.campaign_context_window()?;
    let (band_payload, context_target_tokens, context_payload_tokens) = session
        .provider
        .campaign_context_band_measurement(coordinate.context_band)?;
    let needle = "SC20671-NUMERIC-NEEDLE-9b7a2e".to_string();
    let kernel_prompt = kernel_fixture_prompt(session, prompt, coordinate)?;
    let tool_prompt = format!(
        "{prompt}\n{band_payload}\nCall record_baseline_fact with fact exactly `SC20671 structured fixture`."
    );
    let needle_prompt = needle_fixture_prompt(prompt, &band_payload, &needle);
    let cache_prompt = cache_fixture_prompt(session, prompt, coordinate)?;
    let fixture_prompt_tokens = [
        session.provider.campaign_prompt_tokens(&kernel_prompt)?,
        session.provider.campaign_prompt_tokens(&tool_prompt)?,
        session.provider.campaign_prompt_tokens(&needle_prompt)?,
        session.provider.campaign_prompt_tokens(&cache_prompt)?,
    ];
    let fixture_prompt_sha256 = [
        seal_bytes(kernel_prompt.as_bytes()),
        seal_bytes(tool_prompt.as_bytes()),
        seal_bytes(needle_prompt.as_bytes()),
        seal_bytes(cache_prompt.as_bytes()),
    ];
    for tokens in fixture_prompt_tokens {
        if tokens > max_request_tokens {
            return Err(core_llm::Error::InvalidRequest(format!(
                "fixture raw prompt has {tokens} tokens, exceeding explicit safety ceiling {max_request_tokens}"
            )));
        }
        if tokens > context_window_tokens.saturating_sub(256) {
            return Err(core_llm::Error::InvalidRequest(format!(
                "fixture prompt tokenization exceeds the loaded context: {tokens}/{context_window_tokens}"
            )));
        }
    }
    let kernel_request = fixture_request(kernel_prompt.clone(), Vec::new());
    let kernel_coordinate = match kernel_coordinate {
        Some(run) => {
            reusable_kernel_coordinate(&run, session.session_id(), coordinate)
                .map_err(core_llm::Error::InvalidRequest)?;
            run
        }
        None => run_coordinate_on_session(session, &kernel_prompt, &kernel_request, coordinate)?,
    };
    let kernel = run_fixture_quality_on_session(
        session,
        kernel_coordinate,
        &kernel_prompt,
        kernel_request,
        None,
    )?;
    let tool = run_product_fixture_on_session(
        session,
        &tool_prompt,
        fixture_request(tool_prompt.clone(), vec![structured_fixture_tool()]),
        coordinate,
    )?;
    let needle_result = run_product_fixture_on_session(
        session,
        &needle_prompt,
        fixture_request(needle_prompt.clone(), Vec::new()),
        coordinate,
    )?;
    let multi_turn_prompt = multi_turn_fixture_prompt(session, prompt, coordinate)?;
    let cache = run_fixture_on_session(
        session,
        &cache_prompt,
        fixture_request(cache_prompt.clone(), Vec::new()),
        Some((
            MULTI_TURN_FIXTURE_FOLLOW_UP,
            fixture_request(multi_turn_prompt, Vec::new()),
        )),
        coordinate,
    )?;
    Ok(ProductFixtureHalf {
        kernel,
        tool,
        needle_result,
        cache,
        needle,
        context_window_tokens,
        context_target_tokens,
        context_payload_tokens,
        context_payload_sha256: seal_bytes(band_payload.as_bytes()),
        fixture_prompt_tokens,
        fixture_prompt_sha256,
    })
}

/// A timing run's coordinate operation is reusable as the kernel fixture's only when it is this
/// session's measurement of the kernel fixture prompt under the coordinate's own operation(s).
fn reusable_kernel_coordinate(
    run: &CoordinateRun,
    session_id: &str,
    coordinate: &Coordinate,
) -> Result<(), String> {
    let expected_operation = if coordinate.request_mode == "supported-batch" {
        "supported-batch"
    } else if coordinate.prefill_mode == "chunked" {
        "chunked-prefix-reuse"
    } else {
        "single-shot-generation"
    };
    let needs_secondary =
        coordinate.request_mode == "supported-batch" && coordinate.prefill_mode == "chunked";
    if run.primary.observation.session_id != session_id
        || run.primary.operation != expected_operation
        || run.secondary.is_some() != needs_secondary
        || run
            .secondary
            .as_ref()
            .is_some_and(|secondary| secondary.observation.session_id != session_id)
    {
        return Err(
            "a reused kernel coordinate operation must be this session's run of the coordinate's operation"
                .into(),
        );
    }
    Ok(())
}

/// One timing measurement of the candidate arm (contract v5 `statistics.repeats` / `warmups`):
/// the coordinate operation over the kernel fixture prompt, then one fixed-length steady decode
/// of the row context. The steady decode runs after the coordinate observer has closed, on its
/// own request-scoped cache that is released before the next run, so no coordinate's memory
/// attribution sees it; its `prompt + STEADY_DECODE_TOKENS` live tokens are admitted up front by
/// the preflight live-token bound. Batch rows are timed as one sequence too: the compressed arm
/// has no batch route, so a batched steady decode could not compare the two representations at
/// the same boundary.
fn run_timing_on_session(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
    max_request_tokens: u64,
) -> core_llm::Result<TimingRun> {
    session.validate_coordinate_family(coordinate)?;
    let kernel_prompt = kernel_fixture_prompt(session, prompt, coordinate)?;
    let tokens = session.provider.campaign_prompt_tokens(&kernel_prompt)?;
    let context_window_tokens = session.provider.campaign_context_window()?;
    if tokens > max_request_tokens || tokens > context_window_tokens.saturating_sub(256) {
        return Err(core_llm::Error::InvalidRequest(format!(
            "kernel fixture prompt has {tokens} tokens, exceeding the safety ceiling {max_request_tokens} or the loaded context {context_window_tokens}"
        )));
    }
    let request = fixture_request(kernel_prompt.clone(), Vec::new());
    let coordinate_run = run_coordinate_on_session(session, &kernel_prompt, &request, coordinate)?;
    let steady_decode = session.provider.campaign_steady_decode(
        &kernel_prompt,
        STEADY_DECODE_TOKENS as usize,
        session.compressed(),
    )?;
    Ok(TimingRun {
        coordinate: coordinate_run,
        steady_decode,
    })
}

fn pair_product_fixture_halves(
    candidate: ProductFixtureHalf,
    reference: ProductFixtureHalf,
    reference_kind: QualityReference,
) -> core_llm::Result<ProductFixtureSuite> {
    if candidate.needle != reference.needle
        || candidate.context_window_tokens != reference.context_window_tokens
        || candidate.context_target_tokens != reference.context_target_tokens
        || candidate.context_payload_tokens != reference.context_payload_tokens
        || candidate.context_payload_sha256 != reference.context_payload_sha256
        || candidate.fixture_prompt_tokens != reference.fixture_prompt_tokens
    {
        return Err(core_llm::Error::InvalidRequest(
            "candidate/reference context or tokenizer evidence differs".into(),
        ));
    }
    Ok(ProductFixtureSuite {
        kernel_candidate: candidate.kernel,
        kernel_reference: reference.kernel,
        tool_candidate: candidate.tool,
        tool_reference: reference.tool,
        needle_candidate: candidate.needle_result,
        needle_reference: reference.needle_result,
        cache_candidate: candidate.cache,
        cache_reference: reference.cache,
        needle: candidate.needle,
        context_window_tokens: candidate.context_window_tokens,
        context_target_tokens: candidate.context_target_tokens,
        context_payload_tokens: candidate.context_payload_tokens,
        reference: reference_kind,
        kernel_parity_errors: None,
        teacher_forced_candidate: None,
        forced_continuation: None,
        teacher_forced_cache: None,
        multi_turn_forced_continuation: None,
        multi_turn_forced_pass: None,
        stream_likelihood: None,
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
            self.reference,
            self.kernel_parity_errors.as_deref(),
            self.teacher_forced_candidate.as_ref(),
            self.forced_continuation.as_ref(),
            self.teacher_forced_cache.as_ref(),
            self.multi_turn_forced_continuation.as_ref(),
            self.multi_turn_forced_pass.as_ref(),
            self.stream_likelihood.as_ref(),
        )
    }
}

/// Derive one timing sample from product phase boundaries plus the wall-clock snapshot/provider
/// load that created this session, and the repeat's dedicated fixed-length steady decode. Decode
/// throughput is never a phase delta: the coordinate operation's own generation is EOS- or
/// budget-terminated (a chunked coordinate decodes a single token), so its `first-token` →
/// `decode-steady` interval is a few tokens or none at all. Compile attribution is finalized
/// across the cold dispatch or two real warmups by [`timing_samples_from_product_repeats`].
pub fn timing_from_product_observation(
    observation: &ProductObservations,
    steady_decode: &SteadyDecodeMeasurement,
    process_temperature: &str,
) -> Result<RawTiming, String> {
    if observation.phase_elapsed_ms.len() != REQUIRED_PHASES.len() {
        return Err("timing requires eight product phases".into());
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
    let load_ms = observation.load_elapsed_ms;
    if !load_ms.is_finite() || load_ms <= 0.0 {
        return Err("product snapshot load duration is not positive".into());
    }
    let prefill_ms = positive_delta(2, 1)?;
    let ttft_ms = positive_delta(3, 2)?;
    let first_token_ms = positive_delta(3, 0)?;
    let throughput = steady_decode.tokens_per_second()?;
    if !matches!(process_temperature, "cold" | "warm") {
        return Err("unknown process temperature".into());
    }
    Ok(RawTiming {
        load_ms,
        prefill_ms,
        ttft_ms,
        first_token_ms,
        decode_tokens_per_second: throughput,
        steady_decode: *steady_decode,
        host_state: None,
    })
}

fn expected_coordinate_operation(matrix: &ReceiptMatrix) -> Result<&'static str, String> {
    if matrix.request_mode == "supported-batch" {
        Ok("supported-batch")
    } else if matrix.request_mode == "single" && matrix.prefill_mode == "chunked" {
        Ok("chunked-prefix-reuse")
    } else if matrix.request_mode == "single" && matrix.prefill_mode == "single-shot" {
        Ok("single-shot-generation")
    } else {
        Err("matrix does not select one compile-attribution operation".into())
    }
}

fn validate_compile_attribution(
    attribution: &ReceiptCompileAttribution,
    matrix: &ReceiptMatrix,
) -> Result<(), String> {
    let expected_source = match matrix.process_temperature.as_str() {
        "cold" => "measured-repeats",
        "warm" => "warmup-suites",
        _ => return Err("unknown process temperature".into()),
    };
    let expected_len = if expected_source == "measured-repeats" {
        5
    } else {
        2
    };
    let expected_coordinate = format!(
        "{}-{}-{}-{}-{}",
        matrix.family,
        matrix.context_band,
        matrix.request_mode,
        matrix.prefill_mode,
        matrix.process_temperature
    );
    let digest = |value: &str| {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    };
    if attribution.method != COMPILE_ATTRIBUTION_METHOD
        || attribution.operation != expected_coordinate_operation(matrix)?
        || attribution.source != expected_source
        || attribution.probe_durations_ms.len() != expected_len
        || attribution.probe_evidence.len() != expected_len
        || attribution
            .probe_durations_ms
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || attribution
            .probe_evidence
            .iter()
            .enumerate()
            .any(|(index, evidence)| {
                evidence.index != index as u64
                    || evidence.operation != attribution.operation
                    || evidence.source != attribution.source
                    || evidence.matrix_coordinate != expected_coordinate
                    || !evidence.setup_ms.is_finite()
                    || evidence.setup_ms < 0.0
                    || !evidence.dispatch_ms.is_finite()
                    || evidence.dispatch_ms <= 0.0
                    || evidence.dispatch_ms != attribution.probe_durations_ms[index]
                    || !digest(&evidence.operation_evidence_sha256)
            })
    {
        return Err("compile attribution identity or bound probe evidence is invalid".into());
    }
    let first = attribution.probe_durations_ms[0];
    let steady = if expected_source == "measured-repeats" {
        let mut later = attribution.probe_durations_ms[1..].to_vec();
        later.sort_by(f64::total_cmp);
        (later[1] + later[2]) / 2.0
    } else {
        attribution.probe_durations_ms[1]
    };
    let excess = first - steady;
    if !excess.is_finite()
        || (attribution.first_dispatch_ms - first).abs() > 1e-9
        || (attribution.steady_dispatch_ms - steady).abs() > 1e-9
        || (attribution.first_dispatch_excess_ms - excess).abs() > 1e-9
    {
        return Err(format!(
            "compile attribution does not recompute: first={first:.6}ms steady={steady:.6}ms excess={excess:.6}ms"
        ));
    }
    let samples = attribution
        .noise_samples_ms
        .as_deref()
        .ok_or("compile attribution lacks its steady noise samples")?;
    let samples_valid = if expected_source == "measured-repeats" {
        samples == &attribution.probe_durations_ms[1..]
    } else {
        samples.len() == 5
    };
    if !samples_valid
        || samples
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err("compile attribution steady noise samples are invalid".into());
    }
    let band = noise_band(samples);
    let resolved = excess > band;
    let expected_reason = if resolved {
        None
    } else if excess <= 0.0 {
        Some(COMPILE_COST_NOT_SLOWER)
    } else {
        Some(COMPILE_COST_WITHIN_NOISE)
    };
    if attribution
        .noise_band_ms
        .is_none_or(|value| (value - band).abs() > 1e-9)
        || attribution.compile_cost_resolved != Some(resolved)
        || attribution.compile_cost_ms.is_some() != resolved
        || attribution
            .compile_cost_ms
            .is_some_and(|value| (value - excess).abs() > 1e-9)
        || attribution.compile_cost_unresolved_reason.as_deref() != expected_reason
    {
        return Err(format!(
            "compile cost does not recompute from its noise band: excess={excess:.6}ms band={band:.6}ms"
        ));
    }
    Ok(())
}

/// Run-to-run spread (`max - min`) of steady-state dispatches of the same operation.
fn noise_band(samples: &[f64]) -> f64 {
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

/// `timings.coldCompileMs`: the resolved compile cost (`None` below the noise band).
fn cold_compile_alias(attribution: &ReceiptCompileAttribution) -> Option<f64> {
    attribution.compile_cost_ms
}

fn compile_cost_summary(attribution: &ReceiptCompileAttribution) -> String {
    match (
        attribution.compile_cost_ms,
        attribution.compile_cost_unresolved_reason.as_deref(),
        attribution.noise_band_ms,
    ) {
        (Some(cost), _, Some(band)) => format!("{cost} ms (noise band {band} ms)"),
        (None, Some(reason), Some(band)) => format!("unresolved: {reason} (noise band {band} ms)"),
        _ => "invalid compile attribution".into(),
    }
}

fn compile_attribution_from_probes(
    operation: &str,
    matrix: &ReceiptMatrix,
    probes: Vec<(f64, f64, String)>,
    measured_dispatch_ms: &[f64],
) -> Result<ReceiptCompileAttribution, String> {
    let source = match matrix.process_temperature.as_str() {
        "cold" => "measured-repeats",
        "warm" => "warmup-suites",
        _ => return Err("unknown process temperature".into()),
    };
    let matrix_coordinate = format!(
        "{}-{}-{}-{}-{}",
        matrix.family,
        matrix.context_band,
        matrix.request_mode,
        matrix.prefill_mode,
        matrix.process_temperature
    );
    let probe_evidence = probes
        .into_iter()
        .enumerate()
        .map(
            |(index, (setup_ms, dispatch_ms, operation_evidence_sha256))| {
                ReceiptCompileProbeEvidence {
                    index: index as u64,
                    operation: operation.into(),
                    source: source.into(),
                    matrix_coordinate: matrix_coordinate.clone(),
                    setup_ms,
                    dispatch_ms,
                    operation_evidence_sha256,
                }
            },
        )
        .collect::<Vec<_>>();
    let mut attribution = ReceiptCompileAttribution {
        method: COMPILE_ATTRIBUTION_METHOD.into(),
        operation: operation.into(),
        source: source.into(),
        probe_durations_ms: probe_evidence
            .iter()
            .map(|evidence| evidence.dispatch_ms)
            .collect(),
        probe_evidence,
        first_dispatch_ms: 0.0,
        steady_dispatch_ms: 0.0,
        first_dispatch_excess_ms: 0.0,
        noise_samples_ms: None,
        noise_band_ms: None,
        compile_cost_resolved: None,
        compile_cost_ms: None,
        compile_cost_unresolved_reason: None,
    };
    if attribution.probe_durations_ms.is_empty() {
        return Err("compile attribution has no raw probes".into());
    }
    attribution.first_dispatch_ms = attribution.probe_durations_ms[0];
    attribution.steady_dispatch_ms = if matrix.process_temperature == "cold" {
        if attribution.probe_durations_ms.len() != 5 {
            return Err("cold compile attribution requires five measured repeats".into());
        }
        let mut later = attribution.probe_durations_ms[1..].to_vec();
        later.sort_by(f64::total_cmp);
        (later[1] + later[2]) / 2.0
    } else {
        if attribution.probe_durations_ms.len() != 2 {
            return Err("warm compile attribution requires two warmup suites".into());
        }
        attribution.probe_durations_ms[1]
    };
    let excess = attribution.first_dispatch_ms - attribution.steady_dispatch_ms;
    attribution.first_dispatch_excess_ms = excess;
    // A cold row's steady samples are its post-first repeats; a warm row's are the measured
    // repeats that follow its two warmups.
    let samples = if matrix.process_temperature == "cold" {
        attribution.probe_durations_ms[1..].to_vec()
    } else {
        measured_dispatch_ms.to_vec()
    };
    let band = noise_band(&samples);
    let resolved = excess > band;
    attribution.noise_samples_ms = Some(samples);
    attribution.noise_band_ms = Some(band);
    attribution.compile_cost_resolved = Some(resolved);
    attribution.compile_cost_ms = resolved.then_some(excess);
    attribution.compile_cost_unresolved_reason = (!resolved).then(|| {
        if excess <= 0.0 {
            COMPILE_COST_NOT_SLOWER.to_string()
        } else {
            COMPILE_COST_WITHIN_NOISE.to_string()
        }
    });
    validate_compile_attribution(&attribution, matrix)?;
    Ok(attribution)
}

fn warmup_probe_suite_sha256(
    session_id: &str,
    worker_pid: u32,
    probe_evidence: &[ReceiptCompileProbeEvidence],
) -> Result<String, String> {
    let bytes = canonical_semantic_seal_bytes(&serde_json::json!({
        "sessionId": session_id,
        "workerPid": worker_pid,
        "probeEvidence": probe_evidence,
    }))?;
    Ok(seal_bytes(&bytes))
}

/// Freeze cold-versus-steady compilation attribution before receipt assembly. A cold worker uses
/// five complete product-call probes and compares the first with the median of the next four. A
/// warm worker uses its two real pre-measurement warmups. The excess is resolved as compile cost
/// only when it exceeds the spread of the steady dispatches (the four later cold probes, or the
/// five measured warm repeats); otherwise it is recorded as unresolved, never refused. Prefix-reuse probes add only the seed and
/// observed-hit product calls, excluding allocator reset and footprint instrumentation.
pub fn timing_samples_from_product_repeats(
    runs: &[&CoordinateRun],
    steady_decodes: &[&SteadyDecodeMeasurement],
    coordinate: &Coordinate,
    process_temperature: &str,
    warmups: &[&CoordinateRun],
) -> Result<ProductTimingMeasurements, String> {
    if runs.len() != TIMING_REPEATS || steady_decodes.len() != runs.len() {
        return Err("receipt requires exactly five timing repeats and steady decodes".into());
    }
    let samples = runs
        .iter()
        .zip(steady_decodes)
        .map(|(run, steady)| {
            timing_from_product_observation(&run.primary.observation, steady, process_temperature)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operation = runs[0].primary.operation.as_str();
    if runs.iter().any(|run| run.primary.operation != operation)
        || warmups.iter().any(|run| run.primary.operation != operation)
    {
        return Err("compile attribution mixed coordinate operations".into());
    }
    let probe_runs = match process_temperature {
        "cold" if warmups.is_empty() => runs,
        "warm" if warmups.len() == 2 => warmups,
        "cold" => return Err("cold coordinate must not execute warmup suites".into()),
        "warm" => return Err("warm coordinate requires exactly two product warmup suites".into()),
        _ => return Err("unknown process temperature".into()),
    };
    let probes = probe_runs
        .iter()
        .map(|run| {
            (
                run.primary.compile_setup_ms,
                run.primary.compile_dispatch_ms,
                coordinate_run_evidence_digest(run),
            )
        })
        .collect::<Vec<_>>();
    let matrix = ReceiptMatrix {
        family: coordinate.family.into(),
        context_band: coordinate.context_band.into(),
        request_mode: coordinate.request_mode.into(),
        prefill_mode: coordinate.prefill_mode.into(),
        process_temperature: coordinate.process_temperature.into(),
    };
    let measured_dispatch_ms = runs
        .iter()
        .map(|run| run.primary.compile_dispatch_ms)
        .collect::<Vec<_>>();
    let compile_attribution =
        compile_attribution_from_probes(operation, &matrix, probes, &measured_dispatch_ms)?;
    Ok(ProductTimingMeasurements {
        samples,
        compile_attribution,
    })
}

pub(crate) fn checked_git_revision(root: &Path) -> Result<String, String> {
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

/// Normalize only an authenticated GitHub origin form; callers compare the result with a frozen
/// repository identity instead of stamping a path supplied by the environment.
pub fn canonical_github_repository(remote: &str) -> Result<String, String> {
    let remote = remote.trim().trim_end_matches('/');
    let path = if let Some(path) = remote.strip_prefix("git@github.com:") {
        path
    } else if let Some(path) = remote.strip_prefix("ssh://git@github.com/") {
        path
    } else if let Some(path) = remote.strip_prefix("https://github.com/") {
        path
    } else {
        return Err(format!(
            "repository origin is not an authenticated GitHub URL: {remote}"
        ));
    };
    let path = path.strip_suffix(".git").unwrap_or(path);
    let components = path.split('/').collect::<Vec<_>>();
    if components.len() != 2
        || components.iter().any(|component| {
            component.is_empty()
                || *component == "."
                || *component == ".."
                || component
                    .chars()
                    .any(|character| matches!(character, '?' | '#' | '\\'))
        })
    {
        return Err(format!(
            "repository origin has a non-canonical GitHub path: {remote}"
        ));
    }
    Ok(format!("github.com/{}/{}", components[0], components[1]))
}

/// Read and validate a worktree's `origin` against the source-owned repository identity.
pub fn checked_repository_identity(root: &Path, expected: &str) -> Result<String, String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .map_err(|e| e.to_string())?;
    let remote = String::from_utf8(output.stdout).map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err("cannot read repository origin".into());
    }
    let canonical = canonical_github_repository(&remote)?;
    if canonical != expected {
        return Err(format!(
            "repository origin is {canonical}, expected {expected}"
        ));
    }
    Ok(canonical)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockedMlxIdentity {
    pub version: String,
    pub source: String,
    pub revision: String,
}

pub fn locked_mlx_identity(lock: &[u8]) -> Result<LockedMlxIdentity, String> {
    let text = std::str::from_utf8(lock).map_err(|e| e.to_string())?;
    let quoted = |block: &str, key: &str| {
        block.lines().find_map(|line| {
            line.trim()
                .strip_prefix(&format!("{key} = \""))
                .and_then(|value| value.strip_suffix('\"'))
                .map(str::to_owned)
        })
    };
    for block in text.split("[[package]]") {
        if quoted(block, "name").as_deref() != Some("pmetal-mlx-rs") {
            continue;
        }
        let version = quoted(block, "version").ok_or("pmetal-mlx-rs lock entry has no version")?;
        let source = quoted(block, "source").ok_or("pmetal-mlx-rs lock entry has no source")?;
        let revision = source
            .rsplit_once('#')
            .map(|(_, revision)| revision.to_owned())
            .ok_or("pmetal-mlx-rs lock source has no immutable revision")?;
        let identity = LockedMlxIdentity {
            version,
            source,
            revision,
        };
        if !valid_locked_mlx_fields(&identity.version, &identity.source, &identity.revision) {
            return Err("pmetal-mlx-rs lock identity is not the frozen Git source/revision".into());
        }
        return Ok(identity);
    }
    Err("Cargo.lock does not contain pmetal-mlx-rs identity".into())
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

/// `CPU_Speed_Limit` from `pmset -g therm` when it reports CPU power status (`None` when it
/// prints only its no-history notes). The rest of the text is recorded verbatim, never parsed, so
/// unknown note lines are tolerated; only a malformed or contradictory limit is refused.
pub fn pmset_cpu_speed_limit(value: &str) -> Result<Option<u64>, String> {
    let mut limit = None;
    for line in value.lines() {
        let Some((name, raw)) = line.split_once('=') else {
            continue;
        };
        if !name.trim().eq_ignore_ascii_case("CPU_Speed_Limit") {
            continue;
        }
        let parsed = raw
            .trim()
            .parse::<u64>()
            .map_err(|_| format!("pmset CPU_Speed_Limit is not a number: {:?}", raw.trim()))?;
        if limit.is_some_and(|prior| prior != parsed) {
            return Err("pmset reports contradictory CPU_Speed_Limit values".into());
        }
        limit = Some(parsed);
    }
    Ok(limit)
}

/// Row boundaries at which the host's power mode and thermal state are observed, in order.
pub const HOST_STATE_BOUNDARIES: [&str; 2] = ["row-start", "row-end"];
/// The host state recorded beside each measured timing sample, so throughput drift is attributable.
pub const TIMING_SAMPLE_HOST_BOUNDARY: &str = "timing-sample";
/// Normalized macOS energy modes (`pmset` `powermode` 0/1/2, or the older `lowpowermode`).
pub const POWER_MODES: [&str; 3] = ["automatic", "low-power", "high-power"];
/// `NSProcessInfo.thermalState` 0..=3.
pub const THERMAL_STATES: [&str; 4] = ["nominal", "fair", "serious", "critical"];

/// A real throttle signal: `NSProcessInfo.thermalState` serious or critical, or a `pmset`
/// `CPU_Speed_Limit` below 100%.
pub fn host_state_throttled(thermal_state: &str, cpu_speed_limit: Option<u64>) -> bool {
    matches!(thermal_state, "serious" | "critical")
        || cpu_speed_limit.is_some_and(|limit| limit < 100)
}

/// The host's power mode and thermal state at one boundary. Only a throttled row start refuses
/// the row; later states are recorded and flagged on the provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptHostState {
    pub boundary: String,
    pub captured_at: String,
    pub power_mode: String,
    /// `NSProcessInfo.thermalState`, one of [`THERMAL_STATES`].
    pub thermal_state: String,
    /// `pmset -g therm` `CPU_Speed_Limit` (percent), when reported.
    pub cpu_speed_limit: Option<u64>,
    /// The raw `pmset -g therm` text, recorded verbatim.
    pub pmset_thermal_raw: String,
    /// [`host_state_throttled`] of this state.
    pub throttled: bool,
}

impl ReceiptHostState {
    fn validate(&self, boundary: &str) -> Result<(), String> {
        if self.boundary != boundary
            || !valid_utc_timestamp(&self.captured_at)
            || !POWER_MODES.contains(&self.power_mode.as_str())
            || !THERMAL_STATES.contains(&self.thermal_state.as_str())
            || self.pmset_thermal_raw.trim().is_empty()
            || pmset_cpu_speed_limit(&self.pmset_thermal_raw)? != self.cpu_speed_limit
            || self.throttled != host_state_throttled(&self.thermal_state, self.cpu_speed_limit)
        {
            return Err(format!(
                "host power/thermal provenance: {boundary} state is malformed or does not recompute"
            ));
        }
        Ok(())
    }
}

/// Whether the thermal state or the power mode changed from `start` across `later` observations.
/// A throttled later observation counts as a thermal change.
pub fn host_state_changes<'a>(
    start: &ReceiptHostState,
    later: impl IntoIterator<Item = &'a ReceiptHostState>,
) -> (bool, bool) {
    later
        .into_iter()
        .fold((false, false), |(thermal, power), state| {
            (
                thermal || state.thermal_state != start.thermal_state || state.throttled,
                power || state.power_mode != start.power_mode,
            )
        })
}

/// Refuse a row whose start is thermally throttled; every other state is recorded.
pub fn refuse_throttled_row_start(state: &ReceiptHostState) -> Result<(), String> {
    if state.throttled {
        return Err(format!(
            "row-start host is thermally throttled (thermalState={}, CPU_Speed_Limit={:?}); the row is refused before it runs",
            state.thermal_state, state.cpu_speed_limit
        ));
    }
    Ok(())
}

/// Normalize the active energy mode from `pmset -g` (its "Currently in use" settings): `powermode`
/// 0/1/2 is Automatic/Low Power/High Power; hosts without it report `lowpowermode` (and, on some
/// releases, `highpowermode`) 0/1. A missing, unknown, or contradictory setting is refused.
pub fn normalize_pmset_power_mode(value: &str) -> Result<String, String> {
    let mut in_use = false;
    let mut settings = std::collections::BTreeMap::new();
    for line in value.lines() {
        let line = line.trim();
        if line.eq_ignore_ascii_case("currently in use:") {
            in_use = true;
            continue;
        }
        if !in_use {
            continue;
        }
        let mut fields = line.split_whitespace();
        if let (Some(key @ ("powermode" | "lowpowermode" | "highpowermode")), Some(setting)) =
            (fields.next(), fields.next())
        {
            if settings.insert(key, setting).is_some() {
                return Err(format!("pmset reports {key} twice"));
            }
        }
    }
    let flag = |key: &str| match settings.get(key).copied() {
        None => Ok(None),
        Some("0") => Ok(Some(false)),
        Some("1") => Ok(Some(true)),
        Some(other) => Err(format!("pmset reports unknown {key} {other}")),
    };
    let (low, high) = (flag("lowpowermode")?, flag("highpowermode")?);
    let mode = match settings.get("powermode").copied() {
        Some("0") => "automatic",
        Some("1") => "low-power",
        Some("2") => "high-power",
        Some(other) => return Err(format!("pmset reports unknown powermode {other}")),
        None => match (low, high) {
            (Some(true), Some(true)) => {
                return Err("pmset reports both low and high power mode".into())
            }
            (Some(true), _) => "low-power",
            (_, Some(true)) => "high-power",
            (Some(false), _) | (_, Some(false)) => "automatic",
            (None, None) => {
                return Err("pmset does not report the active power mode".into());
            }
        },
    };
    if (low == Some(true) && mode != "low-power") || (high == Some(true) && mode != "high-power") {
        return Err("pmset power mode settings contradict each other".into());
    }
    Ok(mode.into())
}

/// `NSProcessInfo.thermalState` (0 nominal, 1 fair, 2 serious, 3 critical) as its name.
pub fn normalize_process_thermal_state(state: i64) -> Result<String, String> {
    usize::try_from(state)
        .ok()
        .and_then(|index| THERMAL_STATES.get(index))
        .map(|name| (*name).to_string())
        .ok_or_else(|| format!("unknown process thermal state {state}"))
}

/// One boundary's host state from its raw probes: `pmset -g`, `pmset -g therm`, and
/// `NSProcessInfo.thermalState`. Throttling is recorded here, never refused.
pub fn host_state_from_probes(
    boundary: &str,
    captured_at: String,
    pmset: &str,
    pmset_thermal: &str,
    process_thermal_state: i64,
) -> Result<ReceiptHostState, String> {
    if !HOST_STATE_BOUNDARIES.contains(&boundary) && boundary != TIMING_SAMPLE_HOST_BOUNDARY {
        return Err(format!("unknown host-state boundary {boundary}"));
    }
    let power_mode = normalize_pmset_power_mode(pmset)?;
    let thermal_state = normalize_process_thermal_state(process_thermal_state)?;
    let cpu_speed_limit = pmset_cpu_speed_limit(pmset_thermal)?;
    Ok(ReceiptHostState {
        boundary: boundary.into(),
        captured_at,
        power_mode,
        throttled: host_state_throttled(&thermal_state, cpu_speed_limit),
        thermal_state,
        cpu_speed_limit,
        pmset_thermal_raw: pmset_thermal.into(),
    })
}

/// Probe this host's power mode and thermal state at `boundary`.
pub fn capture_host_state(boundary: &str) -> Result<ReceiptHostState, String> {
    let pmset = probed_command("pmset", &["-g"], "power mode")?;
    let pmset_thermal = probed_command("pmset", &["-g", "therm"], "thermal state")?;
    let process_thermal = process_thermal_state()?;
    host_state_from_probes(
        boundary,
        timestamp_now(),
        &pmset,
        &pmset_thermal,
        process_thermal,
    )
    .map_err(|e| format!("{boundary} host state: {e}"))
}

/// `[[NSProcessInfo processInfo] thermalState]` through the Objective-C runtime.
#[cfg(target_os = "macos")]
fn process_thermal_state() -> Result<i64, String> {
    use std::ffi::{c_char, c_void};
    type Id = *mut c_void;
    type Sel = *mut c_void;
    #[link(name = "Foundation", kind = "framework")]
    extern "C" {}
    #[link(name = "objc")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
    }
    // SAFETY: the class and selector names are C string literals; `objc_msgSend` is called
    // through the exact signatures of `+[NSProcessInfo processInfo]` (returns an object) and
    // `-[NSProcessInfo thermalState]` (returns an `NSInteger`), as the arm64/x86_64 ABIs require.
    unsafe {
        let class = objc_getClass(c"NSProcessInfo".as_ptr());
        if class.is_null() {
            return Err("NSProcessInfo is unavailable".into());
        }
        let send_object = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(Id, Sel) -> Id,
        >(objc_msgSend);
        let info = send_object(class, sel_registerName(c"processInfo".as_ptr()));
        if info.is_null() {
            return Err("NSProcessInfo returned no process info".into());
        }
        let send_integer = std::mem::transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(Id, Sel) -> isize,
        >(objc_msgSend);
        Ok(send_integer(info, sel_registerName(c"thermalState".as_ptr())) as i64)
    }
}

#[cfg(not(target_os = "macos"))]
fn process_thermal_state() -> Result<i64, String> {
    Err("the process thermal state probe requires macOS".into())
}

/// The receipt quality of the row's quality measurement (contract v5: exactly one per arm, since
/// every fixture is deterministic in-process); its fixture artifacts record its own values.
fn receipt_quality_over_repeats(
    suites: &[ProductFixtureSuite],
) -> Result<QualityObservation, String> {
    if suites.len() != QUALITY_MEASUREMENTS {
        return Err(format!(
            "a row's quality is measured exactly {QUALITY_MEASUREMENTS} time(s) per arm, not {}",
            suites.len()
        ));
    }
    if suites.iter().any(|suite| {
        suite.teacher_forced_candidate.is_none() && suite.forced_continuation.is_none()
    }) {
        return Err("every measured repeat requires its teacher-forced greedy agreement".into());
    }
    if suites.iter().any(|suite| {
        suite.teacher_forced_cache.is_none() && suite.multi_turn_forced_continuation.is_none()
    }) {
        return Err(
            "every measured repeat requires its teacher-forced multi-turn prompt-cache agreement"
                .into(),
        );
    }
    let (primary, repeats) = suites
        .split_first()
        .ok_or("product receipt requires its fixture repeats")?;
    // A row's forced continuation is one deterministic measurement shared by every repeat.
    if repeats
        .iter()
        .any(|suite| suite.forced_continuation != primary.forced_continuation)
    {
        return Err("every measured repeat must share the row's one forced continuation".into());
    }
    if repeats.iter().any(|suite| {
        suite.multi_turn_forced_continuation != primary.multi_turn_forced_continuation
            || suite.multi_turn_forced_pass != primary.multi_turn_forced_pass
    }) {
        return Err(
            "every measured repeat must share the row's one turn-2 forced continuation".into(),
        );
    }
    let mut quality = primary.quality()?;
    quality.repeat_metrics = vec![compute_quality(&quality)?];
    let ratio = |q: &QualityObservation| {
        if q.greedy_total == 0 {
            Err("teacher-forced greedy agreement has zero positions".to_string())
        } else {
            Ok(q.greedy_matches as f64 / q.greedy_total as f64)
        }
    };
    quality.greedy_agreement_by_repeat = vec![ratio(&quality)?];
    let mut weakest = (
        ratio(&quality)?,
        quality.greedy_matches,
        quality.greedy_total,
    );
    for repeat in repeats {
        let repeat = repeat.quality()?;
        quality.repeat_metrics.push(compute_quality(&repeat)?);
        quality.needle_discriminating &= repeat.needle_discriminating;
        quality.tool_discriminating &= repeat.tool_discriminating;
        let agreement = ratio(&repeat)?;
        quality.greedy_agreement_by_repeat.push(agreement);
        if agreement < weakest.0 {
            weakest = (agreement, repeat.greedy_matches, repeat.greedy_total);
        }
        quality.free_running_first_divergence = match (
            quality.free_running_first_divergence,
            repeat.free_running_first_divergence,
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        quality.cache_free_running_first_divergence = match (
            quality.cache_free_running_first_divergence,
            repeat.cache_free_running_first_divergence,
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        quality.cache_matched_prefix_tokens = quality
            .cache_matched_prefix_tokens
            .min(repeat.cache_matched_prefix_tokens);
    }
    // The gate is the weakest repeat's agreement; every repeat's value is recorded.
    (quality.greedy_matches, quality.greedy_total) = (weakest.1, weakest.2);
    Ok(quality)
}

/// Lifecycle capabilities a row actually exercised. Dense rows have no compressed lifecycle
/// representation. Compressed rows run single-shot prefill, chunked (prefix-hit suffix) prefill
/// over an imported prefix, append, and cancellation on the compressed cache and exercise an
/// explicit reasoned dense fallback; prompt-cache reuse runs the provider's dense prefix store and
/// is recorded as such.
pub fn receipt_lifecycle(compressed: bool) -> ReceiptLifecycle {
    let mut fallback_reasons = std::collections::BTreeMap::new();
    let unexercised = if compressed {
        "not exercised by the SC-20671 compressed campaign arm"
    } else {
        "dense baseline route has no compressed lifecycle representation"
    };
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
    ] {
        fallback_reasons.insert(format!("{capability}FallbackReason"), unexercised.into());
    }
    if compressed {
        fallback_reasons.insert(
            "promptCacheReuseFallbackReason".into(),
            COMPRESSED_PREFIX_REUSE_FALLBACK_REASON.into(),
        );
    } else {
        fallback_reasons.insert("denseFallbackFallbackReason".into(), unexercised.into());
    }
    ReceiptLifecycle {
        append: true,
        chunked_prefill: true,
        single_shot_prefill: true,
        prompt_cache_reuse: !compressed,
        trim: false,
        rollback: false,
        clear: false,
        cancel: true,
        clone: false,
        batch_split: false,
        batch_merge: false,
        prefix_copy_on_write: false,
        page_import: false,
        page_export: false,
        serialization: false,
        restore: false,
        dense_fallback: compressed,
        post_run_release: true,
        fallback_reasons,
    }
}

/// Every candidate-side product observation of a compressed arm: each fixture's coordinate
/// operation, its lifecycle/quality run, and any secondary coordinate operation.
/// Every observed product dispatch of a compressed arm: the quality measurement's fixtures (their
/// coordinate and quality observations) and every timing run's coordinate operation(s). The
/// kernel fixture's coordinate operation is timing repeat 0's, so it is counted once.
fn compressed_arm_observations<'a>(
    suites: &'a [ProductFixtureSuite],
    timing_runs: &[&'a CoordinateRun],
) -> Vec<&'a ProductObservations> {
    let mut observations = Vec::new();
    for run in timing_runs {
        observations.push(&run.primary.observation);
        if let Some(secondary) = &run.secondary {
            observations.push(&secondary.observation);
        }
    }
    for suite in suites {
        observations.push(&suite.kernel_candidate.quality_observation);
        for result in [
            &suite.tool_candidate,
            &suite.needle_candidate,
            &suite.cache_candidate,
        ] {
            observations.push(&result.observation);
            observations.push(&result.quality_observation);
            if let Some(secondary) = &result.secondary_coordinate_operation {
                observations.push(&secondary.observation);
            }
        }
    }
    observations
}

/// Reduce a compressed arm's product observations to the receipt's compression block. Evidence is
/// exported by the caches themselves; a cache that left the fused path without a recorded reason
/// is refused here rather than published as a silent fallback. Dense reconstructions are counted,
/// never hidden: the receipt validators reject any. Counts cover every observation; the
/// representation and physical bytes come from the primary's coordinate scope only.
pub fn compressed_receipt_block(
    method: CompressedKvMethod,
    primary: &ProductObservations,
    observations: &[&ProductObservations],
) -> Result<ReceiptCompression, String> {
    let evidence = observations
        .iter()
        .flat_map(|observation| &observation.packed_evidence)
        .collect::<Vec<_>>();
    let first = evidence
        .first()
        .ok_or("compressed arm produced no compressed cache evidence")?;
    if first.bits != method.code_bits().bits() {
        return Err(format!(
            "compressed arm for {} produced {}-bit evidence",
            method.id(),
            first.bits
        ));
    }
    if first.representation_identity != method.representation_identity() {
        return Err(format!(
            "compressed arm for {} produced {} evidence",
            method.id(),
            first.representation_identity
        ));
    }
    let mut fallbacks = std::collections::BTreeMap::<(String, String), u64>::new();
    let mut kernel_paths =
        std::collections::BTreeMap::<(String, String, String), (String, u64)>::new();
    let (mut fused_calls, mut full_cache_dequantizations, mut failed_dispatches) = (0_u64, 0, 0);
    for item in &evidence {
        if item.representation_identity != first.representation_identity
            || item.representation_version != first.representation_version
            || item.bits != first.bits
            || item.quantization_group_size != first.quantization_group_size
        {
            return Err("compressed arm mixed representations".into());
        }
        if (item.dense_active
            || item.failed_dispatches != 0
            || item.full_cache_dequantizations != 0)
            && item.fallback_reasons.is_empty()
        {
            return Err("compressed cache left the fused path without a recorded reason".into());
        }
        let count = |value: usize| u64::try_from(value).map_err(|_| "evidence count overflow");
        if item.kernel_paths.iter().map(|path| path.calls).sum::<u64>()
            != count(item.accepted_direct_calls)?
        {
            return Err("compressed cache fused calls are not attributed to kernel paths".into());
        }
        for path in &item.kernel_paths {
            let entry = kernel_paths
                .entry((
                    path.kernel.clone(),
                    path.selection.clone(),
                    path.query_dtype.clone(),
                ))
                .or_insert_with(|| (path.reason.clone(), 0));
            if entry.0 != path.reason {
                return Err("compressed kernel path reported two reasons".into());
            }
            entry.1 += path.calls;
        }
        fused_calls += count(item.accepted_direct_calls)?;
        full_cache_dequantizations += count(item.full_cache_dequantizations)?;
        failed_dispatches += item.failed_dispatches;
        for (operation, reason) in &item.fallback_reasons {
            *fallbacks
                .entry((operation.clone(), reason.clone()))
                .or_default() += 1;
        }
    }
    for observation in observations {
        for (operation, reason) in &observation.dense_fallbacks {
            *fallbacks
                .entry((operation.clone(), reason.clone()))
                .or_default() += 1;
        }
    }
    if fused_calls == 0 {
        return Err("compressed arm never executed the fused compressed-domain reader".into());
    }
    // Representation and physical bytes describe the coordinate operation alone: warmups,
    // fixtures, and the primary's own lifecycle probes (prompt-cache reuse, cancellation) are
    // counted above but never classify it.
    let scope = primary
        .coordinate_scope
        .as_ref()
        .ok_or("compressed primary observation has no coordinate-operation scope")?;
    let coordinate_storage = scope.storage_peak.filter(|_| {
        !scope.packed_evidence.is_empty()
            && scope.dense_fallbacks.is_empty()
            && scope
                .packed_evidence
                .iter()
                .all(|item| !item.dense_active && item.fallback_reasons.is_empty())
    });
    let (persistent_kv_representation, storage) = match coordinate_storage {
        Some(storage) => (COMPRESSED_PERSISTENT_KV, storage),
        None => (
            DENSE_FALLBACK_PERSISTENT_KV,
            crate::primitives::CompressedCacheStorage::default(),
        ),
    };
    let fallbacks = fallbacks
        .into_iter()
        .map(|((operation, reason), calls)| ReceiptCompressionFallback {
            operation,
            reason,
            calls,
        })
        .collect::<Vec<_>>();
    Ok(ReceiptCompression {
        method: method.id().into(),
        representation_identity: first.representation_identity.clone(),
        representation_version: first.representation_version.into(),
        bits: first.bits.into(),
        quantization_group_size: u64::try_from(first.quantization_group_size)
            .map_err(|_| "quantization group size overflows u64")?,
        kernel_gpu_family: method.kernel_gpu_family().as_str().into(),
        kernel_paths: kernel_paths
            .into_iter()
            .map(|((kernel, selection, query_dtype), (reason, calls))| {
                ReceiptCompressionKernelPath {
                    kernel,
                    selection,
                    reason,
                    query_dtype,
                    calls,
                }
            })
            .collect(),
        device_code_bytes: storage.device_code_bytes,
        device_metadata_bytes: storage.device_metadata_bytes,
        host_payload_bytes: storage.host_payload_bytes,
        physical_kv_bytes: storage
            .device_code_bytes
            .checked_add(storage.device_metadata_bytes)
            .and_then(|bytes| bytes.checked_add(storage.host_payload_bytes))
            .ok_or("physical compressed KV bytes overflow u64")?,
        storage_tokens: storage.tokens,
        persistent_kv_representation: persistent_kv_representation.into(),
        fused_calls,
        fallback_calls: fallbacks.iter().map(|fallback| fallback.calls).sum(),
        fallbacks,
        full_cache_dequantizations,
        failed_dispatches,
        measurement_paths: None,
    })
}

#[allow(clippy::too_many_arguments)] // Every argument is producer-owned sealed evidence.
fn product_receipt(
    coordinate: &Coordinate,
    suites: &[ProductFixtureSuite],
    timings: ProductTimingMeasurements,
    executable: &Path,
    fixtures: &[SealedFixtureArtifact],
    warmup_cache_state_version: Option<u64>,
    captured_source: &(String, String),
    admission: ReceiptAdmission,
    compressed: Option<(CompressedKvMethod, Vec<&CoordinateRun>)>,
    reference_model: &SnapshotInventory,
    row_start: ReceiptHostState,
) -> Result<Receipt, String> {
    let suite = suites
        .first()
        .ok_or("product receipt requires its fixture repeats")?;
    let ProductTimingMeasurements {
        samples: timing_samples,
        compile_attribution,
    } = timings;
    let warmup_suite_sha256 = if warmup_cache_state_version.is_some() {
        warmup_probe_suite_sha256(
            &suite.kernel_candidate.observation.session_id,
            std::process::id(),
            &compile_attribution.probe_evidence,
        )?
    } else {
        String::new()
    };
    let quality = receipt_quality_over_repeats(suites)?;
    let observation = &suite.kernel_candidate.observation;
    let required_operations = [
        (
            coordinate.request_mode == "supported-batch",
            "supported-batch",
        ),
        (coordinate.prefill_mode == "chunked", "chunked-prefix-reuse"),
        (
            coordinate.request_mode == "single" && coordinate.prefill_mode == "single-shot",
            "single-shot-generation",
        ),
    ];
    for (_, required_operation) in required_operations.iter().filter(|(required, _)| *required) {
        let primary_executed = observation
            .operations
            .iter()
            .any(|operation| operation == *required_operation);
        let secondary_executed = suite
            .kernel_candidate
            .secondary_coordinate_operation
            .as_ref()
            .is_some_and(|secondary| {
                secondary.operation == *required_operation
                    && secondary
                        .observation
                        .operations
                        .iter()
                        .any(|operation| operation == *required_operation)
            });
        if !primary_executed && !secondary_executed {
            return Err(format!(
                "coordinate did not execute required product operation {required_operation}"
            ));
        }
    }
    // The fixture denominator is whatever the reference arm ran (bf16 for dense rows, the same
    // candidate weights for compressed rows); provenance always names the pinned bf16 reference.
    let denominator = &suite.kernel_reference.observation.snapshot;
    let model = &observation.snapshot;
    if compressed.is_none() && denominator.sha256 != reference_model.sha256 {
        return Err("dense receipt reference arm is not the pinned bf16 reference model".into());
    }
    let reference = reference_model;
    let candidate_contract = benchmark_model(coordinate.family, false)?;
    let reference_contract = benchmark_model(coordinate.family, true)?;
    let inference_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .ok_or("inference root")?;
    let _ = fs::metadata(executable).map_err(|e| e.to_string())?;
    let scene_works_root = PathBuf::from(required_campaign_env("SCENEWORKS_ROOT")?);
    let scene_works_repository =
        checked_repository_identity(&scene_works_root, SCENEWORKS_REPOSITORY)?;
    let inference_repository = checked_repository_identity(inference_root, INFERENCE_REPOSITORY)?;
    let scene_works_revision = captured_source.0.clone();
    let inference_revision = captured_source.1.clone();
    let hardware = probed_command("sysctl", &["-n", "hw.model"], "hardware")?;
    let xcode = probed_command("xcodebuild", &["-version"], "xcode")?;
    // Only a throttled row start refuses a row. Later changes are recorded and flagged.
    let row_end = capture_host_state("row-end")?;
    let (thermal_changed_during_row, power_mode_changed_during_row) = host_state_changes(
        &row_start,
        timing_samples
            .iter()
            .filter_map(|sample| sample.host_state.as_ref())
            .chain(std::iter::once(&row_end)),
    );
    if thermal_changed_during_row || power_mode_changed_during_row {
        eprintln!(
            "sc20671-kv-baseline: host state changed during the row (thermal {} -> {}, power {} -> {}); recorded, not refused",
            row_start.thermal_state, row_end.thermal_state, row_start.power_mode, row_end.power_mode
        );
    }
    let power_mode = row_start.power_mode.clone();
    let normalized_thermal_state = row_start.thermal_state.clone();
    let host_states = vec![row_start, row_end];
    let mlx = locked_mlx_identity(include_bytes!("../../../../Cargo.lock"))?;
    let transcript = format!(
        "{}\n{}\n{}\n{}",
        suite.kernel_candidate.output.text,
        suite.tool_candidate.output.text,
        suite.needle_candidate.output.text,
        suite.cache_candidate.output.text
    );
    let lifecycle = receipt_lifecycle(compressed.is_some());
    let compression = compressed
        .map(|(method, timing_runs)| {
            if suites.iter().any(|suite| {
                suite.reference != QualityReference::DenseKvSameWeights
                    || suite.kernel_parity_errors.is_none()
            }) {
                return Err(
                    "compressed receipt requires same-weights dense-KV fixtures and fused-reader parity"
                        .to_string(),
                );
            }
            compressed_receipt_block(
                method,
                observation,
                &compressed_arm_observations(suites, &timing_runs),
            )
            .map(|mut block| {
                block.measurement_paths = Some(receipt_measurement_paths(
                    coordinate.request_mode,
                    coordinate.prefill_mode,
                    &block.persistent_kv_representation,
                ));
                block
            })
        })
        .transpose()?;
    let mode = if compression.is_some() {
        "compressed"
    } else {
        "dense"
    };
    let cache_bytes = observation
        .allocations
        .iter()
        .filter(|e| e.role == "cache" && e.lifetime == "persistent")
        .map(|e| e.bytes)
        .max()
        .ok_or("coordinate produced no persistent cache snapshot")?;
    let model_weights_bytes = observation
        .allocations
        .iter()
        .filter(|event| event.role == "weights" && event.lifetime == "persistent")
        .map(|event| event.bytes)
        .max()
        .ok_or("coordinate produced no measured model-weight allocation")?;
    let workspace = observation
        .allocations
        .iter()
        .filter(|e| {
            e.lifetime == "transient"
                && (e.role == "cache" || e.role == "attention-workspace" || e.role == "output")
        })
        .map(|event| event.bytes)
        .max()
        .unwrap_or(0);
    let release = observation.phases.last().ok_or("release phase")?;
    let weights_loaded = observation
        .phases
        .iter()
        .find(|phase| phase.phase == "weights-loaded")
        .ok_or("weights-loaded phase")?;
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
                independent_reference: fixture_independent_reference(
                    name,
                    suite.reference,
                    &denominator.sha256,
                ),
            },
        );
    }
    let template = Receipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        harness_version: RECEIPT_HARNESS_VERSION.into(),
        run_id: seal_bytes(
            format!(
                "{}:{}:{}",
                coordinate_slug(coordinate),
                model.sha256,
                seal_bytes(transcript.as_bytes())
            )
            .as_bytes(),
        ),
        captured_at: release.timestamp.clone(),
        mode: mode.into(),
        status: "complete".into(),
        contract_hash: QUALITY_CONTRACT_HASH.into(),
        receipt_sha256: String::new(),
        provenance: ReceiptProvenance {
            scene_works_repository,
            inference_repository,
            scene_works_revision,
            inference_revision,
            mlx_version: mlx.version,
            mlx_source: mlx.source,
            mlx_revision: mlx.revision,
            dependency_lock_sha256: seal_bytes(include_bytes!("../../../../Cargo.lock")),
            os: std::env::consts::OS.into(),
            xcode,
            hardware,
            model_id: format!(
                "{}@{};architecture={};inventory={}",
                candidate_contract.repository,
                candidate_contract.revision,
                candidate_contract.architecture,
                model.sha256
            ),
            model_file_sha256: model.sha256.clone(),
            model_file_bytes: model.bytes,
            reference_model_id: format!(
                "{}@{};architecture={};inventory={}",
                reference_contract.repository,
                reference_contract.revision,
                reference_contract.architecture,
                reference.sha256
            ),
            reference_model_sha256: reference.sha256.clone(),
            reference_model_bytes: reference.bytes,
            power_mode,
            thermal_state: normalized_thermal_state,
            host_states,
            thermal_changed_during_row,
            power_mode_changed_during_row,
            command_template: "sc20671-kv-baseline --mode {mode}".into(),
            command: format!("sc20671-kv-baseline --mode {mode}"),
            campaign_session_id: observation.session_id.clone(),
            campaign_cache_state_version: observation.cache_state_version,
            coordinate_operation_sha256: coordinate_operation_digest(&suite.kernel_candidate),
        },
        matrix: ReceiptMatrix {
            family: coordinate.family.into(),
            context_band: coordinate.context_band.into(),
            request_mode: coordinate.request_mode.into(),
            prefill_mode: coordinate.prefill_mode.into(),
            process_temperature: coordinate.process_temperature.into(),
        },
        geometry: ReceiptGeometry {
            batch: if coordinate.request_mode == "single" {
                1
            } else {
                2
            },
            query_heads: observation.geometry.query_heads,
            kv_heads: observation.geometry.kv_heads,
            head_dimension: observation.geometry.head_dimension,
            query_length: suite.kernel_candidate.coordinate_prompt_tokens,
            kv_length: observation.cache_live_tokens,
            layers: observation.geometry.layers,
            element_bytes: observation.geometry.element_bytes,
            capacity: observation.cache_capacity_tokens,
            context_window_tokens: suite.context_window_tokens,
            context_target_tokens: suite.context_target_tokens,
            context_payload_tokens: suite.context_payload_tokens,
        },
        memory: ReceiptMemory {
            model_weights_bytes,
            persistent_kv_bytes: cache_bytes,
            transient_workspace_bytes: workspace,
            dense_theoretical_kv_bytes: 0,
            prefill_peak_window: observation.prefill_peak_window.clone(),
            phase_samples: vec![],
            allocation_events: vec![],
            reconciliation: ReceiptReconciliation {
                expected_dense_kv_bytes: 0,
                observed_persistent_kv_bytes: 0,
                tolerance_bytes: 0,
            },
            release: release_evidence(weights_loaded, release),
            admission,
            dense_kv_share_bps: 0,
            below_memory_material_share: false,
        },
        timings: ReceiptTimings {
            load_ms: 0.0,
            prefill_ms: 0.0,
            ttft_ms: 0.0,
            first_token_ms: 0.0,
            decode_tokens_per_second: 0.0,
            cold_compile_ms: None,
            warm_compile_ms: 0.0,
            compile_attribution: compile_attribution.clone(),
            samples: vec![],
            summary: ReceiptTimingSummary {
                decode_tokens_per_second_mean: 0.0,
                decode_tokens_per_second_p95: 0.0,
                decode_tokens_per_second_variance: 0.0,
                decode_tokens_per_second_coefficient_of_variation: 0.0,
                confidence_interval_low: 0.0,
                confidence_interval_high: 0.0,
            },
        },
        quality: ReceiptQuality {
            parity_max_error: 0.0,
            perplexity_delta: 0.0,
            greedy_token_agreement: 0.0,
            greedy_agreement_method: String::new(),
            free_running_first_divergence: None,
            greedy_token_agreement_by_repeat: Vec::new(),
            structured_tool_agreement: 0.0,
            needle_retrieval: 0.0,
            needle_discriminating: false,
            tool_discriminating: false,
            multi_turn_prompt_cache: 0.0,
            multi_turn_prompt_cache_method: String::new(),
            multi_turn_free_running_first_divergence: None,
            multi_turn_matched_prefix_tokens: 0,
            multi_turn_cache: ReceiptMultiTurnCache::default(),
            multi_turn_forced_continuation: None,
            multi_turn_forced_pass: None,
            statistics: contract_statistics(),
            fixture_evidence,
            quality_gate: None,
            forced_continuation: None,
        },
        lifecycle,
        cancellation: ReceiptCancellation {
            cleanup_verified: true,
        },
        warmup: ReceiptWarmup {
            required: coordinate.process_temperature == "warm",
            completed: warmup_cache_state_version.is_some(),
            worker_pid: std::process::id(),
            suite_sha256: warmup_suite_sha256,
            session_id: if coordinate.process_temperature == "warm" {
                observation.session_id.clone()
            } else {
                String::new()
            },
            cache_state_version: warmup_cache_state_version.unwrap_or_default(),
        },
        compression,
    };
    ReceiptBuilder {
        template,
        phases: observation.phases.clone(),
        allocations: observation.allocations.clone(),
        timings: timing_samples,
        compile_attribution,
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
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    fn test_compile_probe_evidence(
        operation: &str,
        source: &str,
        matrix_coordinate: &str,
        dispatches: &[f64],
    ) -> Vec<ReceiptCompileProbeEvidence> {
        dispatches
            .iter()
            .enumerate()
            .map(|(index, dispatch_ms)| ReceiptCompileProbeEvidence {
                index: index as u64,
                operation: operation.into(),
                source: source.into(),
                matrix_coordinate: matrix_coordinate.into(),
                setup_ms: 0.0,
                dispatch_ms: *dispatch_ms,
                operation_evidence_sha256: format!("{index:064x}"),
            })
            .collect()
    }

    #[test]
    fn needle_prompt_places_one_neutral_needle_before_the_long_context() {
        let prompt = needle_fixture_prompt("baseline", "DISTRACTOR", "SC20671-NEEDLE");
        assert_eq!(prompt.matches("SC20671-NEEDLE").count(), 1);
        assert!(prompt.find("SC20671-NEEDLE").unwrap() < prompt.find("DISTRACTOR").unwrap());
        assert!(prompt.contains("BEGIN LONG CONTEXT\nDISTRACTOR\nEND LONG CONTEXT"));
        assert!(prompt.ends_with(
            "What is the special magic identifier mentioned in the text above? Reply with only the identifier."
        ));
    }

    /// A synthetic product fixture result whose output text is `text`.
    fn test_fixture_result(text: &str) -> ProductFixtureResult {
        let observation = || ProductObservations {
            snapshot: SnapshotInventory {
                root: PathBuf::new(),
                files: Vec::new(),
                bytes: 1,
                sha256: "b".repeat(64),
            },
            geometry: ProductGeometry {
                query_heads: 24,
                kv_heads: 8,
                head_dimension: 8,
                layers: 28,
                element_bytes: 2,
            },
            phases: vec![
                ReceiptPhase {
                    phase: "prefill-peak".into(),
                    pid: 42,
                    source: PHYS_FOOTPRINT_SOURCE.into(),
                    timestamp: "2026-01-01T00:00:00Z".into(),
                    phys_footprint_bytes: 123,
                    phys_footprint_peak_bytes: 456,
                    mlx: ReceiptMlx {
                        source: "mlx_rs::memory".into(),
                        active_bytes: 12,
                        cache_bytes: 34,
                        peak_bytes: 56,
                    },
                };
                8
            ],
            phase_elapsed_ms: Vec::new(),
            allocations: Vec::new(),
            prefill_logits: Vec::new(),
            token_probabilities: vec![(7, 0.5), (9, 0.25)],
            forced_token_probabilities: Vec::new(),
            session_id: "session".into(),
            cache_state_version: 1,
            operations: Vec::new(),
            load_elapsed_ms: 1.0,
            prefill_peak_window: ReceiptPeakWindow {
                started_at: "2026-01-01T00:00:00Z".into(),
                baseline_active_bytes: 1,
                reset_peak_bytes: 0,
            },
            cache_live_tokens: 32,
            cache_capacity_tokens: 256,
            packed_evidence: Vec::new(),
            dense_fallbacks: Vec::new(),
            coordinate_scope: None,
        };
        ProductFixtureResult {
            observation: observation(),
            quality_observation: observation(),
            output: TextLlmOutput {
                text: text.into(),
                ..Default::default()
            },
            coordinate_operation: "chunked-prefix-reuse".into(),
            coordinate_generated_tokens: 1,
            coordinate_prompt_tokens: 32,
            coordinate_output_sha256: "c".repeat(64),
            compile_setup_ms: 0.0,
            compile_dispatch_ms: 1.0,
            secondary_coordinate_operation: None,
            multi_turn: Some(test_multi_turn()),
        }
    }

    /// A multi-turn fixture's valid per-turn records: turn 1 misses, turn 2 reuses 40 tokens.
    fn test_multi_turn() -> MultiTurnPromptCacheTurns {
        MultiTurnPromptCacheTurns {
            turn1: PromptCacheTurn {
                prompt_tokens: 32,
                prompt_sha256: "1".repeat(64),
                cache_hit: false,
                reused_prefix_tokens: 0,
            },
            turn2: PromptCacheTurn {
                prompt_tokens: 60,
                prompt_sha256: "2".repeat(64),
                cache_hit: true,
                reused_prefix_tokens: 40,
            },
        }
    }

    /// Both sessions' turn records of a compressed row's turn-2 forced continuation.
    fn test_multi_turn_forced_pass() -> MultiTurnForcedPass {
        MultiTurnForcedPass {
            reference: test_multi_turn(),
            candidate: test_multi_turn(),
        }
    }

    /// A synthetic fixture half: valid tool call, `needle_text` as the needle answer.
    fn test_fixture_half(needle_text: &str) -> ProductFixtureHalf {
        let mut tool = test_fixture_result("");
        let mut arguments = serde_json::Map::new();
        arguments.insert(
            "fact".into(),
            serde_json::json!("SC20671 structured fixture"),
        );
        tool.output.tool_calls = vec![core_llm::ToolCall::new("record_baseline_fact", arguments)];
        ProductFixtureHalf {
            kernel: test_fixture_result("kernel"),
            tool,
            needle_result: test_fixture_result(needle_text),
            cache: test_fixture_result("cache"),
            needle: "SC20671-NUMERIC-NEEDLE-9b7a2e".into(),
            context_window_tokens: 4096,
            context_target_tokens: 32,
            context_payload_tokens: 32,
            context_payload_sha256: "d".repeat(64),
            fixture_prompt_tokens: [32; 4],
            fixture_prompt_sha256: std::array::from_fn(|_| "e".repeat(64)),
        }
    }

    #[test]
    fn needle_miss_is_a_recorded_observation_with_role_diagnostics() {
        let half = test_fixture_half;
        let coordinate = required_coordinates()[0].clone();
        let candidate = half("wrong\nanswer");
        let reference = half("SC20671-NUMERIC-NEEDLE-9b7a2e");
        let identity = "a".repeat(64);
        let candidate_line = fixture_half_attempt_diagnostic(
            &candidate,
            &coordinate,
            &identity,
            "candidate",
            "repeat",
            0,
        );
        let reference_line = fixture_half_attempt_diagnostic(
            &reference,
            &coordinate,
            &identity,
            "reference",
            "repeat",
            0,
        );
        assert_eq!(candidate_line.lines().count(), 1);
        assert!(candidate_line.len() < 32 * 1024);
        let parse = |line: &str| -> serde_json::Value {
            serde_json::from_str(line.strip_prefix("SC20671_DIAGNOSTIC_UNACCEPTED ").unwrap())
                .unwrap()
        };
        let candidate_record = parse(&candidate_line);
        let reference_record = parse(&reference_line);
        assert_eq!(candidate_record["resumeIdentitySha256"], identity);
        assert_eq!(candidate_record["role"], "candidate");
        assert_eq!(candidate_record["structuredToolValid"], true);
        assert_eq!(candidate_record["needleRecovered"], false);
        assert_eq!(reference_record["role"], "reference");
        assert_eq!(reference_record["needleRecovered"], true);
        let needle = &candidate_record["fixtures"]["long-context-needle"];
        assert_eq!(needle["output"]["text"]["prefix"], "wrong\nanswer");
        assert_eq!(
            needle["coordinateObservation"]["geometry"]["elementBytes"],
            2
        );
        assert_eq!(needle["coordinateObservation"]["geometry"]["capacity"], 256);
        assert_eq!(
            needle["coordinateObservation"]["phasePeaks"][0]["mlxPeakBytes"],
            56
        );
        let oversized = half(&"x".repeat(100_000));
        let oversized_line = fixture_half_attempt_diagnostic(
            &oversized,
            &coordinate,
            &identity,
            "candidate",
            "repeat",
            1,
        );
        assert!(oversized_line.len() < 32 * 1024);
        assert_eq!(
            parse(&oversized_line)["fixtures"]["long-context-needle"]["output"]["text"]
                ["truncated"],
            true
        );
        // Dense rows are raw characterization: a Q4 miss (or refusal) against a recovering bf16
        // reference is recorded, never a rejected row.
        let dense = pair_product_fixture_halves(
            candidate,
            reference,
            QualityReference::Bf16Characterization,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!((dense.needle_matches, dense.needle_total), (0, 1));
        assert!(!dense.outcomes.candidate_needle_recovered);
        assert!(dense.outcomes.reference_needle_recovered);
        assert!(!dense.needle_discriminating);
        assert_eq!(dense.tool_matches, 1);
        let reversed = pair_product_fixture_halves(
            half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
            half("wrong"),
            QualityReference::Bf16Characterization,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!(reversed.needle_matches, 1);
        assert!(reversed.needle_discriminating);
        assert!(!reversed.outcomes.reference_needle_recovered);
        let mut invalid_tool = half("SC20671-NUMERIC-NEEDLE-9b7a2e");
        invalid_tool.tool.output.tool_calls.clear();
        let characterized = pair_product_fixture_halves(
            invalid_tool,
            half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
            QualityReference::Bf16Characterization,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!(characterized.tool_matches, 0);
        assert!(!characterized.outcomes.candidate_tool_valid);
        assert!(characterized.outcomes.reference_tool_valid);

        // Compressed rows: the same-weights dense-KV run is the denominator.
        let compressed = pair_product_fixture_halves(
            half("wrong"),
            half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
            QualityReference::DenseKvSameWeights,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!(compressed.needle_matches, 0);
        assert!(compressed.needle_discriminating);
        let shared_miss = pair_product_fixture_halves(
            half("I cannot help with that."),
            half("I cannot help with that."),
            QualityReference::DenseKvSameWeights,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!(
            shared_miss.needle_matches, 0,
            "contract v5: the candidate's own (failed) recovery, never agreement with the dense miss"
        );
        assert!(
            !shared_miss.needle_discriminating,
            "a shared miss must be flagged, not silently counted as retrieval"
        );
        let divergent_miss = pair_product_fixture_halves(
            half("something else"),
            half("I cannot help with that."),
            QualityReference::DenseKvSameWeights,
        )
        .unwrap()
        .quality()
        .unwrap();
        assert_eq!(divergent_miss.needle_matches, 0);
        assert!(!divergent_miss.needle_discriminating);
        // Teacher-forced suites, each one arm-pair quality measurement.
        let pair = |compressed: ProductFixtureHalf, dense: ProductFixtureHalf| {
            let mut suite = pair_product_fixture_halves(
                compressed,
                dense,
                QualityReference::DenseKvSameWeights,
            )
            .unwrap();
            suite.teacher_forced_candidate = Some(TeacherForcedChoices {
                choices: stream_tokens(
                    &suite
                        .kernel_reference
                        .quality_observation
                        .token_probabilities,
                ),
                stop_tokens: Vec::new(),
            });
            suite.stream_likelihood =
                Some(StreamLikelihood::from_probabilities(&[0.5, 0.25], &[0.5, 0.25]).unwrap());
            suite.teacher_forced_cache = Some(TeacherForcedChoices {
                choices: stream_tokens(
                    &suite
                        .cache_reference
                        .quality_observation
                        .token_probabilities,
                ),
                stop_tokens: Vec::new(),
            });
            suite
        };
        let mut invalid_dense_tool = half("I cannot help with that.");
        invalid_dense_tool.tool.output.tool_calls.clear();
        let mut invalid_compressed_tool = half("I cannot help with that.");
        invalid_compressed_tool.tool.output.tool_calls.clear();
        let mut repeats = [
            pair(
                half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
                half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
            ),
            pair(invalid_compressed_tool, invalid_dense_tool),
        ];
        assert!(repeats[0].quality().unwrap().needle_discriminating);
        // The suite's greedy agreement is the teacher-forced one: a diverged free-running
        // candidate stream does not lower it when every forced position agrees.
        let reference_stream = stream_tokens(
            &repeats[0]
                .kernel_reference
                .quality_observation
                .token_probabilities,
        );
        repeats[0]
            .kernel_candidate
            .quality_observation
            .token_probabilities
            .iter_mut()
            .for_each(|(token, _)| *token += 1000);
        let forced_quality = repeats[0].quality().unwrap();
        assert_eq!(
            (forced_quality.greedy_matches, forced_quality.greedy_total),
            (reference_stream.len() as u64, reference_stream.len() as u64)
        );
        assert_eq!(forced_quality.free_running_first_divergence, Some(0));
        // Contract v5: the receipt is the row's one quality measurement.
        let [measured, shared_invalid_tools] = repeats;
        let mut once = [measured];
        let forced = once[0].teacher_forced_candidate.take();
        assert!(receipt_quality_over_repeats(&once).is_err());
        once[0].teacher_forced_candidate = forced;
        let original = once[0].teacher_forced_candidate.clone();
        if let Some(forced) = once[0].teacher_forced_candidate.as_mut() {
            forced.choices[0] += 1000;
        }
        let quality = receipt_quality_over_repeats(&once).unwrap();
        once[0].teacher_forced_candidate = original;
        let total = quality.greedy_total;
        assert_eq!(quality.greedy_matches, total - 1);
        assert_eq!(
            quality.greedy_agreement_by_repeat,
            vec![(total - 1) as f64 / total as f64]
        );
        let recorded = receipt_quality_over_repeats(&once).unwrap();
        assert!(recorded.needle_discriminating && recorded.tool_discriminating);
        assert_eq!(recorded.needle_matches, 1);
        let repeats = [once.into_iter().next().unwrap(), shared_invalid_tools];
        let shared_invalid = repeats[1].quality().unwrap();
        assert_eq!(
            shared_invalid.tool_matches, 1,
            "identical invalid calls agree"
        );
        assert!(!shared_invalid.tool_discriminating, "but are flagged");
        let mut other_weights = half("SC20671-NUMERIC-NEEDLE-9b7a2e");
        other_weights
            .needle_result
            .quality_observation
            .snapshot
            .sha256 = "f".repeat(64);
        assert!(pair_product_fixture_halves(
            half("SC20671-NUMERIC-NEEDLE-9b7a2e"),
            other_weights,
            QualityReference::DenseKvSameWeights,
        )
        .unwrap()
        .quality()
        .unwrap_err()
        .contains("same weights"));
    }

    /// SC-20669 fit-boundary finding: `perplexityDelta` is both arms' likelihood of ONE stream (the
    /// reference stream the candidate is teacher-forced on), never each arm's likelihood of its
    /// own free-running stream. The free-running streams of a row with perfect teacher-forced
    /// agreement diverge after a single near-tie flip (position 9 of the Llama fit-boundary row) and
    /// then score two different texts: there, 479/479 forced agreement sat beside a 0.29 delta.
    #[test]
    fn perplexity_delta_scores_both_arms_on_the_reference_stream() {
        let mut suite = pair_product_fixture_halves(
            test_fixture_half(TEST_NEEDLE),
            test_fixture_half(TEST_NEEDLE),
            QualityReference::DenseKvSameWeights,
        )
        .unwrap();
        suite.forced_continuation = Some(test_forced_continuation(FORCED_CONTINUATION_TOKENS));
        suite.multi_turn_forced_continuation = Some(test_multi_turn_forced_continuation(
            FORCED_CONTINUATION_TOKENS,
        ));
        suite.multi_turn_forced_pass = Some(test_multi_turn_forced_pass());
        // The candidate's free-running continuation diverged into a different, less likely text.
        suite
            .kernel_candidate
            .quality_observation
            .token_probabilities = vec![(8, 0.05), (10, 0.1)];
        // A teacher-forced repeat without the scored stream is refused, never silently scored on
        // the free-running streams.
        assert!(suite
            .quality()
            .unwrap_err()
            .contains("reference stream's likelihood"));
        // Both arms scored on the same stream: the delta is the representation's alone.
        let likelihood =
            StreamLikelihood::from_probabilities(&[0.9, 0.8, 0.7], &[0.9, 0.8, 0.69]).unwrap();
        suite.stream_likelihood = Some(likelihood.clone());
        let quality = suite.quality().unwrap();
        assert_eq!(quality.reference_perplexity, likelihood.reference);
        assert_eq!(quality.candidate_perplexity, likelihood.candidate);
        let delta = compute_quality(&quality).unwrap().perplexity_delta;
        let expected = -(0.69f64.ln() - 0.7f64.ln()) / 3.0;
        assert!((delta - expected).abs() < 1e-12, "{delta} != {expected}");
        assert!(delta < COMPRESSED_PERPLEXITY_DELTA_MAX);
        // The arms must be scored on the same positions.
        assert!(StreamLikelihood::from_probabilities(&[0.9, 0.8], &[0.9])
            .unwrap_err()
            .contains("one stream"));
        assert!(StreamLikelihood::from_probabilities(&[0.9, 0.0], &[0.9, 0.5]).is_err());
        // Dense rows: the candidate teacher-forced on the reference's natural stream is scored the
        // same way.
        let mut dense = pair_product_fixture_halves(
            test_fixture_half(TEST_NEEDLE),
            test_fixture_half(TEST_NEEDLE),
            QualityReference::Bf16Characterization,
        )
        .unwrap();
        dense.teacher_forced_candidate = Some(TeacherForcedChoices {
            choices: vec![7, 9],
            stop_tokens: Vec::new(),
        });
        dense.teacher_forced_cache = Some(TeacherForcedChoices {
            choices: vec![7, 9],
            stop_tokens: Vec::new(),
        });
        dense
            .kernel_candidate
            .quality_observation
            .token_probabilities = vec![(8, 0.05), (10, 0.1)];
        assert!(dense.quality().is_err());
        dense.stream_likelihood =
            Some(StreamLikelihood::from_probabilities(&[0.5, 0.25], &[0.5, 0.25]).unwrap());
        let dense = dense.quality().unwrap();
        assert_eq!(dense.candidate_perplexity, dense.reference_perplexity);
    }

    /// The product observer keeps the model's probability of each forced token (the dense-row
    /// candidate's likelihood of the reference stream) and refuses one outside teacher forcing.
    #[test]
    fn product_observer_records_forced_token_probabilities_only_under_teacher_forcing() {
        let mut forced = ProductObserver::teacher_forced(vec![3, 4]);
        forced.forced_token_probability(3, 0.25);
        forced.forced_token_probability(4, 0.5);
        assert!(forced.error.is_none());
        assert_eq!(forced.forced_token_probabilities, vec![0.25, 0.5]);
        forced.forced_token_probability(4, f64::NAN);
        assert!(forced.error.is_some());
        let mut free = ProductObserver::new();
        free.forced_token_probability(3, 0.25);
        assert!(free.error.is_some() && free.forced_token_probabilities.is_empty());
    }

    /// Contract v5 measures quality once: the candidate's kernel fixture reuses timing repeat 0's
    /// coordinate operation only when it is this session's run of the coordinate's operation, and
    /// the compressed arm's evidence counts every timing run's coordinate dispatch and the quality
    /// measurement's dispatches, the reused kernel coordinate operation once.
    #[test]
    fn quality_is_measured_once_beside_the_timing_runs() {
        let chunked = Coordinate {
            family: "llama",
            context_band: "short",
            request_mode: "single",
            prefill_mode: "chunked",
            process_temperature: "cold",
        };
        let run = CoordinateRun::of(&test_fixture_result("kernel"));
        reusable_kernel_coordinate(&run, "session", &chunked).unwrap();
        assert!(reusable_kernel_coordinate(&run, "another session", &chunked).is_err());
        let single_shot = Coordinate {
            prefill_mode: "single-shot",
            ..chunked.clone()
        };
        assert!(reusable_kernel_coordinate(&run, "session", &single_shot).is_err());
        let batch_chunked = Coordinate {
            request_mode: "supported-batch",
            ..chunked.clone()
        };
        let mut batch = run.clone();
        batch.primary.operation = "supported-batch".into();
        assert!(
            reusable_kernel_coordinate(&batch, "session", &batch_chunked).is_err(),
            "a batch+chunked row's reused operation must carry its secondary dispatch"
        );
        batch.secondary = Some(run.primary.clone());
        reusable_kernel_coordinate(&batch, "session", &batch_chunked).unwrap();

        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS);
        let suites = compressed_test_suites(&continuation, &[]);
        let timing = (0..TIMING_REPEATS)
            .map(|_| CoordinateRun::of(&suites[0].kernel_candidate))
            .collect::<Vec<_>>();
        let timing_refs = timing.iter().collect::<Vec<_>>();
        // 5 timing coordinate dispatches + the kernel quality request + 3 fixtures x (coordinate
        // + quality request).
        assert_eq!(
            compressed_arm_observations(&suites, &timing_refs).len(),
            TIMING_REPEATS + 1 + 3 * 2
        );
        assert_eq!(QUALITY_MEASUREMENTS, 1);
        assert_eq!(contract_statistics().repeats, TIMING_REPEATS as u64);
        assert!(contract_statistics().quality_measured_once);
    }

    fn noise_floor_row(slug: &str, chunked: (f64, u64, f64)) -> serde_json::Value {
        let control = |(agreement, flips, delta): (f64, u64, f64)| {
            serde_json::json!({
                "agreement": agreement,
                "matches": 0,
                "flipCount": flips,
                "firstFlipPositions": [],
                "referenceNegativeLogLikelihood": 0.5,
                "controlNegativeLogLikelihood": 0.5 + delta,
                "perplexityDelta": delta,
            })
        };
        serde_json::json!({
            "kind": "sc20669-dense-noise-floor",
            "coordinate": slug,
            "controls": {
                "one-shot-repeat": control((1.0, 0, 0.0)),
                "chunked-prefill": control(chunked),
                "multi-turn-repeat": control((1.0, 0, 0.0)),
            },
        })
    }

    /// Receipt honesty: a compressed receipt names the KV path of each measurement block, so a
    /// supported-batch row's dense-fallback memory and first-token time are never read as the path
    /// its compressed single-sequence decode and quality ran on.
    #[test]
    fn measurement_paths_name_each_blocks_kv_path() {
        let batch = receipt_measurement_paths(
            "supported-batch",
            "single-shot",
            DENSE_FALLBACK_PERSISTENT_KV,
        );
        let path = |kv: &str, operation: &str, sequences: u64| ReceiptMeasurementPath {
            kv_path: kv.into(),
            operation: operation.into(),
            sequences,
        };
        assert_eq!(
            batch,
            ReceiptMeasurementPaths {
                memory: path("dense-fallback", "supported-batch", 2),
                prefill_first_token: path("dense-fallback", "supported-batch", 2),
                decode_timing: path("compressed", "steady-decode", 1),
                quality: path("compressed", "quality", 1),
            }
        );
        let chunked = receipt_measurement_paths("single", "chunked", COMPRESSED_PERSISTENT_KV);
        assert_eq!(
            chunked.memory,
            path("compressed", "chunked-prefix-reuse", 1)
        );
        assert_eq!(chunked.prefill_first_token, chunked.memory);

        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS);
        let suites = compressed_test_suites(&continuation, &[]);
        let mut receipt = compressed_test_builder(receipt_quality_over_repeats(&suites).unwrap())
            .finish()
            .unwrap();
        let compression = receipt.compression.as_mut().unwrap();
        let honest = receipt_measurement_paths(
            &receipt.matrix.request_mode,
            &receipt.matrix.prefill_mode,
            &compression.persistent_kv_representation,
        );
        compression.measurement_paths = Some(honest.clone());
        validate_receipt_semantics(&receipt).unwrap();
        // The field round-trips, and a receipt produced before it still reads.
        let parsed: Receipt = serde_json::from_slice(&receipt.bytes().unwrap()).unwrap();
        assert_eq!(
            parsed.compression.as_ref().unwrap().measurement_paths,
            Some(honest.clone())
        );
        let mut legacy = serde_json::to_value(&receipt).unwrap();
        legacy["compression"]
            .as_object_mut()
            .unwrap()
            .remove("measurementPaths");
        let legacy: Receipt = serde_json::from_value(legacy).unwrap();
        assert!(legacy.compression.unwrap().measurement_paths.is_none());
        // A block that claims another path is refused.
        for edit in [
            (|paths: &mut ReceiptMeasurementPaths| {
                paths.decode_timing.kv_path = DENSE_FALLBACK_PERSISTENT_KV.into()
            }) as fn(&mut ReceiptMeasurementPaths),
            |paths| paths.memory.kv_path = DENSE_FALLBACK_PERSISTENT_KV.into(),
            |paths| paths.prefill_first_token.sequences = 2,
            |paths| paths.quality.operation = "supported-batch".into(),
        ] {
            let mut forged = receipt.clone();
            edit(
                forged
                    .compression
                    .as_mut()
                    .unwrap()
                    .measurement_paths
                    .as_mut()
                    .unwrap(),
            );
            assert!(validate_receipt_semantics(&forged)
                .unwrap_err()
                .contains("measurementPaths"));
        }
    }

    /// The noise floor's same-process dense timing uses A2's definitions and aggregates exactly:
    /// over the same timing samples it reports the values a receipt's `timings` would, keyed by the
    /// coordinate, and the summary pairs each row's dense timing with its coordinate.
    #[test]
    fn noise_floor_dense_timing_is_a2s_timing_of_the_dense_session() {
        let mut builder = test_receipt_builder();
        // Distinct samples, so an aggregate other than the receipt's mean is visible.
        for (index, sample) in builder.timings.iter_mut().enumerate() {
            let step = index as f64;
            sample.prefill_ms += step * 0.5;
            sample.ttft_ms += step * 0.25;
            sample.first_token_ms += step * 0.75;
            sample.decode_tokens_per_second = 1.0 + step * 0.01;
            sample.steady_decode.decode_ms = sample.steady_decode.timed_tokens as f64 * 1_000.0
                / sample.decode_tokens_per_second;
        }
        let samples = builder.timings.clone();
        let receipt = builder.finish().unwrap();
        let matrix = &receipt.matrix;
        let coordinate = required_coordinates()
            .into_iter()
            .find(|coordinate| {
                coordinate.family == matrix.family
                    && coordinate.process_temperature == matrix.process_temperature
            })
            .unwrap();
        let warmups = if coordinate.process_temperature == "warm" {
            TIMING_WARMUPS
        } else {
            0
        };
        let timing =
            noise_floor_dense_timing(&coordinate, "single-shot-generation", warmups, &samples)
                .unwrap();
        assert_eq!(timing["coordinate"], coordinate_slug(&coordinate));
        assert_eq!(timing["kvPath"], "dense");
        assert_eq!(timing["repeats"], TIMING_REPEATS);
        for (key, value) in [
            ("prefillMs", receipt.timings.prefill_ms),
            ("ttftMs", receipt.timings.ttft_ms),
            ("firstTokenMs", receipt.timings.first_token_ms),
            (
                "decodeTokensPerSecond",
                receipt.timings.decode_tokens_per_second,
            ),
        ] {
            assert_eq!(timing[key].as_f64(), Some(value), "{key}");
        }
        assert!(noise_floor_dense_timing(&coordinate, "op", warmups, &samples[1..]).is_err());
        assert!(noise_floor_dense_timing(&coordinate, "op", warmups + 1, &samples).is_err());
        let mut row = noise_floor_row(&coordinate_slug(&coordinate), (1.0, 0, 0.0));
        row["denseTiming"] = timing;
        let summary = noise_floor_summary(&[row]).unwrap();
        let paired = &summary["denseTiming"][coordinate_slug(&coordinate)];
        // Synthetic fixture values (no clock): the paired entry carries the receipt aggregates.
        for (key, value) in [
            (
                "decodeTokensPerSecond",
                receipt.timings.decode_tokens_per_second,
            ),
            ("firstTokenMs", receipt.timings.first_token_ms),
        ] {
            assert_eq!(paired[key].as_f64(), Some(value), "{key}");
        }
    }

    /// A noise-floor control is scored with the compressed rows' own metrics on its reference
    /// stream: the multi-turn control with A2's turn-2 forced-continuation evidence, the kernel
    /// controls with the kernel's, both with the stream-scored perplexity delta.
    #[test]
    fn noise_floor_multi_turn_control_scores_turn_two_like_a2() {
        let reference = ScoredContinuation {
            choices: (0..FORCED_CONTINUATION_TOKENS as i32).collect(),
            stream_probabilities: vec![0.5; FORCED_CONTINUATION_TOKENS as usize],
        };
        let mut control = reference.clone();
        control.choices[3] = -1;
        control.choices[700] = -1;
        control.stream_probabilities[0] = 0.25;
        let multi_turn = noise_floor_control(&reference, &control, true).unwrap();
        assert_eq!(multi_turn["method"], MULTI_TURN_FORCED_CONTINUATION_METHOD);
        assert_eq!(multi_turn["flipCount"], 2);
        assert_eq!(
            multi_turn["firstFlipPositions"],
            serde_json::json!([3, 700])
        );
        assert_eq!(multi_turn["agreement"], 1022.0 / 1024.0);
        let delta = 2.0f64.ln() / FORCED_CONTINUATION_TOKENS as f64;
        assert!((multi_turn["perplexityDelta"].as_f64().unwrap() - delta).abs() < 1e-12);
        let kernel = noise_floor_control(&reference, &control, false).unwrap();
        assert_eq!(kernel["method"], FORCED_CONTINUATION_METHOD);
        assert_eq!(kernel["flipCount"], 2);
        assert!(noise_floor_control(
            &reference,
            &ScoredContinuation {
                choices: vec![0],
                stream_probabilities: vec![0.5],
            },
            true,
        )
        .is_err());
    }

    /// The SC-20669 dense noise floor: the summary is each control's worst case over the rows and
    /// says whether the dense arm already misses a frozen threshold against itself.
    #[test]
    fn noise_floor_summary_is_each_controls_worst_row() {
        let mut rows = [
            noise_floor_row("a", (0.9995, 1, 0.004)),
            noise_floor_row("b", (0.998, 2, -0.013)),
        ];
        rows[0]["controls"]["multi-turn-repeat"]["agreement"] = 0.998.into();
        rows[0]["controls"]["multi-turn-repeat"]["flipCount"] = 2.into();
        let summary = noise_floor_summary(&rows).unwrap();
        // The dense multi-turn control judges the multi-turn threshold; no other control does.
        let multi_turn = &summary["controls"]["multi-turn-repeat"];
        assert_eq!(multi_turn["minAgreement"], 0.998);
        assert_eq!(multi_turn["minAgreementRow"], "a");
        assert_eq!(multi_turn["maxFlipCount"], 2);
        assert_eq!(multi_turn["multiTurnThresholdWithinDenseFloor"], true);
        assert_eq!(
            summary["controls"]["chunked-prefill"]["multiTurnThresholdWithinDenseFloor"],
            false
        );
        assert_eq!(summary["thresholds"]["multiTurnPromptCache"], 0.999);
        let mut without = rows.clone();
        without[1]["controls"]
            .as_object_mut()
            .unwrap()
            .remove("multi-turn-repeat");
        assert!(noise_floor_summary(&without)
            .unwrap_err()
            .contains("multi-turn-repeat"));
        let chunked = &summary["controls"]["chunked-prefill"];
        assert_eq!(chunked["minAgreement"], 0.998);
        assert_eq!(chunked["minAgreementRow"], "b");
        assert_eq!(chunked["maxFlipCount"], 2);
        assert_eq!(chunked["maxAbsPerplexityDelta"], 0.013);
        assert_eq!(chunked["maxAbsPerplexityDeltaRow"], "b");
        assert_eq!(chunked["greedyThresholdWithinDenseFloor"], true);
        assert_eq!(chunked["perplexityThresholdWithinDenseFloor"], true);
        let repeat = &summary["controls"]["one-shot-repeat"];
        assert_eq!(repeat["minAgreement"], 1.0);
        assert_eq!(repeat["greedyThresholdWithinDenseFloor"], false);
        assert_eq!(repeat["perplexityThresholdWithinDenseFloor"], false);
        assert!(noise_floor_summary(&[]).is_err());
        assert!(noise_floor_summary(&[serde_json::json!({"coordinate": "a"})]).is_err());
    }

    /// `noise-floor-parent` resumes sealed rows, stops between rows on a stop file, refuses a row
    /// that does not match its seal, and publishes the sealed summary by renaming the resume dir.
    #[test]
    fn noise_floor_parent_resumes_sealed_rows_and_publishes_the_summary() {
        let root = tempfile::tempdir().unwrap();
        let resume = root.path().join("resume");
        let out = root.path().join("out");
        fs::create_dir_all(&resume).unwrap();
        let prompt = root.path().join("prompt.txt");
        fs::write(&prompt, "baseline").unwrap();
        let policy =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../.github/kv-poc/policies/llm.json");
        let args = |extra: &[&str]| {
            let mut args = [
                "noise-floor-parent",
                "--llama-snapshot",
                "/nonexistent/llama",
                "--qwen-snapshot",
                "/nonexistent/qwen",
                "--llama-fp32-reference-snapshot",
                "/nonexistent/llama-bf16",
                "--qwen-fp32-reference-snapshot",
                "/nonexistent/qwen-bf16",
                "--prompt-file",
                prompt.to_str().unwrap(),
                "--safety-policy",
                policy.to_str().unwrap(),
                "--resume-dir",
                resume.to_str().unwrap(),
                "--out",
                out.to_str().unwrap(),
            ]
            .map(String::from)
            .to_vec();
            args.extend(extra.iter().map(|arg| arg.to_string()));
            args
        };
        let seal_row = |slug: &str| {
            let (bytes, sha256) = seal_json(&noise_floor_row(slug, (1.0, 0, 0.001))).unwrap();
            fs::write(resume.join(format!("{slug}.json")), bytes).unwrap();
            fs::write(
                resume.join(format!("{slug}.json.sha256")),
                format!("{sha256}  {slug}.json\n"),
            )
            .unwrap();
        };
        let slugs = required_coordinates()
            .iter()
            .map(coordinate_slug)
            .collect::<Vec<_>>();
        // Row 0 sealed, row 1 missing: a stop file halts before row 1 spawns any worker.
        seal_row(&slugs[0]);
        fs::write(resume.join(OPERATOR_STOP_FILE_NAME), b"").unwrap();
        let CampaignOutcome::StoppedByOperator(stop) = sc20671_cli(&args(&[])).unwrap() else {
            panic!("a stop file must halt between rows");
        };
        assert_eq!((stop.before_row, stop.row.as_str()), (1, slugs[1].as_str()));
        fs::remove_file(resume.join(OPERATOR_STOP_FILE_NAME)).unwrap();
        // A row whose pinned snapshot is absent is refused before admission, naming the path.
        let missing = sc20671_cli(&args(&[])).unwrap_err();
        assert!(
            missing.contains(&slugs[1]) && missing.contains("/nonexistent/llama/"),
            "{missing}"
        );
        // A row is priced and checked on the one dense candidate session its worker loads: the
        // bf16 reference is never read (the 650993bc5 path read its shards), so its flag is optional.
        let mut unreferenced = args(&[]);
        let flag = unreferenced
            .iter()
            .position(|arg| arg == "--llama-fp32-reference-snapshot")
            .unwrap();
        unreferenced.drain(flag..flag + 2);
        let unreferenced = sc20671_cli(&unreferenced).unwrap_err();
        assert!(
            unreferenced.contains("/nonexistent/llama/") && !unreferenced.contains("bf16"),
            "{unreferenced}"
        );
        for slug in &slugs[1..] {
            seal_row(slug);
        }
        // A row whose bytes no longer match its seal is refused.
        fs::write(resume.join(format!("{}.json", slugs[3])), b"{}").unwrap();
        assert!(sc20671_cli(&args(&[]))
            .unwrap_err()
            .contains("does not match its seal"));
        seal_row(&slugs[3]);
        assert_eq!(sc20671_cli(&args(&[])).unwrap(), CampaignOutcome::Completed);
        assert!(!resume.exists());
        let bytes = fs::read(out.join("summary.json")).unwrap();
        assert_eq!(
            fs::read_to_string(out.join("summary.json.sha256")).unwrap(),
            format!("{}  summary.json\n", seal_bytes(&bytes))
        );
        let summary: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(summary["kind"], NOISE_FLOOR_SUMMARY_KIND);
        assert_eq!(summary["rows"], slugs.len());
        assert!(summary["onlyCoordinate"].is_null());
        // A published destination is never overwritten.
        assert!(sc20671_cli(&args(&[]))
            .unwrap_err()
            .contains("must be absent"));
        // `--only-coordinate` (kv-poc nf_only_coordinate) runs and publishes that one row.
        fs::create_dir_all(&resume).unwrap();
        seal_row(&slugs[3]);
        let with_out = |extra: &[&str], out: &Path| {
            let mut argv = args(extra);
            let position = argv.iter().position(|arg| arg == "--out").unwrap();
            argv[position + 1] = out.to_str().unwrap().into();
            argv
        };
        let only_out = root.path().join("only");
        assert_eq!(
            sc20671_cli(&with_out(&["--only-coordinate", &slugs[3]], &only_out)).unwrap(),
            CampaignOutcome::Completed
        );
        let only_summary: serde_json::Value =
            serde_json::from_slice(&fs::read(only_out.join("summary.json")).unwrap()).unwrap();
        assert_eq!(only_summary["rows"], 1);
        assert_eq!(only_summary["onlyCoordinate"], slugs[3].as_str());
        fs::create_dir_all(&resume).unwrap();
        assert!(sc20671_cli(&with_out(
            &["--only-coordinate", "llama-unknown"],
            &root.path().join("unknown")
        ))
        .unwrap_err()
        .contains("not a scheduled coordinate"));
    }

    /// Byte-exact inference copy of SceneWorks `config/kv-baseline-quality-contract.json`: its
    /// hash is the receipt contract identity, so producer wording and thresholds cannot drift.
    #[test]
    fn quality_contract_copy_pins_hash_wording_and_thresholds() {
        let bytes = include_bytes!("../testdata/kv-baseline-quality-contract.json");
        assert_eq!(seal_bytes(bytes), QUALITY_CONTRACT_HASH);
        let contract: serde_json::Value = serde_json::from_slice(bytes).unwrap();
        assert_eq!(contract["version"], 5);
        // Every receipt records the contract's statistics block verbatim (quality measured once,
        // repeats and warmups are timing only).
        assert_eq!(
            contract["statistics"],
            serde_json::to_value(contract_statistics()).unwrap()
        );
        assert_eq!(contract["statistics"]["qualityMeasuredOnce"], true);
        assert_eq!(contract["changeRecord"]["thresholdsUnchanged"], true);
        assert_eq!(
            contract["changeRecord"]["madeAfterCompressedResultsVisible"],
            true
        );
        assert!(contract["changeRecord"]["from"]
            .as_str()
            .is_some_and(|from| from
                .contains("0a00b520d845d4da4c3da9b904af25e08646d87fba018ed6cf27ff9ecce4cc58")));
        assert!(contract["gate"]["nonDiscriminatingNeedle"]
            .as_str()
            .is_some_and(|rule| rule.contains("observation only")));
        assert_eq!(
            contract["thresholds"]["multiTurnPromptCache"],
            COMPRESSED_MULTI_TURN_PROMPT_CACHE_MIN
        );
        assert_eq!(
            contract["multiTurnFixture"]["followUp"],
            MULTI_TURN_FIXTURE_FOLLOW_UP
        );
        for field in ["turns", "metric", "observations"] {
            assert!(contract["multiTurnFixture"][field]
                .as_str()
                .is_some_and(|text| !text.is_empty()));
        }
        assert_eq!(
            contract["gate"]["compressedReference"],
            COMPRESSED_QUALITY_REFERENCE
        );
        assert_eq!(
            contract["thresholds"]["parityMaxError"],
            COMPRESSED_PARITY_MAX_ERROR
        );
        assert_eq!(
            contract["thresholds"]["greedyTokenAgreement"],
            COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN
        );
        assert_eq!(contract["thresholds"]["perplexityDelta"], 0.01);
        let statement = contract["needleFixture"]["statement"].as_str().unwrap();
        let question = contract["needleFixture"]["question"].as_str().unwrap();
        assert_eq!(
            statement,
            format!("{NEEDLE_FIXTURE_STATEMENT_PREFIX}{{needle}}.")
        );
        assert_eq!(question, NEEDLE_FIXTURE_QUESTION);
        let prompt = needle_fixture_prompt("base", "payload", "NEEDLE-1");
        assert!(prompt.contains(&statement.replace("{needle}", "NEEDLE-1")));
        assert!(prompt.ends_with(question));
    }

    #[test]
    fn needle_observation_gates_compressed_rows_only_against_same_weights_dense() {
        let outcomes = |candidate, reference, outputs_match| FixtureOutcomes {
            candidate_needle_recovered: candidate,
            reference_needle_recovered: reference,
            needle_outputs_match: outputs_match,
            ..FixtureOutcomes::default()
        };
        use QualityReference::{Bf16Characterization as Dense, DenseKvSameWeights as Compressed};
        assert_eq!(
            needle_observation(&outcomes(false, true, false), Dense),
            (false, false)
        );
        assert_eq!(
            needle_observation(&outcomes(true, false, false), Dense),
            (true, true)
        );
        assert_eq!(
            needle_observation(&outcomes(true, true, true), Compressed),
            (true, true)
        );
        assert_eq!(
            needle_observation(&outcomes(false, true, false), Compressed),
            (false, true)
        );
        // Contract v5: after a shared dense miss, agreement with the dense miss text is no longer
        // retrieval; the candidate's own recovery is the (ungated) observation.
        assert_eq!(
            needle_observation(&outcomes(false, false, true), Compressed),
            (false, false)
        );
        assert_eq!(
            needle_observation(&outcomes(true, false, false), Compressed),
            (true, false)
        );
        assert_eq!(
            needle_observation(&outcomes(false, false, false), Compressed),
            (false, false)
        );
        assert_eq!(
            fixture_independent_reference("long-context-needle", Compressed, "abc"),
            "dense-kv-same-weights:abc"
        );
        assert_eq!(
            fixture_independent_reference("kernel-fp32-reference", Compressed, "abc"),
            "host-fp32-dense-attention-v1;dense-kv-same-weights:abc"
        );
        let prompt = needle_fixture_prompt("base", "payload", "NEEDLE-1");
        assert!(prompt.contains("The special magic identifier is NEEDLE-1."));
        assert!(prompt.ends_with(NEEDLE_FIXTURE_QUESTION));
        assert!(!prompt.to_ascii_lowercase().contains("passphrase"));
    }

    #[test]
    fn frozen_fixture_disables_thinking_before_bounded_generation() {
        let request = fixture_request("fixture".into(), vec![structured_fixture_tool()]);
        assert_eq!(request.thinking, ThinkingMode::Disabled);
        assert_eq!(request.max_new_tokens, 64);
        assert_eq!(request.tools.len(), 1);
    }

    #[test]
    fn tool_only_generation_is_observed_and_bound_by_structured_output() {
        let mut arguments = serde_json::Map::new();
        arguments.insert(
            "fact".into(),
            serde_json::json!("SC20671 structured fixture"),
        );
        let output = TextLlmOutput {
            tool_calls: vec![core_llm::ToolCall::new("record_baseline_fact", arguments)],
            usage: core_llm::Usage {
                prompt_tokens: 10,
                generated_tokens: 5,
            },
            finish_reason: Some(core_llm::FinishReason::Stop),
            ..Default::default()
        };
        assert!(output_has_observed_generation(&output, false));
        let mut changed = output.clone();
        changed.tool_calls[0]
            .arguments
            .insert("fact".into(), serde_json::json!("different"));
        assert_ne!(
            product_output_digest(&output),
            product_output_digest(&changed)
        );

        let no_stream_or_tool = TextLlmOutput {
            usage: output.usage,
            ..Default::default()
        };
        assert!(!output_has_observed_generation(&no_stream_or_tool, false));
        assert!(!output_has_observed_generation(
            &TextLlmOutput::default(),
            true
        ));
    }

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
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        let first = inventory_snapshot(dir.path()).unwrap();
        let second = inventory_snapshot(dir.path()).unwrap();
        assert_eq!(first.sha256, second.sha256);
        assert_eq!(first.files[0].path, "a.safetensors");
    }

    #[test]
    fn benchmark_model_selection_is_the_four_exact_public_pins() {
        let llama = benchmark_model("llama", false).unwrap();
        let llama_reference = benchmark_model("llama", true).unwrap();
        let qwen = benchmark_model("qwen", false).unwrap();
        let qwen_reference = benchmark_model("qwen", true).unwrap();
        assert_eq!(llama.repository, "mlx-community/Llama-3.2-3B-Instruct-4bit");
        assert_eq!(llama.revision, "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e");
        assert_eq!(
            llama_reference.repository,
            "mlx-community/Llama-3.2-3B-Instruct-bf16"
        );
        assert_eq!(
            llama_reference.revision,
            "6d88ba43024fef71b10e52e101c7cd4598322601"
        );
        assert_eq!(qwen.repository, "mlx-community/Qwen3-1.7B-4bit");
        assert_eq!(qwen.revision, "3b1b1768f8f8cf8351c712464f906e86c2b8269e");
        assert_eq!(qwen_reference.repository, "mlx-community/Qwen3-1.7B-bf16");
        assert_eq!(
            qwen_reference.revision,
            "9cd6692855d3e06772228e9a962b2606359b2d24"
        );
        assert!(llama.native_context_tokens >= SC20671_MIN_NATIVE_CONTEXT_TOKENS);
        assert!(qwen.native_context_tokens >= SC20671_MIN_NATIVE_CONTEXT_TOKENS);
        assert_ne!(llama.revision, llama_reference.revision);
        assert_ne!(qwen.revision, qwen_reference.revision);
        assert!(
            benchmark_model("mistral", false).is_err(),
            "wrong family must fail closed"
        );
    }

    #[test]
    fn benchmark_snapshot_rejects_substitution_family_and_reference_mismatch() {
        const FILES: &[PinnedSnapshotFile] = &[
            PinnedSnapshotFile {
                path: "config.json",
                bytes: 124,
                sha256: "f165c79f5f1092f615d77da1b2eb8b436e399683be472e9d90af462ccdb10b34",
            },
            PinnedSnapshotFile {
                path: "model.safetensors",
                bytes: 7,
                sha256: "9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c",
            },
        ];
        let candidate = BenchmarkModelSpec {
            family: "llama",
            role: "candidate",
            repository: "example/candidate",
            revision: "a",
            architecture: "LlamaForCausalLM",
            model_type: "llama",
            native_context_tokens: 32_768,
            quantized: true,
            required_files: FILES,
        };
        let reference = BenchmarkModelSpec {
            role: "fp32-reference",
            quantized: false,
            ..candidate
        };
        let wrong_family = BenchmarkModelSpec {
            family: "qwen",
            architecture: "Qwen3ForCausalLM",
            model_type: "qwen3",
            ..candidate
        };
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("config.json"),
            br#"{"architectures":["LlamaForCausalLM"],"model_type":"llama","max_position_embeddings":32768,"quantization_config":{"bits":4}}"#,
        )
        .unwrap();
        fs::write(dir.path().join("model.safetensors"), b"weights").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"tokenizer").unwrap();
        assert!(validate_benchmark_snapshot(dir.path(), &candidate).is_ok());
        // The source-owned config and precision arm catch a caller swapping candidate/reference
        // or family paths even when the snapshot directory itself is valid.
        assert!(validate_benchmark_snapshot(dir.path(), &reference).is_err());
        assert!(validate_benchmark_snapshot(dir.path(), &wrong_family).is_err());
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        assert!(
            validate_benchmark_snapshot(dir.path(), &candidate).is_err(),
            "config hash must reject caller edits"
        );
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
    fn phase_samples_name_the_syscall_and_legacy_footprint_receipts_stay_valid() {
        assert_eq!(PHYS_FOOTPRINT_SOURCE, "proc_pid_rusage");
        assert!(valid_phys_footprint_source(PHYS_FOOTPRINT_SOURCE));
        assert!(valid_phys_footprint_source("footprint -p"));
        for other in ["", "footprint", "mlx_rs::memory", "proc_pid_rusage "] {
            assert!(!valid_phys_footprint_source(other), "{other:?}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn worker_samples_its_own_footprint_without_a_subprocess() {
        let pid = std::process::id();
        let sample = sample_memory(pid).unwrap();
        assert_eq!(sample.pid, pid);
        std::num::NonZeroU64::new(sample.current_bytes).expect("own phys_footprint is zero");
        assert!(sample.peak_bytes >= sample.current_bytes);
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
    fn compile_dispatch_excludes_slow_observer_sampling_callbacks() {
        let mut observer = ProductObserver::new();
        let wall_started = std::time::Instant::now();
        let (_, product_ms) = measure_product_dispatch(&mut observer, |observer| {
            std::thread::sleep(std::time::Duration::from_millis(2));
            let sampling_started = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(30));
            observer.sampling_elapsed_ms += sampling_started.elapsed().as_secs_f64() * 1_000.0;
            std::thread::sleep(std::time::Duration::from_millis(2));
            Ok(())
        })
        .unwrap();
        let wall_ms = wall_started.elapsed().as_secs_f64() * 1_000.0;
        assert!(observer.sampling_elapsed_ms >= 25.0);
        assert!(product_ms > 0.0);
        assert!(product_ms < observer.sampling_elapsed_ms);
        assert!((product_ms + observer.sampling_elapsed_ms - wall_ms).abs() < 10.0);
    }

    #[test]
    fn operation_evidence_digest_is_canonical_for_realistic_floats() {
        let realistic = 14.162040999999993;
        let first = serde_json::json!({
            "operation": "single-shot-generation",
            "compileSetupMs": 0.0,
            "compileDispatchMs": realistic,
        });
        let mut reversed = serde_json::Map::new();
        reversed.insert("compileDispatchMs".into(), serde_json::json!(realistic));
        reversed.insert("compileSetupMs".into(), serde_json::json!(0.0));
        reversed.insert(
            "operation".into(),
            serde_json::json!("single-shot-generation"),
        );
        let second = serde_json::Value::Object(reversed);
        let expected = seal_bytes(&canonical_json_bytes(&first).unwrap());
        assert_eq!(canonical_operation_evidence_sha256(&first), expected);
        assert_eq!(canonical_operation_evidence_sha256(&second), expected);
        let (noncanonical_bytes, _) = sealed_json(&first);
        assert_ne!(expected, seal_bytes(&noncanonical_bytes));
    }

    #[test]
    fn semantic_seal_normalizes_numbers_to_exact_ieee_bits() {
        let realistic = 14.162040999999993_f64;
        let first = serde_json::json!({
            "integer": 1,
            "realistic": realistic,
            "zero": 0.0,
        });
        let mut reversed = serde_json::Map::new();
        reversed.insert("zero".into(), serde_json::json!(0));
        reversed.insert("realistic".into(), serde_json::json!(realistic));
        reversed.insert("integer".into(), serde_json::json!(1.0));
        let second = serde_json::Value::Object(reversed);
        let bytes = canonical_semantic_seal_bytes(&first).unwrap();
        let rendered = String::from_utf8(bytes.clone()).unwrap();
        assert!(rendered.contains(&format!("f64:{:016x}", realistic.to_bits())));
        assert!(rendered.contains("f64:3ff0000000000000"));
        assert!(rendered.contains("f64:0000000000000000"));
        assert_eq!(bytes, canonical_semantic_seal_bytes(&second).unwrap());
        assert_eq!(
            seal_bytes(&bytes),
            "d79227844f5a6cb8c3def3ce381ba7a79d28f5efb5b177e674b185f719363b56",
            "the Rust seal must match the paired JavaScript fixture"
        );
        let parsed: serde_json::Value =
            serde_json::from_str(r#"{"value":91.01400000000001}"#).unwrap();
        assert_eq!(
            parsed["value"].as_f64().unwrap().to_bits(),
            0x4056_c0e5_6041_8938,
            "receipt parsing must retain the exact number published for the JS consumer"
        );
    }

    #[test]
    fn product_observer_preserves_live_and_allocated_cache_geometry() {
        let mut observer = ProductObserver::new();
        Observer::geometry(
            &mut observer,
            ProductGeometry {
                query_heads: 8,
                kv_heads: 2,
                head_dimension: 64,
                layers: 4,
                element_bytes: 0,
            },
        );
        observer.phase = Some("prefill-peak");
        Observer::cache_snapshot(&mut observer, 4096, 2, 256, 2);
        assert_eq!(observer.geometry.unwrap().element_bytes, 2);
        assert_eq!(observer.cache_live_tokens, 2);
        assert_eq!(observer.cache_capacity_tokens, 256);
        assert!(observer.error.is_none());
    }

    #[test]
    fn product_observer_rejects_capacity_below_live_length() {
        let mut observer = ProductObserver::new();
        observer.phase = Some("prefill-peak");
        Observer::cache_snapshot(&mut observer, 4096, 2, 1, 2);
        assert_eq!(
            observer.error.as_deref(),
            Some("product cache snapshot must have positive bytes, valid live/capacity tokens, and element width")
        );
    }

    #[test]
    fn product_observer_rejects_cache_element_width_changes() {
        let mut observer = ProductObserver::new();
        Observer::geometry(
            &mut observer,
            ProductGeometry {
                query_heads: 8,
                kv_heads: 2,
                head_dimension: 64,
                layers: 4,
                element_bytes: 0,
            },
        );
        observer.phase = Some("prefill-peak");
        Observer::cache_snapshot(&mut observer, 4096, 2, 2, 2);
        Observer::cache_snapshot(&mut observer, 8192, 4, 4, 4);
        assert_eq!(
            observer.error.as_deref(),
            Some("product cache element width changed within one coordinate")
        );
    }

    /// sc-20671: an F32-promoted dense cache (the pre-fix F16-scale Llama path) fails the row closed
    /// with a reason, at the live observer and again at receipt acceptance.
    #[test]
    fn a_kv_width_other_than_the_compute_dtype_is_refused() {
        let mut observer = ProductObserver::new();
        Observer::geometry(
            &mut observer,
            ProductGeometry {
                query_heads: 8,
                kv_heads: 2,
                head_dimension: 64,
                layers: 4,
                element_bytes: 0,
            },
        );
        observer.phase = Some("prefill-peak");
        Observer::cache_snapshot(&mut observer, 8192, 2, 256, 4);
        assert_eq!(
            observer.error.as_deref(),
            Some(dense_kv_width_refusal(4).as_str())
        );
        let reason = observer.finish().err().unwrap();
        assert!(reason.contains("sc-20671"), "{reason}");
        assert!(
            reason.contains("observed as Float32 dtype (4 bytes/element)"),
            "{reason}"
        );
        assert!(
            reason.contains("expected the Bfloat16 compute dtype (2 bytes/element)"),
            "{reason}"
        );

        let receipt = builder_test_receipt();
        assert_eq!(
            receipt.geometry.element_bytes,
            DENSE_KV_COMPUTE_ELEMENT_BYTES
        );
        let mut widened = receipt.clone();
        widened.geometry.element_bytes = 4;
        assert_eq!(
            validate_receipt_semantics(&widened).unwrap_err(),
            dense_kv_width_refusal(4)
        );
    }

    #[test]
    fn required_covering_set_is_exactly_eight_coordinates() {
        let coordinates = required_coordinates();
        assert_eq!(coordinates.len(), 8);
        assert_eq!(
            coordinates.iter().map(coordinate_slug).collect::<Vec<_>>(),
            [
                "llama-short-single-chunked-cold",
                "llama-medium-supported-batch-single-shot-warm",
                "llama-memory-material-single-single-shot-warm",
                "llama-fit-boundary-single-chunked-cold",
                "qwen-short-single-single-shot-cold",
                "qwen-medium-supported-batch-chunked-warm",
                "qwen-memory-material-single-single-shot-warm",
                "qwen-fit-boundary-single-chunked-cold",
            ]
        );
        assert_eq!(legacy_required_schedule().len(), 64);
        for (index, coordinate) in coordinates.iter().enumerate() {
            assert!(!coordinates[index + 1..].contains(coordinate));
        }
    }

    /// `--only-coordinate` selects exactly one scheduled row (an unscheduled name is refused), its
    /// publication must match the resume identity's filter and hold exactly that row, and a
    /// partial-run manifest is never loadable as a complete campaign.
    #[test]
    fn only_coordinate_runs_one_row_and_never_publishes_a_campaign() {
        const SLUG: &str = "llama-memory-material-single-single-shot-warm";
        assert_eq!(
            selected_schedule_indices(None).unwrap(),
            (0..8).collect::<Vec<_>>()
        );
        let selected = selected_schedule_indices(Some(SLUG)).unwrap();
        assert_eq!(selected, vec![2]);
        assert_eq!(coordinate_slug(&required_schedule()[2].coordinate), SLUG);
        assert!(selected_schedule_indices(Some("llama-32k"))
            .unwrap_err()
            .contains("not a scheduled coordinate"));
        let args = |values: &[&str]| values.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        assert_eq!(
            optional_flag(&args(&["parent"]), "--only-coordinate").unwrap(),
            None
        );
        assert_eq!(
            optional_flag(
                &args(&["parent", "--only-coordinate", SLUG]),
                "--only-coordinate"
            )
            .unwrap()
            .as_deref(),
            Some(SLUG)
        );
        for bad in [&["--only-coordinate"][..], &["--only-coordinate", "--out"]] {
            assert!(
                optional_flag(&args(bad), "--only-coordinate").is_err(),
                "{bad:?}"
            );
        }

        let policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1024,
            child_footprint_cap_bytes: 2048,
            max_context_tokens: 4096,
            max_request_tokens: 4096,
            stdout_cap_bytes: 4096,
            stderr_cap_bytes: 4096,
        };
        let root = tempfile::tempdir().unwrap();
        let destination = root.path().join("partial");
        let full = serde_json::json!({ "kind": "sc-20671-resume-identity" });
        let mut partial = full.clone();
        partial["onlyCoordinate"] = SLUG.into();
        // The publication scope must be the identity's: neither can stand in for the other.
        for (identity, only) in [(&full, Some(SLUG)), (&partial, None)] {
            assert_eq!(
                publish_campaign(&destination, &[], identity, &policy, only).unwrap_err(),
                "publication scope differs from the resume identity's coordinate filter"
            );
        }
        assert_eq!(
            publish_campaign(&destination, &[], &partial, &policy, Some(SLUG)).unwrap_err(),
            format!("partial run of {SLUG} requires exactly its one prepared receipt")
        );
        assert!(!destination.exists());

        // A partial manifest (correctly sealed) is refused by the complete-campaign loader.
        fs::create_dir(&destination).unwrap();
        let manifest = canonical_json_bytes(&serde_json::json!({
            "schemaVersion": 2,
            "kind": SC20671_PARTIAL_RUN_KIND,
            "scheduleVersion": SC20671_SCHEDULE_VERSION,
            "partial": true,
            "publishable": false,
            "onlyCoordinate": SLUG,
            "coordinates": [],
        }))
        .unwrap();
        fs::write(destination.join("campaign.json"), &manifest).unwrap();
        fs::write(
            destination.join("campaign.json.sha256"),
            format!("{}  campaign.json\n", seal_bytes(&manifest)),
        )
        .unwrap();
        assert_eq!(
            load_validated_complete_campaign(&destination).unwrap_err(),
            "SC-20671 campaign manifest identity is invalid"
        );
    }

    /// One real prepared row published as a partial run: the manifest is marked partial and
    /// non-publishable, binds its one row, and the complete-campaign loader refuses it.
    #[test]
    fn a_real_prepared_row_publishes_as_a_partial_run_the_campaign_loader_refuses() {
        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS);
        let suites = compressed_test_suites(&continuation, &[]);
        let quality = receipt_quality_over_repeats(&suites).unwrap();
        let mut receipt = compressed_test_builder(quality).finish().unwrap();
        // The builder's receipt is llama short single-shot cold; the schedule's single-shot cold
        // short row is qwen's.
        receipt.matrix.family = "qwen".into();
        receipt.timings.compile_attribution.probe_evidence = test_compile_probe_evidence(
            "single-shot-generation",
            "measured-repeats",
            "qwen-short-single-single-shot-cold",
            &[3.0, 1.0, 1.0, 1.0, 1.0],
        );
        let matrix = &receipt.matrix;
        let coordinate = required_coordinates()
            .into_iter()
            .find(|coordinate| {
                coordinate_slug(coordinate)
                    == format!(
                        "{}-{}-{}-{}-{}",
                        matrix.family,
                        matrix.context_band,
                        matrix.request_mode,
                        matrix.prefill_mode,
                        matrix.process_temperature
                    )
            })
            .expect("the test receipt is a scheduled coordinate");
        // The test suites carry placeholder producer identities; bind them to this receipt's
        // provenance (as one product session's halves are) and reseal each artifact.
        let fixtures = sealed_product_fixture_artifacts(&suites, &coordinate)
            .unwrap()
            .into_iter()
            .map(|artifact| {
                let mut value: serde_json::Value = serde_json::from_slice(&artifact.bytes).unwrap();
                let repeat = value["binding"]["repeat"].as_u64().unwrap() as usize;
                let provenance = &receipt.provenance;
                for (role, session, inventory) in [
                    (
                        "candidate",
                        provenance.campaign_session_id.clone(),
                        provenance.model_file_sha256.clone(),
                    ),
                    (
                        "reference",
                        "f".repeat(64),
                        provenance.model_file_sha256.clone(),
                    ),
                ] {
                    let arm = value["binding"][role].as_object_mut().unwrap();
                    for key in ["coordinateSessionId", "qualitySessionId"] {
                        arm.insert(key.into(), session.clone().into());
                    }
                    for key in ["coordinateInventorySha256", "qualityInventorySha256"] {
                        arm.insert(key.into(), inventory.clone().into());
                    }
                    // This row's operation (single, single-shot).
                    arm.insert("operation".into(), "single-shot-generation".into());
                }
                let kernel = value["fixture"] == "kernel-fp32-reference";
                let candidate = value["binding"]["candidate"].as_object_mut().unwrap();
                if kernel {
                    candidate.insert(
                        "coordinateEvidenceSha256".into(),
                        receipt
                            .provenance
                            .coordinate_operation_sha256
                            .clone()
                            .into(),
                    );
                }
                let probe = &receipt.timings.compile_attribution.probe_evidence[repeat];
                candidate.insert("compileSetupMs".into(), probe.setup_ms.into());
                candidate.insert("compileDispatchMs".into(), probe.dispatch_ms.into());
                candidate.insert(
                    "operationEvidenceSha256".into(),
                    probe.operation_evidence_sha256.clone().into(),
                );
                let bytes = canonical_json_bytes(&value).unwrap();
                SealedFixtureArtifact {
                    sidecar: format!("{}  {}\n", seal_bytes(&bytes), artifact.name),
                    name: artifact.name,
                    bytes,
                }
            })
            .collect::<Vec<_>>();
        // Bind the receipt's fixture evidence to the sealed artifacts, as `product_receipt` does.
        for name in REQUIRED_FIXTURES {
            let artifact = fixtures
                .iter()
                .find(|artifact| artifact.name == format!("fixtures/{name}.json"))
                .unwrap();
            let evidence = receipt.quality.fixture_evidence.get_mut(name).unwrap();
            evidence.artifact_name = artifact.name.clone();
            evidence.artifact_sha256 = seal_bytes(&artifact.bytes);
            evidence.artifact_sidecar_sha256 = seal_bytes(artifact.sidecar.as_bytes());
        }
        let bundle = assemble_artifacts_with_fixtures(receipt, fixtures).unwrap();
        validate_artifact_bundle(&bundle).unwrap();
        let prepared = [PreparedCoordinateReceipt {
            coordinate: coordinate.clone(),
            bundle,
        }];
        let policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1 << 30,
            child_footprint_cap_bytes: 1 << 30,
            max_context_tokens: 1 << 20,
            max_request_tokens: 1 << 20,
            stdout_cap_bytes: 4096,
            stderr_cap_bytes: 4096,
        };
        let slug = coordinate_slug(&coordinate);
        let identity = serde_json::json!({
            "schemaVersion": 1, "kind": "sc-20671-resume-identity",
            "scheduleVersion": SC20671_SCHEDULE_VERSION,
            "coordinates": required_coordinates().iter().map(coordinate_slug).collect::<Vec<_>>(),
            "policySha256": policy.seal().unwrap(),
            "onlyCoordinate": slug,
        });
        let root = tempfile::tempdir().unwrap();
        // The full-campaign path refuses a single row; the partial path publishes it.
        let full = root.path().join("full");
        let mut full_identity = identity.clone();
        full_identity
            .as_object_mut()
            .unwrap()
            .remove("onlyCoordinate");
        assert_eq!(
            publish_campaign(&full, &prepared, &full_identity, &policy, None).unwrap_err(),
            "complete campaign requires exactly eight prepared receipts"
        );
        let destination = root.path().join("partial");
        publish_campaign(&destination, &prepared, &identity, &policy, Some(&slug)).unwrap();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(destination.join("campaign.json")).unwrap()).unwrap();
        assert_eq!(manifest["kind"], SC20671_PARTIAL_RUN_KIND);
        assert_eq!(manifest["partial"], true);
        assert_eq!(manifest["publishable"], false);
        assert_eq!(manifest["onlyCoordinate"], slug.as_str());
        assert_eq!(manifest["coordinates"].as_array().unwrap().len(), 1);
        assert_eq!(manifest["coordinates"][0]["coordinate"], slug.as_str());
        assert!(destination.join(&slug).join("receipt.json").is_file());
        assert_eq!(
            load_validated_complete_campaign(&destination).unwrap_err(),
            "SC-20671 campaign manifest identity is invalid"
        );
        // The one outcome must be its selected row.
        let other = root.path().join("other");
        let mut wrong = identity.clone();
        let other_slug = required_coordinates()
            .iter()
            .map(coordinate_slug)
            .find(|other| *other != slug)
            .unwrap();
        wrong["onlyCoordinate"] = other_slug.as_str().into();
        assert!(publish_campaign(&other, &prepared, &wrong, &policy, Some(&other_slug)).is_err());
        assert!(!other.exists());
    }

    #[test]
    fn safety_policy_and_resume_identity_hashes_match_cross_language_fixture() {
        let policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1024,
            child_footprint_cap_bytes: 2048,
            max_context_tokens: 4096,
            max_request_tokens: 4096,
            stdout_cap_bytes: 4096,
            stderr_cap_bytes: 4096,
        };
        policy.validate().unwrap();
        assert_eq!(
            policy.seal().unwrap(),
            "03b18fb4b6e8e729189ad1243fecc31934ff1b4aa011cdf00fb6489029a17d2a"
        );
        let identity = serde_json::json!({
            "schemaVersion": 1, "kind": "sc-20671-resume-identity", "scheduleVersion": 2,
            "coordinates": required_coordinates().iter().map(coordinate_slug).collect::<Vec<_>>(),
            "inferenceRevision": "a".repeat(40), "sceneWorksRevision": "b".repeat(40),
            "executableSha256": "c".repeat(64), "promptSha256": "d".repeat(64),
            "policySha256": policy.seal().unwrap(),
            "llamaCandidate": {"sha256": "e".repeat(64), "bytes": 101},
            "qwenCandidate": {"sha256": "f".repeat(64), "bytes": 102},
            "llamaReference": {"sha256": "1".repeat(64), "bytes": 103},
            "qwenReference": {"sha256": "2".repeat(64), "bytes": 104},
        });
        assert_eq!(
            seal_json(&identity).unwrap().1,
            "a9bcc30bc11a2c86a4057489e1e0c9bc1519f0802b6f576200562406a8d02112"
        );
    }

    #[test]
    fn material_and_fit_rows_are_admitted_by_runtime_guards_not_a_static_peak_proof() {
        let policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1_024,
            child_footprint_cap_bytes: 2_048,
            max_context_tokens: u64::MAX,
            max_request_tokens: u64::MAX,
            stdout_cap_bytes: 100,
            stderr_cap_bytes: 100,
        };
        for coordinate in required_coordinates().into_iter().filter(|coordinate| {
            ["memory-material", "fit-boundary"].contains(&coordinate.context_band)
        }) {
            // No band refusal: the row reaches the same snapshot-backed preflight as short rows.
            let error = static_row_requirements(
                &coordinate,
                Path::new("/nonexistent-candidate"),
                Path::new("/nonexistent-reference"),
                "prompt",
                &policy,
                false,
            )
            .unwrap_err();
            assert!(
                error.contains("load pinned tokenizer for safety preflight"),
                "{error}"
            );
        }

        // Admitted with every runtime guard configured; the stated cap and estimate are recorded.
        let admission = runtime_guarded_admission(&policy, 1_000, SC20671_ESTIMATE_SOURCE).unwrap();
        assert_eq!(
            admission,
            ReceiptAdmission {
                mode: RUNTIME_GUARDED_ADMISSION.into(),
                rule: campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE.into(),
                child_footprint_cap_bytes: 2_048,
                host_free_reserve_bytes: 1_024,
                static_footprint_floor_bytes: 1_000,
                estimate_source: SC20671_ESTIMATE_SOURCE.into(),
                estimate_bytes: 1_000,
                host_memory_components: None,
            }
        );
        // Refused without guards, or when the conservative floor already exceeds the cap.
        for unguarded in [
            CampaignSafetyPolicy {
                child_footprint_cap_bytes: 0,
                ..policy.clone()
            },
            CampaignSafetyPolicy {
                host_free_reserve_bytes: 0,
                ..policy.clone()
            },
            CampaignSafetyPolicy {
                row_deadline_seconds: 0,
                ..policy.clone()
            },
            CampaignSafetyPolicy {
                poll_millis: 0,
                ..policy.clone()
            },
        ] {
            assert!(runtime_guarded_admission(&unguarded, 1_000, SC20671_ESTIMATE_SOURCE).is_err());
        }
        assert!(runtime_guarded_admission(&policy, 2_049, SC20671_ESTIMATE_SOURCE).is_err());

        struct Host {
            free: u64,
            child: u64,
        }
        impl campaign_supervisor::MemoryProbe for Host {
            fn host_memory(
                &mut self,
                _: std::time::Instant,
            ) -> std::io::Result<campaign_supervisor::HostMemory> {
                Ok(campaign_supervisor::HostMemory::from_pages(
                    1,
                    campaign_supervisor::VmStatPages {
                        free: self.free,
                        ..Default::default()
                    },
                )
                .unwrap())
            }
            fn child_footprint_bytes(
                &mut self,
                _: u32,
                _: std::time::Instant,
            ) -> std::io::Result<u64> {
                Ok(self.child)
            }
        }
        let temporary = tempfile::tempdir().unwrap();
        let request = |name: &str| RunRequest {
            context_tokens: 10,
            request_tokens: 10,
            estimate: admission.estimate(),
            stdout_path: temporary.path().join(format!("{name}.stdout.log")),
            stderr_path: temporary.path().join(format!("{name}.stderr.log")),
        };
        let record = |name: &str, failure: campaign_supervisor::Failure| {
            let path = temporary.path().join(format!("{name}.unaccepted.json"));
            write_unaccepted_row_record(
                &path,
                "sc-20671-unaccepted-row",
                name,
                &admission.with_host_memory(failure.host_memory.as_deref().cloned()),
                (
                    &format!("{:?}", failure.reason),
                    &failure.detail,
                    failure.pid,
                    failure.watchdog_host_memory.as_deref(),
                ),
            )
            .unwrap();
            serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap()).unwrap()
        };
        // Insufficient host RAM for the row's estimate plus reserve: refused before spawn,
        // recorded unaccepted.
        let refused = campaign_supervisor::run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request("refused"),
            &policy.supervisor(),
            &mut Host {
                free: 1_000 + 1_024 - 1,
                child: 1,
            },
        )
        .unwrap_err();
        assert_eq!(
            refused.reason,
            campaign_supervisor::StopReason::PreflightMemory
        );
        let refused = record("refused", refused);
        assert_eq!(refused["accepted"], false);
        assert_eq!(refused["outcome"], "refused");
        assert_eq!(refused["admission"]["childFootprintCapBytes"], 2_048);
        assert_eq!(refused["admission"]["staticFootprintFloorBytes"], 1_000);
        // The refusal records the rule, the estimate and the measurement it was refused on.
        assert_eq!(
            refused["admission"]["rule"],
            campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE
        );
        assert_eq!(
            refused["admission"]["estimateSource"],
            SC20671_ESTIMATE_SOURCE
        );
        assert_eq!(refused["admission"]["estimateBytes"], 1_000);
        assert_eq!(
            refused["admission"]["hostMemoryComponents"]["availableBytes"],
            1_000 + 1_024 - 1
        );
        // Admitted on its estimate (far below cap plus reserve), a child that grows past the cap
        // trips the watchdog: an aborted row, never an accepted one.
        let aborted = campaign_supervisor::run_guarded(
            Command::new("/bin/sleep").arg("5"),
            &request("aborted"),
            &policy.supervisor(),
            &mut Host {
                free: 1_000 + 1_024,
                child: 2_049,
            },
        )
        .unwrap_err();
        assert_eq!(
            aborted.reason,
            campaign_supervisor::StopReason::ChildFootprint
        );
        let aborted = record("aborted", aborted);
        assert_eq!(aborted["outcome"], "aborted");
        assert_eq!(aborted["reason"], "ChildFootprint");
        // A pre-spawn refusal leaves only its record (the supervisor removes the logs): the next
        // attempt must not reuse that prefix, and a record-write failure is appended to, never
        // substituted for, the row's own reason.
        let logs = temporary.path().join("logs");
        fs::create_dir_all(&logs).unwrap();
        assert_eq!(
            unused_attempt_prefix(&logs, "row").unwrap(),
            "row.attempt-0"
        );
        fs::write(logs.join("row.attempt-0.unaccepted.json"), b"{}").unwrap();
        assert_eq!(
            unused_attempt_prefix(&logs, "row").unwrap(),
            "row.attempt-1"
        );
        let stop = ("PreflightMemory", "host free 1 bytes", None, None);
        let collided = unaccepted_row_error(
            &logs.join("row.attempt-0.unaccepted.json"),
            "sc-20671-unaccepted-row",
            "row",
            &admission,
            stop,
            "coordinate row stopped (PreflightMemory)".into(),
        );
        assert!(collided.starts_with("coordinate row stopped (PreflightMemory)"));
        assert!(collided.contains("could not be written"), "{collided}");
        let recorded = unaccepted_row_error(
            &logs.join("row.attempt-1.unaccepted.json"),
            "sc-20671-unaccepted-row",
            "row",
            &admission,
            stop,
            "coordinate row stopped (PreflightMemory)".into(),
        );
        assert!(recorded.contains("; not accepted ("), "{recorded}");
        // A host-reserve watchdog abort records the live sample that tripped it.
        let sample = campaign_supervisor::HostMemory::from_pages(
            16_384,
            campaign_supervisor::VmStatPages {
                free: 7,
                ..Default::default()
            },
        )
        .unwrap();
        let tripped = logs.join("row.attempt-2.unaccepted.json");
        write_unaccepted_row_record(
            &tripped,
            "sc-20671-unaccepted-row",
            "row",
            &admission,
            ("HostMemory", "host available", Some(9), Some(&sample)),
        )
        .unwrap();
        let tripped: serde_json::Value =
            serde_json::from_slice(&fs::read(tripped).unwrap()).unwrap();
        assert_eq!(tripped["outcome"], "aborted");
        assert_eq!(
            tripped["watchdogHostMemory"],
            serde_json::to_value(&sample).unwrap()
        );
        assert!(logs.join("row.attempt-1.unaccepted.json").exists());
        // Exactly estimate plus reserve and a child under the cap: admitted and supervised to
        // completion.
        assert!(campaign_supervisor::run_guarded(
            &mut Command::new("/usr/bin/true"),
            &request("admitted"),
            &policy.supervisor(),
            &mut Host {
                free: 1_000 + 1_024,
                child: 1,
            },
        )
        .unwrap()
        .success());
    }

    pub(crate) fn write_stub_dtype_snapshot(
        root: &Path,
        spec: &BenchmarkModelSpec,
        scale_dtype: &str,
    ) {
        fs::create_dir_all(root).unwrap();
        fs::write(
            root.join("config.json"),
            br#"{"torch_dtype":"bfloat16","num_hidden_layers":28,"num_key_value_heads":8,"head_dim":128,"num_attention_heads":24,"hidden_size":3072,"intermediate_size":8192,"vocab_size":128256}"#,
        )
        .unwrap();
        let mut header = serde_json::Map::new();
        for layer in 0..28 {
            for projection in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                let stem = format!("model.layers.{layer}.{projection}");
                header.insert(
                    format!("{stem}.weight"),
                    serde_json::json!({"dtype": if spec.quantized { "U32" } else { "BF16" }}),
                );
                if spec.quantized {
                    for suffix in ["scales", "biases"] {
                        header.insert(
                            format!("{stem}.{suffix}"),
                            serde_json::json!({"dtype": scale_dtype}),
                        );
                    }
                }
            }
        }
        let empty = serde_json::Map::new();
        for (index, required) in spec
            .required_files
            .iter()
            .filter(|file| file.path.ends_with(".safetensors"))
            .enumerate()
        {
            let bytes = serde_json::to_vec(if index == 0 {
                &header
            } else {
                // Sharded references may put all tested tensors in the first shard.
                &empty
            })
            .unwrap();
            let mut file = Vec::from((bytes.len() as u64).to_le_bytes());
            file.extend_from_slice(&bytes);
            fs::write(root.join(required.path), file).unwrap();
        }
    }

    #[test]
    fn impossible_child_cap_refuses_known_model_load_and_kv_before_spawn() {
        let temporary = tempfile::tempdir().unwrap();
        let candidate = temporary.path().join("candidate");
        let reference = temporary.path().join("reference");
        write_stub_dtype_snapshot(&candidate, &LLAMA_CANDIDATE, "F16");
        write_stub_dtype_snapshot(&reference, &LLAMA_REFERENCE, "BF16");
        let mut policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: 1,
            child_footprint_cap_bytes: 2_048,
            max_context_tokens: 4_096,
            max_request_tokens: 4_096,
            stdout_cap_bytes: 4_096,
            stderr_cap_bytes: 4_096,
        };
        let row = &required_coordinates()[0];
        let refusal = static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy)
            .unwrap_err();
        assert!(refusal.contains("child footprint cap 2048 refuses before spawn"));
        policy.child_footprint_cap_bytes = u64::MAX - policy.host_free_reserve_bytes;
        let budget =
            static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy).unwrap();
        assert!(budget > 12_000_000_000); // BF16 reference dominates the sequential roles.
    }

    /// sc-20671: the static floor prices the request's prefill activations beside the load
    /// reservation, the total-live KV and the attention tile. Without them the fit-boundary row's
    /// floor (40.2 GiB for the bf16 reference) sat ~12 GiB under what a 130k-token prefill
    /// alongside a stored prefix actually holds.
    #[test]
    fn static_floor_prices_request_prefill_activations() {
        let temporary = tempfile::tempdir().unwrap();
        let reference = temporary.path().join("reference");
        write_stub_dtype_snapshot(&reference, &LLAMA_REFERENCE, "BF16");
        let payload: u64 = LLAMA_REFERENCE
            .required_files
            .iter()
            .filter(|file| file.path.ends_with(".safetensors"))
            .map(|file| file.bytes)
            .sum();
        // The fit-boundary row's preflight: a ~130.7k-token prefix beside a full request.
        let (total, request) = (261_483_u64, 130_741_u64);
        let floor =
            static_role_footprint_budget(&LLAMA_REFERENCE, &reference, total, request).unwrap();
        // 28 layers x K/V x 8 heads x 128 x BF16.
        let kv = total * 28 * 2 * 8 * 128 * 2;
        let tile = 3 * 8 * request * 24 * 4;
        // One layer's projections/MLP/residuals per request token (3 x 8192 + 8 x 3072 BF16
        // elements) plus one logits row.
        let activations = request * (3 * 8_192 + 8 * 3_072) * 2 + 128_256 * 2;
        assert_eq!(floor, 2 * payload + kv + tile + activations);
        assert!(activations > 11 << 30, "{activations}");
    }

    /// Run 37055710213: the noise floor's worker estimate prices what it holds, once. Its live
    /// tokens are a compressed row's dense reference arm's (forced 1024-token kernel and turn-2
    /// continuations), whose bound already counts stored turn 1 beside turn 2's cache, so the
    /// budget is one dense session over them — never a second K/V history stacked on top (that
    /// double count refused llama fit-boundary at 76.8 GB against the 68 GiB cap).
    #[test]
    fn noise_floor_estimate_prices_one_session_over_its_forced_passes() {
        let temporary = tempfile::tempdir().unwrap();
        let candidate = temporary.path().join("candidate");
        write_stub_dtype_snapshot(&candidate, &LLAMA_CANDIDATE, "BF16");
        let (total, request) = (261_483_u64, 130_741_u64);
        assert_eq!(
            noise_floor_footprint_budget(&LLAMA_CANDIDATE, &candidate, total, request).unwrap(),
            static_role_footprint_budget(&LLAMA_CANDIDATE, &candidate, total, request).unwrap()
        );
        // Live tokens: nf's forced turn-2 continuation (1024 tokens) is priced, not a dense
        // row's 64-token natural answer.
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = serde_json::json!({
            "version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" }, "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": { "<unk>": 0, "context": 1 },
                       "unk_token": "<unk>" },
        });
        fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let spec = BenchmarkModelSpec {
            family: "llama",
            role: "candidate",
            repository: "test/tiny",
            revision: "0",
            architecture: "LlamaForCausalLM",
            model_type: "llama",
            native_context_tokens: 4096,
            quantized: true,
            required_files: &[],
        };
        let coordinate = Coordinate {
            family: "llama",
            context_band: "short",
            request_mode: "single",
            prefill_mode: "single-shot",
            process_temperature: "cold",
        };
        let noise_floor =
            noise_floor_live_tokens(dir.path(), &spec, &coordinate, "baseline prompt").unwrap();
        let dense_row =
            preflight_total_live_tokens(dir.path(), &spec, &coordinate, "baseline prompt", false)
                .unwrap();
        assert!(
            noise_floor.0 > dense_row.0,
            "{noise_floor:?} <= {dense_row:?}"
        );
        assert_eq!(
            noise_floor,
            preflight_total_live_tokens(dir.path(), &spec, &coordinate, "baseline prompt", true)
                .unwrap()
        );
    }

    /// sc-20671: the pinned dense KV width is the BF16 compute dtype's for every role, whatever
    /// dtype the snapshot stores its quantized scales in — an F16-scale candidate no longer plans
    /// (or records) the F32-promoted 4-byte cache. Unsupported scale dtypes still refuse.
    #[test]
    fn pinned_kv_width_is_the_compute_dtype_whatever_the_stored_scale_dtype() {
        let temporary = tempfile::tempdir().unwrap();
        let llama_reference = temporary.path().join("llama-reference");
        let qwen_reference = temporary.path().join("qwen-reference");
        write_stub_dtype_snapshot(&llama_reference, &LLAMA_REFERENCE, "BF16");
        write_stub_dtype_snapshot(&qwen_reference, &QWEN_REFERENCE, "BF16");
        let mut candidates = Vec::new();
        for (spec, name) in [(&LLAMA_CANDIDATE, "llama"), (&QWEN_CANDIDATE, "qwen")] {
            for scale in ["BF16", "F16", "F32"] {
                let path = temporary.path().join(format!("{name}-candidate-{scale}"));
                write_stub_dtype_snapshot(&path, spec, scale);
                candidates.push((spec, path));
            }
        }
        for (spec, path) in candidates.iter().map(|(spec, path)| (*spec, path)).chain([
            (&LLAMA_REFERENCE, &llama_reference),
            (&QWEN_REFERENCE, &qwen_reference),
        ]) {
            assert_eq!(
                pinned_dense_kv_element_bytes(spec, path, 28).unwrap(),
                DENSE_KV_COMPUTE_ELEMENT_BYTES,
                "{} {}",
                spec.repository,
                path.display()
            );
        }
        // The static floor therefore no longer depends on the stored scale dtype.
        let budget = |scale: &str| {
            static_role_footprint_budget(
                &LLAMA_CANDIDATE,
                &temporary.path().join(format!("llama-candidate-{scale}")),
                262_144,
                131_072,
            )
            .unwrap()
        };
        assert_eq!(budget("F16"), budget("BF16"));
        assert_eq!(budget("F32"), budget("BF16"));
        let unsupported = temporary.path().join("unsupported-scales");
        write_stub_dtype_snapshot(&unsupported, &LLAMA_CANDIDATE, "F64");
        assert!(
            pinned_dense_kv_element_bytes(&LLAMA_CANDIDATE, &unsupported, 28)
                .unwrap_err()
                .contains("unsupported pinned scale dtype")
        );
    }

    #[test]
    fn resume_identity_rejects_changed_behavior_and_corrupted_sidecar() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("resume");
        let mut identity = serde_json::json!({"inferenceRevision": "a".repeat(40), "sceneWorksRevision": "b".repeat(40), "executableSha256": "c".repeat(64)});
        let sha = prepare_resume_root(&root, &mut identity).unwrap();
        assert_eq!(prepare_resume_root(&root, &mut identity).unwrap(), sha);
        let mut moved_ref = identity.clone();
        moved_ref["inferenceRevision"] = serde_json::Value::String("d".repeat(40));
        assert_eq!(prepare_resume_root(&root, &mut moved_ref).unwrap(), sha);
        let mut changed_executable = identity.clone();
        changed_executable["executableSha256"] = serde_json::Value::String("e".repeat(64));
        assert!(prepare_resume_root(&root, &mut changed_executable).is_err());
        fs::write(root.join("identity.json.sha256"), "corrupt").unwrap();
        assert!(prepare_resume_root(&root, &mut identity).is_err());
    }

    #[test]
    fn schedule_is_exact_and_rejects_duplicate_or_reused_cold_workers() {
        let schedule = required_schedule();
        assert_eq!(schedule.len(), 8);
        assert_eq!(
            schedule
                .iter()
                .filter(|row| row.discipline == ProcessDiscipline::FreshChild)
                .count(),
            4
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
        assert_eq!(seen.len(), 8);
        assert_eq!(outcomes.len(), 8);
        assert_eq!(
            seen.iter()
                .filter(|(_, discipline)| *discipline == ProcessDiscipline::ReusedWarmWorker)
                .count(),
            4
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
            outcomes: FixtureOutcomes::default(),
            needle_discriminating: true,
            tool_discriminating: true,
            free_running_first_divergence: None,
            greedy_agreement_by_repeat: vec![1.0; QUALITY_MEASUREMENTS],
            repeat_metrics: Vec::new(),
            forced_continuation: None,
            cache_free_running_first_divergence: None,
            cache_matched_prefix_tokens: 1,
            cache_candidate_turns: test_multi_turn(),
            cache_reference_turns: test_multi_turn(),
            multi_turn_forced_continuation: None,
            multi_turn_forced_pass: None,
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

    #[cfg(target_os = "macos")]
    #[test]
    fn dense_kernel_fixture_uses_independent_host_fp32_reference() {
        let errors = dense_kernel_fp32_parity_errors().unwrap();
        assert!(!errors.is_empty());
        assert!(errors.into_iter().fold(0.0_f64, f64::max) <= 0.0001);
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
    fn model_weight_residency_is_a_positive_candidate_only_active_delta() {
        let sample = |timestamp: &str, pid: u32, active: u64, cache: u64| MemorySample {
            captured_at: timestamp.into(),
            pid,
            current_bytes: active + cache + 1_000,
            peak_bytes: active + cache + 1_000,
            mlx_active_bytes: active,
            mlx_cache_bytes: cache,
            mlx_peak_bytes: active,
        };
        let start = sample("2026-01-01T00:00:00.000Z", 7, 500, 900);
        let loaded = sample("2026-01-01T00:00:01.000Z", 7, 532, 1_900);
        assert_eq!(measured_model_weight_bytes(&start, &loaded), Ok(32));

        let no_candidate_delta = sample("2026-01-01T00:00:01.000Z", 7, 500, 9_000);
        assert!(measured_model_weight_bytes(&start, &no_candidate_delta).is_err());
        let lower_active = sample("2026-01-01T00:00:01.000Z", 7, 499, 0);
        assert!(measured_model_weight_bytes(&start, &lower_active).is_err());
        let wrong_pid = sample("2026-01-01T00:00:01.000Z", 8, 532, 0);
        assert!(measured_model_weight_bytes(&start, &wrong_pid).is_err());
        let stale = sample("2025-12-31T23:59:59.000Z", 7, 532, 0);
        assert!(measured_model_weight_bytes(&start, &stale).is_err());
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

    /// A fixed-length steady decode of a 32-token context whose 255 timed tokens took `decode_ms`.
    /// A nominal, unthrottled host state observed at `boundary`.
    fn test_host_state(boundary: &str, captured_at: &str) -> ReceiptHostState {
        ReceiptHostState {
            boundary: boundary.into(),
            captured_at: captured_at.into(),
            power_mode: "automatic".into(),
            thermal_state: "nominal".into(),
            cpu_speed_limit: None,
            pmset_thermal_raw: PMSET_NOMINAL_THERMAL.into(),
            throttled: false,
        }
    }

    fn test_steady_decode(decode_ms: f64) -> SteadyDecodeMeasurement {
        SteadyDecodeMeasurement {
            prompt_tokens: 32,
            generated_tokens: STEADY_DECODE_TOKENS,
            timed_tokens: STEADY_DECODE_TOKENS - 1,
            decode_ms,
            forced_stop_tokens: 3,
        }
    }

    /// A complete, valid dense receipt assembled by the real builder from minimal evidence.
    fn builder_test_receipt() -> Receipt {
        test_receipt_builder()
            .finish()
            .expect("builder must produce a complete v4 receipt")
    }

    /// The dense builder behind [`builder_test_receipt`], before `finish`.
    fn test_receipt_builder() -> ReceiptBuilder {
        let mut template = Receipt {
            schema_version: RECEIPT_SCHEMA_VERSION,
            harness_version: RECEIPT_HARNESS_VERSION.into(),
            run_id: "run".into(),
            captured_at: "2026-01-01T00:00:00Z".into(),
            mode: "dense".into(),
            status: "complete".into(),
            contract_hash: QUALITY_CONTRACT_HASH.into(),
            receipt_sha256: String::new(),
            provenance: ReceiptProvenance {
                scene_works_repository: SCENEWORKS_REPOSITORY.into(),
                inference_repository: INFERENCE_REPOSITORY.into(),
                scene_works_revision: "a".repeat(40),
                inference_revision: "b".repeat(40),
                mlx_version: "0.25.8".into(),
                mlx_source: format!(
                    "git+{PMETAL_MLX_REPOSITORY}?rev={}#{}",
                    "1".repeat(40),
                    "1".repeat(40)
                ),
                mlx_revision: "1".repeat(40),
                dependency_lock_sha256: "c".repeat(64),
                os: "macOS".into(),
                xcode: "xcode".into(),
                hardware: "hardware".into(),
                model_id: "model".into(),
                model_file_sha256: "d".repeat(64),
                model_file_bytes: 100,
                reference_model_id: "reference-model".into(),
                reference_model_sha256: "9".repeat(64),
                reference_model_bytes: 1,
                power_mode: "automatic".into(),
                thermal_state: "nominal".into(),
                host_states: vec![
                    test_host_state("row-start", "2025-12-31T23:59:59.000Z"),
                    test_host_state("row-end", "2026-01-01T00:00:09.000Z"),
                ],
                thermal_changed_during_row: false,
                power_mode_changed_during_row: false,
                command_template: "run --mode {mode}".into(),
                command: "run --mode dense".into(),
                campaign_session_id: "e".repeat(64),
                campaign_cache_state_version: 1,
                coordinate_operation_sha256: "f".repeat(64),
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
                context_window_tokens: 4096,
                context_target_tokens: 32,
                context_payload_tokens: 32,
            },
            memory: ReceiptMemory {
                model_weights_bytes: 1,
                persistent_kv_bytes: 4,
                transient_workspace_bytes: 1,
                dense_theoretical_kv_bytes: 4,
                prefill_peak_window: ReceiptPeakWindow {
                    started_at: "2026-01-01T00:00:01.500Z".into(),
                    baseline_active_bytes: 3,
                    reset_peak_bytes: 0,
                },
                phase_samples: vec![],
                allocation_events: vec![],
                reconciliation: ReceiptReconciliation {
                    expected_dense_kv_bytes: 4,
                    observed_persistent_kv_bytes: 4,
                    tolerance_bytes: 0,
                },
                release: ReceiptRelease {
                    verified: true,
                    phys_footprint_tolerance_bytes: POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES,
                    mlx_active_tolerance_bytes: 0,
                    mlx_cache_tolerance_bytes: 0,
                    mlx_active_residual_bytes: 0,
                    mlx_cache_residual_bytes: 0,
                },
                admission: ReceiptAdmission {
                    mode: RUNTIME_GUARDED_ADMISSION.into(),
                    rule: campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE.into(),
                    child_footprint_cap_bytes: 1 << 30,
                    host_free_reserve_bytes: 1 << 30,
                    static_footprint_floor_bytes: 1 << 20,
                    estimate_source: SC20671_ESTIMATE_SOURCE.into(),
                    estimate_bytes: 1 << 20,
                    host_memory_components: campaign_supervisor::HostMemory::from_pages(
                        16_384,
                        campaign_supervisor::VmStatPages {
                            free: 200_000,
                            speculative: 4_000,
                            purgeable: 100,
                            inactive: 90_000,
                            file_backed: 60_000,
                            anonymous: 40_000,
                            throttled: 0,
                            active: 0,
                        },
                    ),
                },
                dense_kv_share_bps: 0,
                below_memory_material_share: false,
            },
            timings: ReceiptTimings {
                load_ms: 1.0,
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 1.0,
                decode_tokens_per_second: 1.0,
                cold_compile_ms: Some(1.0),
                warm_compile_ms: 1.0,
                compile_attribution: ReceiptCompileAttribution {
                    method: COMPILE_ATTRIBUTION_METHOD.into(),
                    operation: "single-shot-generation".into(),
                    source: "measured-repeats".into(),
                    probe_durations_ms: vec![3.0, 1.0, 1.0, 1.0, 1.0],
                    probe_evidence: test_compile_probe_evidence(
                        "single-shot-generation",
                        "measured-repeats",
                        "llama-short-single-single-shot-cold",
                        &[3.0, 1.0, 1.0, 1.0, 1.0],
                    ),
                    first_dispatch_ms: 3.0,
                    steady_dispatch_ms: 1.0,
                    first_dispatch_excess_ms: 2.0,
                    noise_samples_ms: Some(vec![1.0; 4]),
                    noise_band_ms: Some(0.0),
                    compile_cost_resolved: Some(true),
                    compile_cost_ms: Some(2.0),
                    compile_cost_unresolved_reason: None,
                },
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
                greedy_agreement_method: GREEDY_AGREEMENT_METHOD.into(),
                free_running_first_divergence: None,
                greedy_token_agreement_by_repeat: vec![1.0; QUALITY_MEASUREMENTS],
                structured_tool_agreement: 1.0,
                needle_retrieval: 1.0,
                needle_discriminating: true,
                tool_discriminating: true,
                multi_turn_prompt_cache: 1.0,
                multi_turn_prompt_cache_method: MULTI_TURN_PROMPT_CACHE_METHOD.into(),
                multi_turn_free_running_first_divergence: None,
                multi_turn_matched_prefix_tokens: 1,
                multi_turn_cache: ReceiptMultiTurnCache {
                    candidate: test_multi_turn(),
                    reference: test_multi_turn(),
                },
                multi_turn_forced_continuation: None,
                multi_turn_forced_pass: None,
                statistics: contract_statistics(),
                fixture_evidence: std::collections::BTreeMap::new(),
                quality_gate: None,
                forced_continuation: None,
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
                session_id: String::new(),
                cache_state_version: 0,
            },
            compression: None,
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
        let active_bytes = [2, 3, 8, 7, 7, 7, 3, 3];
        let peak_bytes = [2, 3, 8, 8, 8, 8, 8, 8];
        let phases = REQUIRED_PHASES
            .iter()
            .enumerate()
            .map(|(index, phase)| ReceiptPhase {
                phase: (*phase).into(),
                pid: 7,
                source: PHYS_FOOTPRINT_SOURCE.into(),
                timestamp: format!("2026-01-01T00:00:0{index}.000Z"),
                phys_footprint_bytes: 100,
                phys_footprint_peak_bytes: 100,
                mlx: ReceiptMlx {
                    source: "mlx_rs::memory".into(),
                    active_bytes: active_bytes[index],
                    cache_bytes: 0,
                    peak_bytes: peak_bytes[index],
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
            ReceiptAllocation {
                kind: "dense-kv-allocation".into(),
                role: "cache".into(),
                lifetime: "persistent".into(),
                phase: "decode-steady".into(),
                timestamp: "2026-01-01T00:00:04.100Z".into(),
                bytes: 4,
            },
            ReceiptAllocation {
                kind: "product-cache_release".into(),
                role: "cache".into(),
                lifetime: "released".into(),
                phase: "decode-steady".into(),
                timestamp: "2026-01-01T00:00:04.300Z".into(),
                bytes: 4,
            },
        ];
        let timings = (0..5)
            .map(|repeat| RawTiming {
                load_ms: 1.0,
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 1.0,
                decode_tokens_per_second: 1.0,
                steady_decode: test_steady_decode(255_000.0),
                host_state: Some(test_host_state(
                    TIMING_SAMPLE_HOST_BOUNDARY,
                    &format!("2026-01-01T00:00:08.{:03}Z", 100 + repeat * 10),
                )),
            })
            .collect();
        let compile_attribution = ReceiptCompileAttribution {
            method: COMPILE_ATTRIBUTION_METHOD.into(),
            operation: "single-shot-generation".into(),
            source: "measured-repeats".into(),
            probe_durations_ms: vec![3.0, 1.0, 1.0, 1.0, 1.0],
            probe_evidence: test_compile_probe_evidence(
                "single-shot-generation",
                "measured-repeats",
                "llama-short-single-single-shot-cold",
                &[3.0, 1.0, 1.0, 1.0, 1.0],
            ),
            first_dispatch_ms: 3.0,
            steady_dispatch_ms: 1.0,
            first_dispatch_excess_ms: 2.0,
            noise_samples_ms: Some(vec![1.0; 4]),
            noise_band_ms: Some(0.0),
            compile_cost_resolved: Some(true),
            compile_cost_ms: Some(2.0),
            compile_cost_unresolved_reason: None,
        };
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
            outcomes: FixtureOutcomes::default(),
            needle_discriminating: true,
            tool_discriminating: true,
            free_running_first_divergence: None,
            greedy_agreement_by_repeat: vec![1.0; QUALITY_MEASUREMENTS],
            repeat_metrics: Vec::new(),
            forced_continuation: None,
            cache_free_running_first_divergence: None,
            cache_matched_prefix_tokens: 1,
            cache_candidate_turns: test_multi_turn(),
            cache_reference_turns: test_multi_turn(),
            multi_turn_forced_continuation: None,
            multi_turn_forced_pass: None,
        };
        ReceiptBuilder {
            template,
            phases,
            allocations,
            timings,
            compile_attribution,
            quality,
        }
    }

    /// The same builder row measured in compressed mode (as [`compress_test_receipt`] turns a
    /// finished receipt), with `quality` as its measured repeats.
    fn compressed_test_builder(quality: QualityObservation) -> ReceiptBuilder {
        let mut builder = test_receipt_builder();
        let template = &mut builder.template;
        template.mode = "compressed".into();
        template.provenance.command = template
            .provenance
            .command_template
            .replace("{mode}", "compressed");
        for name in REQUIRED_FIXTURES {
            template
                .quality
                .fixture_evidence
                .get_mut(name)
                .unwrap()
                .independent_reference = fixture_independent_reference(
                name,
                QualityReference::DenseKvSameWeights,
                &template.provenance.model_file_sha256,
            );
        }
        template.lifecycle = receipt_lifecycle(true);
        template.compression = Some(test_compression_block());
        let compressed_kv = template.memory.persistent_kv_bytes / 2;
        template.memory.persistent_kv_bytes = compressed_kv;
        for event in &mut builder.allocations {
            if event.role == "cache" && event.lifetime != "transient" {
                event.bytes = compressed_kv;
            }
        }
        builder.quality = quality;
        builder
    }

    fn test_packed_evidence(fused: usize) -> crate::primitives::PackedCacheEvidence {
        crate::primitives::PackedCacheEvidence {
            representation_identity: "sc-20676-packed-group-affine-v1".into(),
            representation_version: 2,
            bits: 2,
            quantization_group_size: 32,
            accepted_direct_calls: fused,
            // A prefill chunk on the NAX kernel and decode steps on the per-row kernel.
            kernel_paths: test_kernel_paths(fused),
            kernel_warmed: true,
            ..Default::default()
        }
    }

    fn test_kernel_paths(fused: usize) -> Vec<crate::primitives::PackedKernelPathEvidence> {
        let path = |kernel: &str, selection: &str, calls: usize| {
            crate::primitives::PackedKernelPathEvidence {
                kernel: kernel.into(),
                selection: selection.into(),
                reason: "test".into(),
                query_dtype: "bfloat16".into(),
                calls: calls as u64,
            }
        };
        match fused {
            0 => Vec::new(),
            1 => vec![path(
                crate::primitives::PACKED_PER_ROW_KERNEL,
                crate::primitives::PACKED_SELECTION_BELOW_MULTI_ROW,
                1,
            )],
            _ => vec![
                path(
                    crate::primitives::PACKED_NAX_KERNEL,
                    crate::primitives::PACKED_SELECTION_NAX,
                    1,
                ),
                path(
                    crate::primitives::PACKED_PER_ROW_KERNEL,
                    crate::primitives::PACKED_SELECTION_BELOW_MULTI_ROW,
                    fused - 1,
                ),
            ],
        }
    }

    const TEST_STORAGE: crate::primitives::CompressedCacheStorage =
        crate::primitives::CompressedCacheStorage {
            device_code_bytes: 1,
            device_metadata_bytes: 1,
            host_payload_bytes: 1,
            tokens: 1,
            element_bytes: 2,
        };

    /// An observation whose coordinate scope holds exactly `evidence`, `dense_fallbacks`, and
    /// `storage` (lifecycle evidence is appended by callers outside the scope).
    fn test_compressed_observation(
        evidence: Vec<crate::primitives::PackedCacheEvidence>,
        dense_fallbacks: &[(&str, &str)],
        storage: Option<crate::primitives::CompressedCacheStorage>,
    ) -> ProductObservations {
        let dense_fallbacks = dense_fallbacks
            .iter()
            .map(|(operation, reason)| ((*operation).into(), (*reason).into()))
            .collect::<Vec<(String, String)>>();
        ProductObservations {
            snapshot: SnapshotInventory {
                root: PathBuf::new(),
                files: Vec::new(),
                bytes: 1,
                sha256: "d".repeat(64),
            },
            geometry: ProductGeometry {
                query_heads: 2,
                kv_heads: 1,
                head_dimension: 64,
                layers: 1,
                element_bytes: 2,
            },
            phases: Vec::new(),
            phase_elapsed_ms: Vec::new(),
            allocations: Vec::new(),
            prefill_logits: Vec::new(),
            token_probabilities: Vec::new(),
            forced_token_probabilities: Vec::new(),
            session_id: "e".repeat(64),
            cache_state_version: 1,
            operations: Vec::new(),
            load_elapsed_ms: 1.0,
            prefill_peak_window: ReceiptPeakWindow {
                started_at: "2026-01-01T00:00:00Z".into(),
                baseline_active_bytes: 1,
                reset_peak_bytes: 0,
            },
            cache_live_tokens: 1,
            cache_capacity_tokens: 256,
            packed_evidence: evidence.clone(),
            dense_fallbacks: dense_fallbacks.clone(),
            coordinate_scope: Some(CoordinateCompressionScope {
                packed_evidence: evidence,
                dense_fallbacks,
                storage_peak: storage,
            }),
        }
    }

    /// The producer's block for a primary single-shot coordinate that stayed compressed (its own
    /// lifecycle prompt-cache reuse took the reasoned dense prefix path after the coordinate scope
    /// closed, exactly as `finish_coordinate_lifecycle` records it) plus a fixture lifecycle run.
    fn test_compression_block() -> ReceiptCompression {
        let mut primary =
            test_compressed_observation(vec![test_packed_evidence(4)], &[], Some(TEST_STORAGE));
        primary.dense_fallbacks.push((
            "prompt-cache-reuse".into(),
            COMPRESSED_PREFIX_REUSE_FALLBACK_REASON.into(),
        ));
        let lifecycle = test_compressed_observation(
            vec![test_packed_evidence(6)],
            &[(
                "prompt-cache-reuse",
                COMPRESSED_PREFIX_REUSE_FALLBACK_REASON,
            )],
            None,
        );
        compressed_receipt_block(
            CompressedKvMethod::GroupAffine,
            &primary,
            &[&primary, &lifecycle],
        )
        .unwrap()
    }

    /// A forced continuation over the full length with `matches` agreeing positions (the flips
    /// are the last positions).
    fn test_forced_continuation(matches: u64) -> ReceiptForcedContinuation {
        let reference = (0..FORCED_CONTINUATION_TOKENS as i32).collect::<Vec<_>>();
        let choices = reference
            .iter()
            .enumerate()
            .map(|(position, token)| {
                if (position as u64) < matches {
                    *token
                } else {
                    -1
                }
            })
            .collect::<Vec<_>>();
        forced_continuation_evidence(&reference, &choices).unwrap()
    }

    /// [`test_forced_continuation`] of the multi-turn fixture's turn 2.
    fn test_multi_turn_forced_continuation(matches: u64) -> ReceiptForcedContinuation {
        ReceiptForcedContinuation {
            method: MULTI_TURN_FORCED_CONTINUATION_METHOD.into(),
            ..test_forced_continuation(matches)
        }
    }

    /// Record `continuation` as the receipt's greedy agreement and its gate as the frozen
    /// evaluation of the receipt-level values for every repeat.
    fn set_test_compressed_quality(receipt: &mut Receipt, continuation: ReceiptForcedContinuation) {
        let quality = &mut receipt.quality;
        let multi_turn = test_multi_turn_forced_continuation(FORCED_CONTINUATION_TOKENS);
        quality.multi_turn_prompt_cache = multi_turn.agreement;
        quality.multi_turn_forced_continuation = Some(multi_turn);
        quality.multi_turn_forced_pass = Some(test_multi_turn_forced_pass());
        quality.greedy_token_agreement = continuation.agreement;
        quality.greedy_token_agreement_by_repeat =
            vec![continuation.agreement; QUALITY_MEASUREMENTS];
        let metrics = QualityMetrics {
            parity_max_error: quality.parity_max_error,
            perplexity_delta: quality.perplexity_delta,
            greedy_token_agreement: continuation.agreement,
            structured_tool_agreement: quality.structured_tool_agreement,
            needle_retrieval: quality.needle_retrieval,
            multi_turn_prompt_cache: quality.multi_turn_prompt_cache,
        };
        quality.forced_continuation = Some(continuation);
        quality.quality_gate = Some(quality_gate_from_repeats(
            &[metrics; QUALITY_MEASUREMENTS],
            quality.needle_discriminating,
        ));
    }

    /// Turn the builder's dense receipt into the same row measured in compressed mode: same-weights
    /// fixtures, compressed lifecycle, a persistent KV below the dense geometry, and the block.
    fn compress_test_receipt(receipt: &mut Receipt, compression: ReceiptCompression) {
        receipt.mode = "compressed".into();
        receipt.provenance.command = receipt
            .provenance
            .command_template
            .replace("{mode}", "compressed");
        for name in REQUIRED_FIXTURES {
            receipt
                .quality
                .fixture_evidence
                .get_mut(name)
                .unwrap()
                .independent_reference = fixture_independent_reference(
                name,
                QualityReference::DenseKvSameWeights,
                &receipt.provenance.model_file_sha256,
            );
        }
        receipt.lifecycle = receipt_lifecycle(true);
        set_test_compressed_quality(
            receipt,
            test_forced_continuation(FORCED_CONTINUATION_TOKENS),
        );
        let compressed_kv = receipt.memory.persistent_kv_bytes / 2;
        for event in &mut receipt.memory.allocation_events {
            if event.role == "cache" && event.lifetime != "transient" {
                event.bytes = compressed_kv;
            }
        }
        receipt.memory.persistent_kv_bytes = compressed_kv;
        receipt.memory.reconciliation.observed_persistent_kv_bytes = compressed_kv;
        receipt.compression = Some(compression);
        receipt.receipt_sha256 = receipt_semantic_seal(receipt).unwrap();
    }

    #[test]
    fn compressed_row_round_trips_producer_evidence_through_the_validators() {
        let block = test_compression_block();
        assert_eq!(block.method, "group-affine");
        assert_eq!(block.fused_calls, 10);
        assert_eq!(block.fallback_calls, 2);
        assert_eq!(
            block.physical_kv_bytes, 3,
            "device code + metadata + host staging"
        );
        assert_eq!(block.storage_tokens, 1);
        assert_eq!(block.persistent_kv_representation, COMPRESSED_PERSISTENT_KV);
        assert_eq!(block.full_cache_dequantizations, 0);

        let dense = builder_test_receipt();
        assert!(!String::from_utf8(dense.bytes().unwrap())
            .unwrap()
            .contains("compression"));
        let mut compressed = builder_test_receipt();
        compress_test_receipt(&mut compressed, block.clone());
        validate_sealed_receipt(&compressed).expect("same-weights compressed row is accepted");
        let parsed: Receipt = serde_json::from_slice(&compressed.bytes().unwrap()).unwrap();
        validate_sealed_receipt(&parsed).expect("serialized compressed receipt still validates");
        assert_eq!(parsed.compression, Some(block.clone()));

        let rejects = |mutate: &dyn Fn(&mut Receipt), expected: &str| {
            let mut receipt = compressed.clone();
            mutate(&mut receipt);
            receipt.receipt_sha256 = receipt_semantic_seal(&receipt).unwrap();
            let error = validate_sealed_receipt(&receipt).unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        };
        // Different weights: a bf16 (or any other model's) denominator is refused.
        rejects(
            &|r| {
                for name in REQUIRED_FIXTURES {
                    r.quality
                        .fixture_evidence
                        .get_mut(name)
                        .unwrap()
                        .independent_reference = fixture_independent_reference(
                        name,
                        QualityReference::Bf16Characterization,
                        &r.provenance.reference_model_sha256,
                    );
                }
            },
            "contract v3 denominator",
        );
        // The fused reader's tuning profile must be a known GPU family.
        rejects(
            &|r| r.compression.as_mut().unwrap().kernel_gpu_family = "made-up-family".into(),
            "identity is incomplete",
        );
        // The receipt names the kernel paths the reader actually ran: the NAX kernel exactly with
        // the NAX selection, and every fused call attributed.
        assert_eq!(
            block
                .kernel_paths
                .iter()
                .map(|path| (path.kernel.as_str(), path.selection.as_str(), path.calls))
                .collect::<Vec<_>>(),
            vec![
                (
                    crate::primitives::PACKED_NAX_KERNEL,
                    crate::primitives::PACKED_SELECTION_NAX,
                    2
                ),
                (
                    crate::primitives::PACKED_PER_ROW_KERNEL,
                    crate::primitives::PACKED_SELECTION_BELOW_MULTI_ROW,
                    8
                ),
            ]
        );
        let path_rejected = "disagrees with its selection or is unordered";
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().kernel_paths[0].selection =
                    crate::primitives::PACKED_SELECTION_NAX_UNAVAILABLE.into()
            },
            path_rejected,
        );
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().kernel_paths[0].kernel =
                    crate::primitives::PACKED_TILED_KERNEL.into()
            },
            path_rejected,
        );
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().kernel_paths[1].selection =
                    crate::primitives::PACKED_SELECTION_NAX.into()
            },
            path_rejected,
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().kernel_paths[0].query_dtype = "float32".into(),
            path_rejected,
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().kernel_paths[0].reason = " ".into(),
            path_rejected,
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().kernel_paths.swap(0, 1),
            path_rejected,
        );
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().kernel_gpu_family =
                    crate::primitives::PackedMetalGpuFamily::ConservativeUnknownApple
                        .as_str()
                        .into()
            },
            path_rejected,
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().kernel_paths[1].calls -= 1,
            "not all attributed to a kernel path",
        );
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().kernel_paths.clear();
            },
            "not all attributed to a kernel path",
        );
        // Silent fallback: unreasoned or uncounted dense execution, or no fused execution at all.
        rejects(
            &|r| r.compression.as_mut().unwrap().fallback_calls += 1,
            "not fully reasoned",
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().fallbacks[0].reason = " ".into(),
            "unreasoned",
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().fused_calls = 0,
            "never executed the fused",
        );
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.fallbacks.insert(
                    0,
                    ReceiptCompressionFallback {
                        operation: "z".into(),
                        reason: "r".into(),
                        calls: 1,
                    },
                );
                c.fallback_calls += 1;
            },
            "unordered",
        );
        rejects(
            &|r| {
                r.lifecycle.dense_fallback = false;
                r.lifecycle
                    .fallback_reasons
                    .insert("denseFallbackFallbackReason".into(), "unsupported".into());
            },
            "not fully reasoned",
        );
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.persistent_kv_representation = DENSE_FALLBACK_PERSISTENT_KV.into();
                c.fallbacks.clear();
                c.fallback_calls = 0;
            },
            "persistent KV representation",
        );
        // A representation labelled compressed may not reach the dense geometry once its host
        // copy and staged tail are counted.
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.host_payload_bytes += 1;
                c.physical_kv_bytes += 1;
            },
            "persistent KV representation",
        );
        // The persistent KV is exactly the coordinate storage's device share, at the receipt's
        // KV length.
        rejects(
            &|r| {
                let dense = r.memory.dense_theoretical_kv_bytes;
                for event in &mut r.memory.allocation_events {
                    if event.role == "cache" && event.lifetime != "transient" {
                        event.bytes = dense;
                    }
                }
                r.memory.persistent_kv_bytes = dense;
                r.memory.reconciliation.observed_persistent_kv_bytes = dense;
            },
            "coordinate's measured storage",
        );
        rejects(
            &|r| r.compression.as_mut().unwrap().storage_tokens += 1,
            "coordinate's measured storage",
        );
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.device_code_bytes += 1;
                c.host_payload_bytes -= 1;
            },
            "coordinate's measured storage",
        );
        // A dense coordinate claims no compressed storage (so it can never yield a reduction).
        rejects(
            &|r| {
                r.compression.as_mut().unwrap().persistent_kv_representation =
                    DENSE_FALLBACK_PERSISTENT_KV.into()
            },
            "persistent KV representation",
        );
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.persistent_kv_representation = DENSE_FALLBACK_PERSISTENT_KV.into();
                c.storage_tokens = 0;
            },
            "persistent KV representation",
        );
        rejects(
            &|r| {
                let c = r.compression.as_mut().unwrap();
                c.persistent_kv_representation = DENSE_FALLBACK_PERSISTENT_KV.into();
                (c.device_code_bytes, c.device_metadata_bytes) = (0, 0);
                (c.host_payload_bytes, c.physical_kv_bytes) = (0, 0);
            },
            "persistent KV representation",
        );
        // Dense reconstruction: counted by the cache, or witnessed by an explicit event.
        rejects(
            &|r| r.compression.as_mut().unwrap().full_cache_dequantizations = 1,
            "reconstructed a dense full cache",
        );
        rejects(
            &|r| {
                r.memory.allocation_events.push(ReceiptAllocation {
                    kind: FULL_CACHE_MATERIALIZATION_KIND.into(),
                    role: "cache".into(),
                    lifetime: "transient".into(),
                    phase: "prefill-peak".into(),
                    timestamp: "2026-01-01T00:00:02.300Z".into(),
                    bytes: 1,
                })
            },
            "explicit full-cache temporary",
        );
        // Measured storage must reconcile, and the block belongs exactly to compressed rows.
        rejects(
            &|r| r.compression.as_mut().unwrap().host_payload_bytes += 1,
            "do not reconcile with measured storage",
        );
        rejects(&|r| r.compression = None, "present exactly on compressed");
        let mut dense_with_block = builder_test_receipt();
        dense_with_block.compression = Some(block);
        assert!(validate_receipt_semantics(&dense_with_block)
            .unwrap_err()
            .contains("present exactly on compressed"));
    }

    const TEST_NEEDLE: &str = "SC20671-NUMERIC-NEEDLE-9b7a2e";

    /// The row's compressed quality measurement(s) sharing `continuation`; `needle_miss` ones lose the needle the
    /// same-weights dense run recovered.
    fn compressed_test_suites(
        continuation: &ReceiptForcedContinuation,
        needle_miss: &[usize],
    ) -> Vec<ProductFixtureSuite> {
        (0..QUALITY_MEASUREMENTS)
            .map(|repeat| {
                let candidate = test_fixture_half(if needle_miss.contains(&repeat) {
                    "wrong"
                } else {
                    TEST_NEEDLE
                });
                let mut suite = pair_product_fixture_halves(
                    candidate,
                    test_fixture_half(TEST_NEEDLE),
                    QualityReference::DenseKvSameWeights,
                )
                .unwrap();
                suite.kernel_parity_errors = Some(vec![0.00005]);
                suite.forced_continuation = Some(continuation.clone());
                suite.stream_likelihood =
                    Some(StreamLikelihood::from_probabilities(&[0.5, 0.25], &[0.5, 0.25]).unwrap());
                suite.multi_turn_forced_continuation = Some(test_multi_turn_forced_continuation(
                    FORCED_CONTINUATION_TOKENS,
                ));
                suite.multi_turn_forced_pass = Some(test_multi_turn_forced_pass());
                suite
            })
            .collect()
    }

    /// The sealed repeat-major artifacts of `suites`, as the bundle validator reads them.
    fn sealed_test_repeats(suites: &[ProductFixtureSuite]) -> Vec<SealedFixtureArtifact> {
        sealed_product_fixture_artifacts(suites, &required_coordinates()[0]).unwrap()
    }

    fn repeat_pairs(fixtures: &[SealedFixtureArtifact]) -> Vec<(&'static str, &[u8])> {
        (0..QUALITY_MEASUREMENTS)
            .flat_map(|repeat| {
                REQUIRED_FIXTURES.map(|fixture| {
                    let name = fixture_artifact_name(fixture, repeat);
                    (
                        fixture,
                        fixtures
                            .iter()
                            .find(|artifact| artifact.name == name)
                            .unwrap()
                            .bytes
                            .as_slice(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn forced_continuation_is_admitted_up_to_the_native_window() {
        assert_eq!(forced_continuation_live_tokens(60, 131_072), 60 + 1024);
        assert_eq!(forced_continuation_live_tokens(130_600, 131_072), 131_072);
    }

    #[test]
    fn forced_continuation_records_every_position_and_the_first_flips() {
        let evidence = forced_continuation_evidence(&[1, 2, 3, 4], &[1, 0, 3, 0]).unwrap();
        assert_eq!(
            (evidence.tokens, evidence.matches, evidence.flip_count),
            (4, 2, 2)
        );
        assert_eq!(evidence.agreement, 0.5);
        assert_eq!(evidence.first_flip_positions, vec![1, 3]);
        assert!(forced_continuation_evidence(&[1, 2], &[1]).is_err());
        assert!(forced_continuation_evidence(&[], &[]).is_err());
        // One flip in 1024 positions clears the frozen 0.999; two do not.
        let one = test_forced_continuation(FORCED_CONTINUATION_TOKENS - 1);
        let two = test_forced_continuation(FORCED_CONTINUATION_TOKENS - 2);
        assert!(one.agreement >= COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN);
        assert!(two.agreement < COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN);
        let many = test_forced_continuation(100);
        assert_eq!(many.flip_count, FORCED_CONTINUATION_TOKENS - 100);
        assert_eq!(
            many.first_flip_positions,
            (100..100 + FORCED_CONTINUATION_RECORDED_FLIPS as u64).collect::<Vec<_>>()
        );
        validate_forced_continuation(&many, "short").unwrap();
        // Only a fit-boundary row may run a shorter continuation, never below the reserve.
        let short = forced_continuation_evidence(&[1; 300], &[1; 300]).unwrap();
        assert!(validate_forced_continuation(&short, "short").is_err());
        validate_forced_continuation(&short, "fit-boundary").unwrap();
        let tiny = forced_continuation_evidence(&[1; 8], &[1; 8]).unwrap();
        assert!(validate_forced_continuation(&tiny, "fit-boundary").is_err());
        for edit in [
            (|c: &mut ReceiptForcedContinuation| c.matches -= 1) as fn(&mut _),
            |c| c.agreement = 1.0,
            |c| c.flip_count += 1,
            |c| c.first_flip_positions.reverse(),
            |c| c.first_flip_positions.pop().map(drop).unwrap_or(()),
            |c| c.method = "natural-stream".into(),
            |c| c.reference_stream_sha256 = "x".into(),
        ] {
            let mut forged = many.clone();
            edit(&mut forged);
            assert!(validate_forced_continuation(&forged, "short").is_err());
        }
    }

    /// Contract v4 fixture sizing: only the fit-boundary band's multi-turn payload is capped, so
    /// turn 2 plus the full 1024-token forced continuation always fits the native window, and a
    /// row whose turn 2 would not leave it is refused before any model loads.
    #[test]
    fn multi_turn_fixture_is_sized_to_leave_the_full_forced_continuation() {
        for (native, family) in [(131_072_u64, "llama"), (40_960, "qwen")] {
            for band in CONTEXT_BANDS {
                let target = context_band_target(native, band).unwrap();
                let sized = multi_turn_payload_target(native, target).unwrap();
                if band == "fit-boundary" {
                    assert_eq!(
                        sized,
                        native - 1024 - 64 - MULTI_TURN_PROMPT_RESERVE_TOKENS,
                        "{family}"
                    );
                    assert!(sized < target);
                } else {
                    assert_eq!(sized, target, "{family} {band} keeps its band payload");
                }
                assert!(sized + 64 + MULTI_TURN_PROMPT_RESERVE_TOKENS + 1024 <= native);
            }
        }
        assert_eq!(
            multi_turn_payload_target(131_072, 130_560).unwrap(),
            129_472
        );
        assert_eq!(multi_turn_payload_target(40_960, 40_448).unwrap(), 39_360);
        assert!(multi_turn_payload_target(1_500, 1_000).is_err());

        // Preflight over a real tokenizer and template at a 4096-token window: every band of
        // the sized fixture is admitted, and an over-long prompt whose turn 2 would leave fewer
        // than 1024 tokens is refused before load.
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = serde_json::json!({
            "version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": { "type": "Whitespace" }, "post_processor": null, "decoder": null,
            "model": { "type": "WordLevel", "vocab": { "<unk>": 0, "context": 1 },
                       "unk_token": "<unk>" },
        });
        fs::write(dir.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let spec = BenchmarkModelSpec {
            family: "llama",
            role: "candidate",
            repository: "test/tiny",
            revision: "0",
            architecture: "LlamaForCausalLM",
            model_type: "llama",
            native_context_tokens: 4096,
            quantized: true,
            required_files: &[],
        };
        let coordinate = |context_band| Coordinate {
            family: "llama",
            context_band,
            request_mode: "single",
            prefill_mode: "single-shot",
            process_temperature: "cold",
        };
        for band in CONTEXT_BANDS {
            for compressed in [false, true] {
                preflight_total_live_tokens(
                    dir.path(),
                    &spec,
                    &coordinate(band),
                    "baseline prompt",
                    compressed,
                )
                .unwrap_or_else(|e| panic!("{band} compressed={compressed}: {e}"));
            }
        }
        let long_prompt = "word ".repeat(2_400);
        let error = preflight_total_live_tokens(
            dir.path(),
            &spec,
            &coordinate("memory-material"),
            &long_prompt,
            true,
        )
        .unwrap_err();
        assert!(
            error.contains("leaves fewer than the 1024-token forced continuation"),
            "{error}"
        );
    }

    /// Quality contract v4: `multiTurnPromptCache` is teacher-forced on turn 2 after its
    /// prompt-cache hit (a compressed row's turn-2 forced continuation, a dense row's natural
    /// reference stream), every measured repeat needs that pass, turn 2 must have hit the cache
    /// in both arms, and the gate's minimum is 0.999 like greedy agreement. Free-running turn-2
    /// agreement is an observation only.
    #[test]
    fn multi_turn_prompt_cache_is_teacher_forced_on_a_cache_hit_turn_two() {
        let pair = || {
            pair_product_fixture_halves(
                test_fixture_half(TEST_NEEDLE),
                test_fixture_half(TEST_NEEDLE),
                QualityReference::DenseKvSameWeights,
            )
            .unwrap()
        };
        // Dense rows: teacher-forced on the reference turn-2 stream. A free-running candidate
        // that diverged at position 0 does not lower it when every forced position agrees.
        let mut suite = pair();
        let reference_stream = stream_tokens(
            &suite
                .cache_reference
                .quality_observation
                .token_probabilities,
        );
        suite
            .cache_candidate
            .quality_observation
            .token_probabilities
            .iter_mut()
            .for_each(|(token, _)| *token += 1000);
        assert_eq!(
            suite.quality().unwrap().cache_matches,
            0,
            "free-running warmup scoring"
        );
        suite.teacher_forced_cache = Some(TeacherForcedChoices {
            choices: reference_stream.clone(),
            stop_tokens: Vec::new(),
        });
        let forced = suite.quality().unwrap();
        assert_eq!(
            (forced.cache_matches, forced.cache_total),
            (reference_stream.len() as u64, reference_stream.len() as u64)
        );
        assert_eq!(forced.cache_free_running_first_divergence, Some(0));
        assert_eq!(forced.cache_matched_prefix_tokens, 0);
        // Compressed rows: the turn-2 forced continuation supplies the counts.
        suite.multi_turn_forced_continuation = Some(test_multi_turn_forced_continuation(1022));
        assert!(suite
            .quality()
            .unwrap_err()
            .contains("sealed with both sessions' turn records"));
        suite.multi_turn_forced_pass = Some(test_multi_turn_forced_pass());
        let compressed = suite.quality().unwrap();
        assert_eq!(
            (compressed.cache_matches, compressed.cache_total),
            (1022, 1024)
        );
        // A turn 2 the cache did not serve, or a fixture with no per-turn record, fails closed.
        for arm in ["candidate", "reference"] {
            let mut missed = pair();
            let result = if arm == "candidate" {
                &mut missed.cache_candidate
            } else {
                &mut missed.cache_reference
            };
            let mut turns = test_multi_turn();
            turns.turn2.cache_hit = false;
            turns.turn2.reused_prefix_tokens = 0;
            result.multi_turn = Some(turns);
            let error = missed.quality().unwrap_err();
            assert!(
                error.contains(&format!("{arm} multi-turn prompt cache")),
                "{error}"
            );
            let mut unrecorded = pair();
            unrecorded.cache_reference.multi_turn = None;
            assert!(unrecorded
                .quality()
                .unwrap_err()
                .contains("per-turn cache record"));
        }
        // The quality measurement requires its teacher-forced turn-2 pass.
        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS);
        let mut suites = compressed_test_suites(&continuation, &[]);
        suites[0].multi_turn_forced_continuation = None;
        assert!(receipt_quality_over_repeats(&suites)
            .unwrap_err()
            .contains("teacher-forced multi-turn prompt-cache agreement"));
        // One flip in 1024 meets the 0.999 minimum; two do not.
        for (matches, passes) in [(1023, true), (1022, false)] {
            let suites = compressed_test_suites(&continuation, &[])
                .into_iter()
                .map(|mut suite| {
                    suite.multi_turn_forced_continuation =
                        Some(test_multi_turn_forced_continuation(matches));
                    suite
                })
                .collect::<Vec<_>>();
            let receipt = compressed_test_builder(receipt_quality_over_repeats(&suites).unwrap())
                .finish()
                .unwrap();
            let gate = receipt.quality.quality_gate.as_ref().unwrap();
            assert_eq!(
                gate.failures
                    .iter()
                    .filter(|failure| failure.metric == "multiTurnPromptCache")
                    .count(),
                if passes { 0 } else { QUALITY_MEASUREMENTS },
                "{gate:?}"
            );
            assert_eq!(
                receipt.quality.multi_turn_prompt_cache,
                matches as f64 / 1024.0
            );
            assert_eq!(
                receipt.quality.multi_turn_prompt_cache_method,
                MULTI_TURN_PROMPT_CACHE_METHOD
            );
        }
        // The receipt validators bind the value, method, records and contract identity.
        let receipt = compressed_test_builder(
            receipt_quality_over_repeats(&compressed_test_suites(&continuation, &[])).unwrap(),
        )
        .finish()
        .unwrap();
        let refuses = |edit: &dyn Fn(&mut Receipt), expected: &str| {
            let mut forged = receipt.clone();
            edit(&mut forged);
            let error = validate_receipt_semantics(&forged).unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        };
        refuses(
            &|r| r.quality.multi_turn_prompt_cache = 0.5625,
            "not the row's turn-2 forced-continuation agreement",
        );
        refuses(
            &|r| r.quality.multi_turn_forced_continuation = None,
            "no turn-2 forced continuation",
        );
        refuses(
            &|r| {
                r.quality
                    .multi_turn_forced_continuation
                    .as_mut()
                    .unwrap()
                    .method = FORCED_CONTINUATION_METHOD.into()
            },
            "forced continuation evidence is inconsistent",
        );
        refuses(
            &|r| r.quality.multi_turn_prompt_cache_method = "free-running".into(),
            "not teacher-forced on a cache-hit turn 2",
        );
        refuses(
            &|r| r.quality.multi_turn_cache.reference.turn2.cache_hit = false,
            "reference multi-turn prompt cache did not serve turn 2",
        );
        refuses(
            &|r| r.quality.multi_turn_forced_pass = None,
            "no turn records for its turn-2 forced continuation",
        );
        refuses(
            &|r| {
                r.quality
                    .multi_turn_forced_pass
                    .as_mut()
                    .unwrap()
                    .candidate
                    .turn2
                    .prompt_sha256 = "3".repeat(64)
            },
            "reference and candidate turn-2 prompts differ",
        );
        refuses(
            &|r| {
                let pass = r.quality.multi_turn_forced_pass.as_mut().unwrap();
                pass.reference.turn2.prompt_sha256 = "4".repeat(64);
                pass.candidate.turn2.prompt_sha256 = "4".repeat(64);
            },
            "not the multi-turn fixture's turn-2 prompt",
        );
        // A contract v3 receipt is refused outright.
        refuses(
            &|r| {
                r.contract_hash =
                    "58eaa007c35084c8acac2b35b5a2a1dff5af55832a7533557741d43a05944b49".into()
            },
            "receipt constants mismatch",
        );
        // Sealed repeat artifacts carry the turn-2 continuation, method and turn records.
        let suites = compressed_test_suites(&continuation, &[]);
        let artifacts = sealed_test_repeats(&suites);
        let pairs = repeat_pairs(&artifacts);
        let metrics = sealed_repeat_quality_metrics(&receipt, &pairs).unwrap();
        assert!(metrics
            .iter()
            .all(|metrics| metrics.multi_turn_prompt_cache == 1.0));
        let cache_index = REQUIRED_FIXTURES
            .iter()
            .position(|name| *name == "multi-turn-prompt-cache")
            .unwrap();
        let tamper = |edit: &dyn Fn(&mut serde_json::Value), expected: &str| {
            let mut value: serde_json::Value =
                serde_json::from_slice(pairs[cache_index].1).unwrap();
            edit(&mut value);
            let bytes = serde_json::to_vec(&value).unwrap();
            let mut forged = pairs.clone();
            forged[cache_index].1 = &bytes;
            let error = sealed_repeat_quality_metrics(&receipt, &forged).unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        };
        tamper(
            &|v| v["evidence"]["matches"] = serde_json::json!(36),
            "not its turn-2 forced continuation's",
        );
        tamper(
            &|v| v["evidence"]["method"] = serde_json::json!("free-running"),
            "not teacher-forced on a cache-hit turn 2",
        );
        tamper(
            &|v| {
                v["evidence"]["turns"]["candidate"]["turn2"]["cacheHit"] = serde_json::json!(false)
            },
            "candidate multi-turn prompt cache",
        );
        tamper(
            &|v| {
                v["evidence"]
                    .as_object_mut()
                    .unwrap()
                    .remove("forcedContinuation");
            },
            "forced continuation is not the receipt's",
        );
        tamper(
            &|v| v["evidence"]["turns"]["reference"]["turn2"]["reusedPrefixTokens"] = 41.into(),
            "receipt multiTurnCache is not the primary repeat's sealed turn records",
        );
        tamper(
            &|v| {
                v["evidence"].as_object_mut().unwrap().remove("forcedPass");
            },
            "forced-pass turn records are not the receipt's",
        );
        // Forced passes over different turn-2 prompts never score a row.
        let mut mismatched = test_multi_turn_forced_pass();
        mismatched.candidate.turn2.prompt_sha256 = "5".repeat(64);
        assert!(mismatched
            .validate()
            .unwrap_err()
            .contains("turn-2 prompts differ"));
        // Same-weights arms that rendered different turn-2 prompts are not comparable.
        let mut diverged = pair();
        diverged
            .cache_candidate
            .multi_turn
            .as_mut()
            .unwrap()
            .turn2
            .prompt_sha256 = "6".repeat(64);
        assert!(diverged
            .quality()
            .unwrap_err()
            .contains("turn-2 prompts differ"));
    }

    #[test]
    fn compressed_quality_miss_is_accepted_as_a_measured_gate_failure() {
        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS - 2);
        let suites = compressed_test_suites(&continuation, &[0]);
        let quality = receipt_quality_over_repeats(&suites).unwrap();
        let receipt = compressed_test_builder(quality)
            .finish()
            .expect("a validly measured quality miss is an accepted, measured row");
        let agreement = 1022.0 / 1024.0;
        let failure = |metric: &str, fixture: &str, repeat: u64, value: f64, threshold: f64| {
            ReceiptQualityGateFailure {
                metric: metric.into(),
                fixture: fixture.into(),
                repeat,
                value,
                threshold,
                comparison: "minimum".into(),
            }
        };
        // The one quality measurement misses greedy agreement and (dense recovered) the needle.
        let expected = vec![
            failure(
                "greedyTokenAgreement",
                "kernel-fp32-reference",
                0,
                agreement,
                0.999,
            ),
            failure("needleRetrieval", "long-context-needle", 0, 0.0, 1.0),
        ];
        assert_eq!(
            receipt.quality.quality_gate,
            Some(ReceiptQualityGate {
                passed: false,
                failures: expected
            })
        );
        assert_eq!(receipt.quality.greedy_token_agreement, agreement);
        assert_eq!(
            receipt.quality.greedy_token_agreement_by_repeat,
            vec![agreement; QUALITY_MEASUREMENTS]
        );
        assert_eq!(receipt.quality.needle_retrieval, 0.0);
        assert_eq!(
            receipt.quality.forced_continuation,
            Some(continuation.clone())
        );
        let sealed = receipt_semantic_seal(&receipt)
            .map(|seal| Receipt {
                receipt_sha256: seal,
                ..receipt.clone()
            })
            .unwrap();
        validate_sealed_receipt(&sealed).unwrap();
        let parsed: Receipt = serde_json::from_slice(&sealed.bytes().unwrap()).unwrap();
        assert_eq!(parsed.quality.quality_gate, receipt.quality.quality_gate);
        validate_sealed_receipt(&parsed).unwrap();

        // The gate is exactly the evaluation of the sealed repeat artifacts.
        let fixtures = sealed_test_repeats(&suites);
        let repeats = sealed_repeat_quality_metrics(&receipt, &repeat_pairs(&fixtures)).unwrap();
        assert_eq!(repeats[0].needle_retrieval, 0.0);
        validate_sealed_quality_gate(&receipt, &repeats).unwrap();
        let bundle = assemble_artifacts_with_fixtures(receipt.clone(), fixtures).unwrap();
        let human = String::from_utf8(bundle.human).unwrap();
        assert!(human.contains(
            "- Quality gate: FAILED: greedyTokenAgreement repeat 0 = 0.998046875 (minimum 0.999, fixture kernel-fp32-reference);"
        ), "{human}");
        assert!(
            human.contains("needleRetrieval repeat 0 = 0 (minimum 1, fixture long-context-needle)")
        );

        // A pass can never be claimed over a failing value.
        let rejects = |edit: &dyn Fn(&mut ReceiptQualityGate), expected: &str| {
            let mut forged = receipt.clone();
            edit(forged.quality.quality_gate.as_mut().unwrap());
            let error = validate_receipt_semantics(&forged).unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        };
        rejects(
            &|gate| *gate = ReceiptQualityGate { passed: true, failures: Vec::new() },
            "does not record greedyTokenAgreement repeat 0 = 0.998046875 against the frozen minimum 0.999",
        );
        rejects(
            &|gate| gate.passed = true,
            "claims passed=true with 2 recorded failure(s)",
        );
        rejects(
            &|gate| gate.failures[0].value = 0.5,
            "does not record greedyTokenAgreement repeat 0",
        );
        rejects(
            &|gate| gate.failures[0].threshold = 0.9,
            "not an ordered frozen-threshold miss",
        );
        rejects(
            &|gate| gate.failures.swap(0, 1),
            "not an ordered frozen-threshold miss",
        );
        rejects(
            &|gate| gate.failures[1].fixture = "kernel-fp32-reference".into(),
            "not an ordered frozen-threshold miss",
        );
        // A discriminating needle miss is visible in the receipt and cannot be dropped.
        rejects(
            &|gate| {
                gate.failures
                    .retain(|failure| failure.metric != "needleRetrieval")
            },
            "does not record needleRetrieval repeat 0",
        );
        // Contract v5: when the same-weights dense run missed the needle too, the compressed miss
        // is an observation: it is recorded (needleRetrieval 0, needleDiscriminating false) and
        // never gated, and a gate that records it is refused.
        let mut shared = compressed_test_suites(&continuation, &[0]);
        shared[0].needle_reference = test_fixture_half("wrong").needle_result;
        let shared_quality = receipt_quality_over_repeats(&shared).unwrap();
        assert!(!shared_quality.needle_discriminating);
        let observed = compressed_test_builder(shared_quality).finish().unwrap();
        assert_eq!(observed.quality.needle_retrieval, 0.0);
        assert_eq!(
            observed
                .quality
                .quality_gate
                .as_ref()
                .unwrap()
                .failures
                .iter()
                .map(|failure| failure.metric.as_str())
                .collect::<Vec<_>>(),
            ["greedyTokenAgreement"]
        );
        let mut gated = observed.clone();
        gated
            .quality
            .quality_gate
            .as_mut()
            .unwrap()
            .failures
            .push(failure(
                "needleRetrieval",
                "long-context-needle",
                0,
                0.0,
                1.0,
            ));
        assert!(validate_receipt_semantics(&gated)
            .unwrap_err()
            .contains("not an ordered frozen-threshold miss"));
        let shared_fixtures = sealed_test_repeats(&shared);
        let shared_repeats =
            sealed_repeat_quality_metrics(&observed, &repeat_pairs(&shared_fixtures)).unwrap();
        validate_sealed_quality_gate(&observed, &shared_repeats).unwrap();
        validate_repeat_discrimination(&observed, repeat_pairs(&shared_fixtures)).unwrap();

        // A row that meets every threshold records passed with no failures.
        let clean = compressed_test_suites(
            &test_forced_continuation(FORCED_CONTINUATION_TOKENS - 1),
            &[],
        );
        let passed = compressed_test_builder(receipt_quality_over_repeats(&clean).unwrap())
            .finish()
            .unwrap();
        assert_eq!(
            passed.quality.quality_gate,
            Some(ReceiptQualityGate {
                passed: true,
                failures: Vec::new()
            })
        );
        let clean_fixtures = sealed_test_repeats(&clean);
        validate_sealed_quality_gate(
            &passed,
            &sealed_repeat_quality_metrics(&passed, &repeat_pairs(&clean_fixtures)).unwrap(),
        )
        .unwrap();
        // The campaign is gate-failed when any row failed; dense campaigns carry no verdict.
        assert_eq!(
            campaign_quality_gate_passed(&[passed.clone(), passed.clone()]),
            Ok(Some(true))
        );
        assert_eq!(
            campaign_quality_gate_passed(&[passed.clone(), receipt.clone()]),
            Ok(Some(false))
        );
        let dense = builder_test_receipt();
        assert_eq!(
            campaign_quality_gate_passed(&[dense.clone(), dense.clone()]),
            Ok(None)
        );
        assert!(campaign_quality_gate_passed(&[dense, passed]).is_err());
    }

    #[test]
    fn compressed_integrity_failures_still_refuse_the_row() {
        let continuation = test_forced_continuation(FORCED_CONTINUATION_TOKENS);
        let quality =
            || receipt_quality_over_repeats(&compressed_test_suites(&continuation, &[])).unwrap();
        // Kernel parity against the host-fp32 dequantize-then-attend reference is correctness.
        let mut broken_kernel = quality();
        broken_kernel.parity_errors = vec![0.5];
        let error = compressed_test_builder(broken_kernel).finish().unwrap_err();
        assert!(
            error
                .contains("kernel parity failed: metric=parityMaxError value=0.5 threshold=0.0001"),
            "{error}"
        );
        // No forced continuation: the greedy agreement was never measured.
        let mut unmeasured = quality();
        unmeasured.forced_continuation = None;
        let error = compressed_test_builder(unmeasured).finish().unwrap_err();
        assert!(
            error.contains("no forced-continuation greedy agreement"),
            "{error}"
        );
        // No quality measurement, or more than the contract's one, cannot be gated.
        let mut partial = quality();
        partial.repeat_metrics.pop();
        assert!(compressed_test_builder(partial)
            .finish()
            .unwrap_err()
            .contains("one quality measurement"));
        let mut repeated = quality();
        repeated.repeat_metrics.push(repeated.repeat_metrics[0]);
        assert!(compressed_test_builder(repeated)
            .finish()
            .unwrap_err()
            .contains("one quality measurement"));
        // Contract v5: quality is measured once per arm; five quality repeats are refused.
        let mut suites = compressed_test_suites(&continuation, &[]);
        suites.extend(compressed_test_suites(&continuation, &[]));
        assert!(receipt_quality_over_repeats(&suites)
            .unwrap_err()
            .contains("measured exactly 1 time(s) per arm"));
        let receipt = compressed_test_builder(quality()).finish().unwrap();
        let refuses = |edit: &dyn Fn(&mut Receipt), expected: &str| {
            let mut forged = receipt.clone();
            edit(&mut forged);
            let error = validate_receipt_semantics(&forged).unwrap_err();
            assert!(error.contains(expected), "{expected}: {error}");
        };
        // Fixture evidence is integrity, named by fixture.
        refuses(
            &|r| {
                r.quality
                    .fixture_evidence
                    .get_mut("long-context-needle")
                    .unwrap()
                    .passed = false
            },
            "fixture evidence failed: fixture=long-context-needle passed=false",
        );
        refuses(
            &|r| {
                r.quality
                    .fixture_evidence
                    .get_mut("structured-tool-call")
                    .unwrap()
                    .artifact_sha256 = "A".repeat(64)
            },
            "fixture evidence failed: fixture=structured-tool-call artifactSha256=",
        );
        refuses(
            &|r| r.quality.quality_gate = None,
            "no measured quality-gate record",
        );
        refuses(
            &|r| r.quality.forced_continuation.as_mut().unwrap().matches -= 1,
            "forced continuation evidence is inconsistent",
        );
        refuses(
            &|r| r.quality.greedy_token_agreement_by_repeat[0] = 0.5,
            "greedy agreement is not the row's one recorded quality measurement",
        );
        // A dense row is characterization: it can carry neither a gate nor a continuation.
        let mut dense = builder_test_receipt();
        dense.quality.quality_gate = receipt.quality.quality_gate.clone();
        assert!(validate_receipt_semantics(&dense)
            .unwrap_err()
            .contains("only on compressed receipts"));
        // Sealed artifacts must carry the receipt's forced continuation and its counts.
        let suites = compressed_test_suites(&continuation, &[]);
        let fixtures = sealed_test_repeats(&suites);
        let mut pairs = repeat_pairs(&fixtures);
        let mut forged: serde_json::Value = serde_json::from_slice(pairs[0].1).unwrap();
        forged["evidence"]["forcedContinuation"]["matches"] = serde_json::json!(1000);
        let forged = serde_json::to_vec(&forged).unwrap();
        pairs[0].1 = &forged;
        assert!(sealed_repeat_quality_metrics(&receipt, &pairs)
            .unwrap_err()
            .contains("repeat 0 kernel fixture forced continuation"));
        let mut miscounted: serde_json::Value =
            serde_json::from_slice(repeat_pairs(&fixtures)[0].1).unwrap();
        miscounted["evidence"]["greedyMatches"] = serde_json::json!(1000);
        let miscounted = serde_json::to_vec(&miscounted).unwrap();
        let mut pairs = repeat_pairs(&fixtures);
        pairs[0].1 = &miscounted;
        assert!(sealed_repeat_quality_metrics(&receipt, &pairs)
            .unwrap_err()
            .contains("greedy counts"));
        let mut parity: serde_json::Value =
            serde_json::from_slice(repeat_pairs(&fixtures)[0].1).unwrap();
        parity["evidence"]["parityErrors"] = serde_json::json!([0.5]);
        let parity = serde_json::to_vec(&parity).unwrap();
        let mut pairs = repeat_pairs(&fixtures);
        pairs[0].1 = &parity;
        let repeats = sealed_repeat_quality_metrics(&receipt, &pairs).unwrap();
        assert!(validate_sealed_quality_gate(&receipt, &repeats)
            .unwrap_err()
            .contains("kernel parity failed: metric=parityMaxError value=0.5 threshold=0.0001 comparison=maximum fixture=kernel-fp32-reference repeat=0"));
    }

    #[test]
    fn compressed_producer_refuses_silent_fallback_and_counts_reconstructions() {
        let silent = crate::primitives::PackedCacheEvidence {
            dense_active: true,
            ..test_packed_evidence(3)
        };
        let primary = test_compressed_observation(vec![silent], &[], Some(TEST_STORAGE));
        assert!(
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary])
                .unwrap_err()
                .contains("without a recorded reason")
        );
        let unattributed = crate::primitives::PackedCacheEvidence {
            kernel_paths: test_kernel_paths(2),
            ..test_packed_evidence(3)
        };
        let primary = test_compressed_observation(vec![unattributed], &[], Some(TEST_STORAGE));
        assert!(
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary])
                .unwrap_err()
                .contains("not attributed to kernel paths")
        );
        let dense_only = test_compressed_observation(
            vec![test_packed_evidence(0)],
            &[("cache-selection", "refused")],
            Some(TEST_STORAGE),
        );
        assert!(compressed_receipt_block(
            CompressedKvMethod::GroupAffine,
            &dense_only,
            &[&dense_only]
        )
        .unwrap_err()
        .contains("never executed the fused"));
        let reconstructed = crate::primitives::PackedCacheEvidence {
            dense_active: true,
            full_cache_dequantizations: 1,
            fallback_reasons: vec![(
                "update".into(),
                "decoder layer requires dense attention".into(),
            )],
            ..test_packed_evidence(3)
        };
        let primary = test_compressed_observation(vec![reconstructed], &[], Some(TEST_STORAGE));
        let block =
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary])
                .unwrap();
        assert_eq!(block.full_cache_dequantizations, 1);
        assert_eq!(block.fallback_calls, 1);
        assert_eq!(
            block.persistent_kv_representation, DENSE_FALLBACK_PERSISTENT_KV,
            "a primary operation that left the fused path is not labelled compressed"
        );
        let mut receipt = builder_test_receipt();
        compress_test_receipt(&mut receipt, block);
        assert!(validate_sealed_receipt(&receipt)
            .unwrap_err()
            .contains("reconstructed a dense full cache"));
    }

    /// Drive a real `ProductObserver` through the producer's coordinate sequence: the events the
    /// decode stream emits for a compressed single-shot dispatch, then the shared
    /// `finish_coordinate_lifecycle` tail (prompt-cache reuse dense fallback, cancellation run
    /// replay, release). Only the coordinate operation may classify the representation.
    #[cfg(target_os = "macos")]
    #[test]
    fn producer_lifecycle_does_not_classify_the_compressed_coordinate() {
        let coordinate = crate::primitives::CompressedCacheStorage {
            device_code_bytes: 64,
            device_metadata_bytes: 32,
            host_payload_bytes: 16,
            tokens: 9,
            element_bytes: 2,
        };
        // The cancellation probe's cache: smaller device share, larger host staging, and a
        // reasoned fallback of its own. None of it describes the coordinate.
        let cancellation = crate::primitives::CompressedCacheStorage {
            device_code_bytes: 32,
            device_metadata_bytes: 16,
            host_payload_bytes: 4096,
            tokens: 3,
            element_bytes: 2,
        };
        let pid = std::process::id();
        let sample = |current_bytes| MemorySample {
            captured_at: timestamp_now(),
            pid,
            current_bytes,
            peak_bytes: current_bytes,
            mlx_active_bytes: current_bytes,
            mlx_cache_bytes: 0,
            mlx_peak_bytes: current_bytes,
        };
        let mut observer = ProductObserver::new();
        observer.bind_session("e".repeat(64));
        observer.load_elapsed_ms = Some(1.0);
        observer.load_boundary = Some((sample(1), sample(2)));
        Observer::phase(&mut observer, "process-start");
        Observer::snapshot_inventory(
            &mut observer,
            &SnapshotInventory {
                root: PathBuf::new(),
                files: Vec::new(),
                bytes: 1,
                sha256: "d".repeat(64),
            },
        );
        Observer::geometry(
            &mut observer,
            ProductGeometry {
                query_heads: 2,
                kv_heads: 1,
                head_dimension: 64,
                layers: 1,
                element_bytes: 0,
            },
        );
        observer.prefill_peak_window = Some(ReceiptPeakWindow {
            started_at: timestamp_now(),
            baseline_active_bytes: 1,
            reset_peak_bytes: 0,
        });
        Observer::phase(&mut observer, "weights-loaded");
        // Measured dispatch, as `observe_cache_events` reports a compressed cache.
        Observer::phase(&mut observer, "prefill-peak");
        Observer::logits(&mut observer, "prefill", &[0.25, 0.75]);
        let report = |observer: &mut ProductObserver,
                      storage: &crate::primitives::CompressedCacheStorage| {
            Observer::cache_snapshot(
                observer,
                storage.device_bytes(),
                storage.tokens,
                256,
                storage.element_bytes,
            );
            Observer::compressed_storage(observer, storage);
        };
        report(&mut observer, &coordinate);
        Observer::phase(&mut observer, "first-token");
        Observer::token_probability(&mut observer, "decode", 1, 0.5);
        Observer::phase(&mut observer, "decode-steady");
        report(&mut observer, &coordinate);
        Observer::packed_cache_evidence(&mut observer, &test_packed_evidence(4));
        Observer::release_event(
            &mut observer,
            "cache_release",
            "cache",
            coordinate.device_bytes(),
        );
        observer.operation("single-shot-generation");
        finish_coordinate_lifecycle(
            &mut observer,
            true,
            || Ok(()),
            || 1,
            |observer| {
                Observer::phase(observer, "cancellation-cleanup");
                report(observer, &cancellation);
                Observer::packed_cache_evidence(
                    observer,
                    &crate::primitives::PackedCacheEvidence {
                        dense_active: true,
                        fallback_reasons: vec![("update".into(), "cancelled".into())],
                        ..test_packed_evidence(1)
                    },
                );
                Observer::release_event(
                    observer,
                    "cache_release",
                    "cache",
                    cancellation.device_bytes(),
                );
                Ok(())
            },
            || {},
        )
        .unwrap();
        let primary = observer.finish().expect("producer sequence finalizes");
        let block =
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary])
                .unwrap();
        assert_eq!(block.persistent_kv_representation, COMPRESSED_PERSISTENT_KV);
        assert_eq!(block.physical_kv_bytes, 64 + 32 + 16);
        assert_eq!(
            (block.device_code_bytes, block.device_metadata_bytes),
            (64, 32)
        );
        assert_eq!(block.storage_tokens, primary.cache_live_tokens);
        // The lifecycle facts are still counted arm-wide.
        assert_eq!(block.fused_calls, 5);
        assert_eq!(block.fallback_calls, 2);
        assert!(block
            .fallbacks
            .iter()
            .any(|fallback| fallback.operation == "prompt-cache-reuse"));
    }

    #[test]
    fn coordinate_storage_is_taken_at_the_persistent_kv_peak() {
        // A host copy released after upload must not pin the physical bytes to an earlier,
        // smaller device share than the persistent KV the receipt reports.
        let mut observer = ProductObserver::new();
        let staged = crate::primitives::CompressedCacheStorage {
            host_payload_bytes: 100,
            ..TEST_STORAGE
        };
        let uploaded = crate::primitives::CompressedCacheStorage {
            device_code_bytes: 20,
            host_payload_bytes: 0,
            tokens: 2,
            ..TEST_STORAGE
        };
        Observer::compressed_storage(&mut observer, &staged);
        Observer::compressed_storage(&mut observer, &uploaded);
        observer.end_coordinate_operation();
        assert_eq!(
            observer.coordinate_scope.unwrap().storage_peak,
            Some(uploaded)
        );
    }

    /// A chunked-prefix-reuse coordinate on a compressed arm: the dense prefix-store seed is setup
    /// recorded before the measured dispatch opens, so an accepted prefix import (compressed
    /// evidence, no fallback inside the dispatch) classifies the row `compressed`. A declined import,
    /// recorded inside the dispatch, classifies it `dense-fallback`. Both keep the seed counted.
    #[test]
    fn chunked_prefix_seed_is_setup_and_an_accepted_import_classifies_compressed() {
        let classify = |declined_import: bool| {
            let mut observer = ProductObserver::new();
            Observer::dense_fallback(
                &mut observer,
                "chunked-prefix-seed",
                COMPRESSED_PREFIX_REUSE_FALLBACK_REASON,
            );
            observer.begin_coordinate_operation();
            if declined_import {
                Observer::dense_fallback(
                    &mut observer,
                    crate::decode::prefix::COMPRESSED_PREFIX_IMPORT_OPERATION,
                    "the compressed cache declined the reused prefix",
                );
            } else {
                Observer::compressed_storage(&mut observer, &TEST_STORAGE);
                Observer::packed_cache_evidence(&mut observer, &test_packed_evidence(4));
            }
            observer.end_coordinate_operation();
            let mut primary = test_compressed_observation(
                observer.packed_evidence.clone(),
                &[],
                observer.compressed_storage_peak,
            );
            primary.dense_fallbacks = observer.dense_fallbacks.clone();
            primary.coordinate_scope = observer.coordinate_scope.take();
            let fixture = test_compressed_observation(vec![test_packed_evidence(2)], &[], None);
            compressed_receipt_block(
                CompressedKvMethod::GroupAffine,
                &primary,
                &[&primary, &fixture],
            )
            .unwrap()
        };
        let accepted = classify(false);
        assert_eq!(
            accepted.persistent_kv_representation,
            COMPRESSED_PERSISTENT_KV
        );
        assert_eq!(accepted.physical_kv_bytes, 3);
        assert!(accepted
            .fallbacks
            .iter()
            .any(|fallback| fallback.operation == "chunked-prefix-seed"));
        assert_eq!(accepted.kernel_gpu_family, "apple7-or-newer");
        let mut receipt = builder_test_receipt();
        compress_test_receipt(&mut receipt, accepted);
        validate_sealed_receipt(&receipt).expect("an imported-prefix compressed row is valid");

        let declined = classify(true);
        assert_eq!(
            declined.persistent_kv_representation,
            DENSE_FALLBACK_PERSISTENT_KV
        );
        assert_eq!(declined.fallback_calls, 2);
    }

    #[test]
    fn compressed_classification_reads_only_the_coordinate_scope() {
        // Out-of-scope lifecycle evidence (a fallback plus larger storage) never reclassifies a
        // compressed coordinate or supplies its bytes.
        let mut primary =
            test_compressed_observation(vec![test_packed_evidence(4)], &[], Some(TEST_STORAGE));
        primary.dense_fallbacks.push((
            "prompt-cache-reuse".into(),
            COMPRESSED_PREFIX_REUSE_FALLBACK_REASON.into(),
        ));
        let warmup = test_compressed_observation(
            vec![test_packed_evidence(2)],
            &[],
            Some(crate::primitives::CompressedCacheStorage {
                host_payload_bytes: 1 << 20,
                tokens: 99,
                ..TEST_STORAGE
            }),
        );
        let block = compressed_receipt_block(
            CompressedKvMethod::GroupAffine,
            &primary,
            &[&primary, &warmup],
        )
        .unwrap();
        assert_eq!(block.persistent_kv_representation, COMPRESSED_PERSISTENT_KV);
        assert_eq!((block.physical_kv_bytes, block.storage_tokens), (3, 1));

        // A coordinate that itself ran dense (batch / prefix reuse) claims no compressed storage,
        // even though the arm's fixtures stayed compressed.
        let batch = test_compressed_observation(
            Vec::new(),
            &[("supported-batch", COMPRESSED_BATCH_FALLBACK_REASON)],
            None,
        );
        let block =
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &batch, &[&batch, &warmup])
                .unwrap();
        assert_eq!(
            block.persistent_kv_representation,
            DENSE_FALLBACK_PERSISTENT_KV
        );
        assert_eq!(
            (
                block.physical_kv_bytes,
                block.device_code_bytes,
                block.storage_tokens
            ),
            (0, 0, 0)
        );
        let mut receipt = builder_test_receipt();
        compress_test_receipt(&mut receipt, block);
        validate_sealed_receipt(&receipt).expect("a reasoned dense coordinate is a valid row");

        // A method is bound to its code width: 2-bit evidence cannot become a 4-bit receipt.
        assert_eq!(
            compressed_receipt_block(CompressedKvMethod::GroupAffine4, &primary, &[&primary])
                .unwrap_err(),
            "compressed arm for group-affine-4 produced 2-bit evidence"
        );
        // ...and to its reader identity: 4-bit evidence under the 2-bit identity is refused.
        for item in &mut primary.packed_evidence {
            item.bits = 4;
        }
        assert_eq!(
            compressed_receipt_block(CompressedKvMethod::GroupAffine4, &primary, &[&primary])
                .unwrap_err(),
            "compressed arm for group-affine-4 produced sc-20676-packed-group-affine-v1 evidence"
        );
        for item in &mut primary.packed_evidence {
            item.bits = 2;
        }

        // A primary that never closed its coordinate scope cannot be classified.
        primary.coordinate_scope = None;
        assert!(
            compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary])
                .unwrap_err()
                .contains("no coordinate-operation scope")
        );
    }

    #[test]
    fn compressed_receipt_method_is_bound_to_its_width_and_identity() {
        let validate = |edit: &dyn Fn(&mut ReceiptCompression)| {
            let mut block = test_compression_block();
            edit(&mut block);
            let mut receipt = builder_test_receipt();
            compress_test_receipt(&mut receipt, block);
            validate_sealed_receipt(&receipt)
        };
        let set = |method: &'static str, bits: u64, identity: &'static str| {
            move |block: &mut ReceiptCompression| {
                block.method = method.into();
                block.bits = bits;
                block.representation_identity = identity.into();
            }
        };
        const B2: &str = "sc-20676-packed-group-affine-v1";
        const B4: &str = "sc-20676-packed-group-affine-b4-v1";
        const B8: &str = "sc-20676-packed-group-affine-b8-v1";
        let table = [
            ("group-affine", 2, B2),
            ("group-affine-4", 4, B4),
            ("group-affine-8", 8, B8),
        ];
        // Every (method, bits, identity) combination: only the method's own pair validates.
        for (method, own_bits, own_identity) in table {
            for (_, bits, _) in table {
                for (_, _, identity) in table {
                    let result = validate(&set(method, bits, identity));
                    if bits == own_bits && identity == own_identity {
                        result.unwrap();
                    } else {
                        let error = result.unwrap_err();
                        assert!(
                            error.contains(&format!("compressed method {method} is")),
                            "{method}/{bits}/{identity}: {error}"
                        );
                    }
                }
            }
        }
        let error = validate(&set("rvq-unwired", 2, B2)).unwrap_err();
        assert!(error.contains("unknown compressed KV method"), "{error}");
    }

    #[test]
    fn compressed_mode_flags_and_resume_identity_bind_the_method() {
        let args = |values: &[&str]| values.iter().map(|v| v.to_string()).collect::<Vec<_>>();
        assert_eq!(compressed_mode_flags(&args(&["worker"])).unwrap(), None);
        assert_eq!(
            compressed_mode_flags(&args(&["worker", "--mode", "dense"])).unwrap(),
            None
        );
        assert_eq!(
            compressed_mode_flags(&args(&[
                "parent",
                "--mode",
                "compressed",
                "--kv-method",
                "group-affine"
            ]))
            .unwrap(),
            Some(CompressedKvMethod::GroupAffine)
        );
        assert_eq!(
            compressed_mode_flags(&args(&[
                "parent",
                "--mode",
                "compressed",
                "--kv-method",
                "group-affine-4"
            ]))
            .unwrap(),
            Some(CompressedKvMethod::GroupAffine4)
        );
        for method in CompressedKvMethod::ALL {
            assert_eq!(CompressedKvMethod::parse(method.id()), Ok(method));
        }
        assert_eq!(
            CompressedKvMethod::GroupAffine.representation_identity(),
            "sc-20676-packed-group-affine-v1"
        );
        assert_eq!(
            CompressedKvMethod::GroupAffine4.representation_identity(),
            "sc-20676-packed-group-affine-b4-v1"
        );
        assert_eq!(
            CompressedKvMethod::GroupAffine8.representation_identity(),
            "sc-20676-packed-group-affine-b8-v1"
        );
        assert_eq!(
            compressed_mode_flags(&args(&[
                "parent",
                "--mode",
                "compressed",
                "--kv-method",
                "group-affine-8"
            ]))
            .unwrap(),
            Some(CompressedKvMethod::GroupAffine8)
        );
        for bad in [
            &["--mode", "compressed"][..],
            &["--kv-method", "group-affine"],
            &["--mode", "compressed", "--kv-method", "rvq-unwired"],
            &["--mode", "sparse"],
        ] {
            assert!(compressed_mode_flags(&args(bad)).is_err(), "{bad:?}");
        }
        let mut identity = serde_json::json!({ "kind": "sc-20671-resume-identity" });
        let dense_bytes = canonical_json_bytes(&identity).unwrap();
        bind_resume_mode(&mut identity, None);
        assert_eq!(canonical_json_bytes(&identity).unwrap(), dense_bytes);
        assert_eq!(resume_identity_mode(&identity).unwrap(), None);
        bind_resume_mode(&mut identity, Some(CompressedKvMethod::GroupAffine));
        assert_eq!(
            resume_identity_mode(&identity).unwrap(),
            Some(CompressedKvMethod::GroupAffine)
        );
        let two_bit = canonical_json_bytes(&identity).unwrap();
        bind_resume_mode(&mut identity, Some(CompressedKvMethod::GroupAffine4));
        assert_eq!(
            resume_identity_mode(&identity).unwrap(),
            Some(CompressedKvMethod::GroupAffine4)
        );
        assert_ne!(canonical_json_bytes(&identity).unwrap(), two_bit);
        identity["kvMethod"] = serde_json::Value::Null;
        assert!(resume_identity_mode(&identity).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn group_affine_fused_reader_parity_is_within_the_frozen_contract() {
        for method in CompressedKvMethod::ALL {
            let errors = method.arm().unwrap().kernel_parity_errors().unwrap();
            // Decode (S_q = 1), causal prefill chunk (S_q = 5), and split-KV decode (S_q = 1).
            assert_eq!(errors.len(), 4 * 128 * (1 + 5 + 1));
            let max = errors.iter().copied().fold(0.0, f64::max);
            assert!(
                max <= COMPRESSED_PARITY_MAX_ERROR,
                "{}: fused reader parity {max}",
                method.id()
            );
        }
    }

    #[test]
    fn artifact_bundle_rejects_tampering_and_partial_outputs() {
        let mut receipt = builder_test_receipt();
        assert_eq!(receipt.provenance.model_file_bytes, 100);
        assert_eq!(receipt.memory.model_weights_bytes, 1);
        let mut padded = receipt.clone();
        padded.geometry.capacity = 256;
        padded.memory.persistent_kv_bytes = 1024;
        padded.memory.dense_theoretical_kv_bytes = 1024;
        padded.memory.reconciliation.expected_dense_kv_bytes = 1024;
        padded.memory.reconciliation.observed_persistent_kv_bytes = 1024;
        for event in &mut padded.memory.allocation_events {
            if event.role == "cache" {
                event.bytes = 1024;
            }
        }
        for sample in padded.memory.phase_samples.iter_mut().take(6).skip(2) {
            sample.mlx.active_bytes = 1027;
            sample.phys_footprint_bytes = 2000;
        }
        for sample in padded.memory.phase_samples.iter_mut().skip(2) {
            sample.mlx.peak_bytes = 1028;
            sample.phys_footprint_peak_bytes = 2000;
        }
        (
            padded.memory.dense_kv_share_bps,
            padded.memory.below_memory_material_share,
        ) = dense_kv_share(
            1024,
            padded.memory.phase_samples[2].phys_footprint_bytes,
            &padded.matrix.context_band,
        )
        .unwrap();
        padded.receipt_sha256 = receipt_semantic_seal(&padded).unwrap();
        validate_receipt_semantics(&padded).expect("physical block capacity reconciles exactly");
        let mut loosened_release = receipt.clone();
        loosened_release
            .memory
            .release
            .phys_footprint_tolerance_bytes += 1;
        assert_eq!(
            validate_receipt_semantics(&loosened_release).unwrap_err(),
            "release tolerances or residuals differ from the platform contract"
        );
        // A small allocator residual is recorded and accepted; a material leak still fails.
        let released_with = |active: u64, cache: u64| {
            let mut released = receipt.clone();
            let end = released.memory.phase_samples.last_mut().unwrap();
            end.mlx.active_bytes = active;
            end.mlx.cache_bytes = cache;
            end.mlx.peak_bytes = end.mlx.peak_bytes.max(active);
            end.phys_footprint_bytes = end.phys_footprint_bytes.max(active + cache);
            end.phys_footprint_peak_bytes = end.phys_footprint_peak_bytes.max(active + cache);
            released.memory.release = release_evidence(
                &released.memory.phase_samples[1],
                released.memory.phase_samples.last().unwrap(),
            );
            released
        };
        let slack = post_release_mlx_slack_bytes(3);
        assert_eq!(slack, POST_RELEASE_MLX_SLACK_FLOOR_BYTES);
        let residual = released_with(3 + slack, 0);
        assert_eq!(residual.memory.release.mlx_active_residual_bytes, slack);
        validate_receipt_semantics(&residual).expect("a residual within slack is recorded");
        let mut unrecorded = residual.clone();
        unrecorded.memory.release.mlx_active_residual_bytes = 0;
        assert!(validate_receipt_semantics(&unrecorded).is_err());
        let leak = released_with(3 + slack + 1, 0);
        assert!(!leak.memory.release.verified);
        let release_error = validate_receipt_semantics(&leak).unwrap_err();
        assert!(
            release_error.contains("release did not return within tolerance"),
            "{release_error}"
        );
        let cache_leak = released_with(3, slack + 1);
        assert!(validate_receipt_semantics(&cache_leak).is_err());
        assert_eq!(post_release_mlx_slack_bytes(4_000_000_000), 4_000_000);
        assert_eq!(post_release_mlx_slack_bytes(4_000_000_001), 4_000_001);
        let mut reference_only_memory = receipt.clone();
        reference_only_memory.memory.phase_samples[0]
            .mlx
            .active_bytes = 100;
        reference_only_memory.memory.phase_samples[0].mlx.peak_bytes = 100;
        reference_only_memory.memory.phase_samples[1]
            .mlx
            .active_bytes = 100;
        reference_only_memory.memory.phase_samples[1].mlx.peak_bytes = 100;
        for phase in &mut reference_only_memory.memory.phase_samples[2..] {
            phase.mlx.peak_bytes = phase.mlx.peak_bytes.max(100);
        }
        assert!(validate_receipt_semantics(&reference_only_memory).is_err());
        let mut different_session = receipt.clone();
        different_session.provenance.campaign_session_id = "7".repeat(64);
        assert_eq!(
            campaign_family_identity(&receipt).unwrap(),
            campaign_family_identity(&different_session).unwrap(),
            "volatile product sessions must not alter stable family/model identity"
        );
        let mut source_drift = receipt.clone();
        source_drift.provenance.inference_revision = "8".repeat(40);
        assert_ne!(
            campaign_global_identity(&receipt).unwrap(),
            campaign_global_identity(&source_drift).unwrap()
        );
        let mut tampered = receipt.clone();
        tampered.geometry.capacity = 2;
        assert!(validate_receipt_semantics(&tampered).is_err());
        let mut widened = receipt.clone();
        widened.memory.reconciliation.tolerance_bytes = 1;
        assert!(validate_receipt_semantics(&widened).is_err());
        let mut timing_tampered = receipt.clone();
        timing_tampered.timings.decode_tokens_per_second += 1.0;
        assert!(validate_receipt_semantics(&timing_tampered).is_err());
        // Each sample's throughput is exactly its recorded fixed-length steady decode.
        let steady_error = |edit: &dyn Fn(&mut ReceiptTimingSample)| {
            let mut tampered = receipt.clone();
            edit(&mut tampered.timings.samples[2]);
            validate_receipt_semantics(&tampered).unwrap_err()
        };
        for edit in [
            &(|s: &mut ReceiptTimingSample| s.steady_decode_ms *= 2.0) as &dyn Fn(&mut _),
            &|s: &mut ReceiptTimingSample| {
                // A short EOS-terminated generation, internally consistent, is still refused.
                s.steady_decode_generated_tokens = 24;
                s.steady_decode_timed_tokens = 23;
                s.decode_tokens_per_second = 23.0 * 1_000.0 / s.steady_decode_ms;
            },
            &|s: &mut ReceiptTimingSample| s.steady_decode_timed_tokens = STEADY_DECODE_TOKENS,
            &|s: &mut ReceiptTimingSample| s.steady_decode_prompt_tokens = 4_000,
            &|s: &mut ReceiptTimingSample| s.steady_decode_forced_stop_tokens = 257,
        ] {
            let error = steady_error(edit);
            assert!(error.contains("steady decode"), "{error}");
        }
        // Power mode and thermal state are recorded at row start, each timing sample, and row
        // end. Only a throttled row start is refused; later changes are recorded and flagged.
        let host_error = |edit: &dyn Fn(&mut ReceiptProvenance)| {
            let mut tampered = receipt.clone();
            edit(&mut tampered.provenance);
            validate_receipt_semantics(&tampered).unwrap_err()
        };
        let mut warmed = receipt.clone();
        warmed.provenance.host_states[1].thermal_state = "serious".into();
        warmed.provenance.host_states[1].throttled = true;
        warmed.provenance.host_states[1].power_mode = "low-power".into();
        warmed.provenance.thermal_changed_during_row = true;
        warmed.provenance.power_mode_changed_during_row = true;
        validate_receipt_semantics(&warmed).expect("a row-end change is recorded, not refused");
        let mut sample_change = receipt.clone();
        sample_change.timings.samples[2].host_state.thermal_state = "fair".into();
        assert!(validate_receipt_semantics(&sample_change)
            .unwrap_err()
            .contains("change flags do not recompute"));
        sample_change.provenance.thermal_changed_during_row = true;
        validate_receipt_semantics(&sample_change).expect("a timing-sample change is recorded");
        let mut sample_order = receipt.clone();
        sample_order.timings.samples.swap(0, 4);
        assert!(validate_receipt_semantics(&sample_order).is_err());
        for edit in [
            &(|p: &mut ReceiptProvenance| p.host_states[1].thermal_state = "fair".into())
                as &dyn Fn(&mut _),
            &|p: &mut ReceiptProvenance| p.host_states[1].power_mode = "low-power".into(),
            &|p: &mut ReceiptProvenance| {
                p.host_states[0].thermal_state = "critical".into();
                p.host_states[0].throttled = true;
                p.thermal_state = "critical".into();
            },
            &|p: &mut ReceiptProvenance| {
                p.host_states[0].pmset_thermal_raw = "CPU_Speed_Limit = 50\n".into();
                p.host_states[0].cpu_speed_limit = Some(50);
                p.host_states[0].throttled = true;
            },
            &|p: &mut ReceiptProvenance| p.host_states[1].throttled = true,
            &|p: &mut ReceiptProvenance| p.thermal_state = "fair".into(),
            &|p: &mut ReceiptProvenance| p.host_states.reverse(),
            &|p: &mut ReceiptProvenance| {
                p.host_states.pop();
            },
            &|p: &mut ReceiptProvenance| {
                p.host_states[0].captured_at = "2026-01-01T00:00:03.000Z".into()
            },
            &|p: &mut ReceiptProvenance| {
                p.host_states[1].captured_at = "2026-01-01T00:00:05.000Z".into()
            },
            &|p: &mut ReceiptProvenance| {
                p.power_mode = "nominal".into();
                for state in &mut p.host_states {
                    state.power_mode = "nominal".into();
                }
            },
        ] {
            let error = host_error(edit);
            assert!(error.contains("host power/thermal"), "{error}");
        }
        let mut unstable_timing = receipt.clone();
        unstable_timing
            .timings
            .summary
            .decode_tokens_per_second_coefficient_of_variation = 0.1;
        assert!(validate_receipt_semantics(&unstable_timing)
            .unwrap_err()
            .contains("coefficientOfVariation=0.1, maximum=0.05"));
        let mut quality_tampered = receipt.clone();
        quality_tampered.mode = "compressed".into();
        quality_tampered.provenance.command = "run --mode compressed".into();
        quality_tampered.quality.parity_max_error = 0.1;
        assert!(validate_receipt_semantics(&quality_tampered)
            .unwrap_err()
            .contains("kernel parity failed: metric=parityMaxError value=0.1 threshold=0.0001"));
        // Contract v3: compressed quality is gated only against the same-weights dense-KV run.
        let same_weights = |receipt: &mut Receipt| {
            for name in REQUIRED_FIXTURES {
                let reference = fixture_independent_reference(
                    name,
                    QualityReference::DenseKvSameWeights,
                    &receipt.provenance.model_file_sha256,
                );
                receipt
                    .quality
                    .fixture_evidence
                    .get_mut(name)
                    .unwrap()
                    .independent_reference = reference;
            }
        };
        let mut compressed = receipt.clone();
        compressed.mode = "compressed".into();
        compressed.provenance.command = "run --mode compressed".into();
        assert!(validate_receipt_semantics(&compressed)
            .unwrap_err()
            .contains("contract v3 denominator"));
        same_weights(&mut compressed);
        assert!(validate_receipt_semantics(&compressed)
            .unwrap_err()
            .contains("compression evidence must be present"));
        compress_test_receipt(&mut compressed, test_compression_block());
        validate_receipt_semantics(&compressed).expect("same-weights compressed receipt");
        let mut compressed_miss = compressed.clone();
        compressed_miss.quality.needle_retrieval = 0.0;
        assert!(validate_receipt_semantics(&compressed_miss).is_err());
        let mut dense_claiming_gate = receipt.clone();
        same_weights(&mut dense_claiming_gate);
        assert!(validate_receipt_semantics(&dense_claiming_gate)
            .unwrap_err()
            .contains("contract v3 denominator"));
        // Dense rows record a needle miss as characterization, never a rejection, but may not
        // hide it behind a discriminating flag.
        let mut recorded_miss = receipt.clone();
        recorded_miss.quality.needle_retrieval = 0.0;
        recorded_miss.quality.needle_discriminating = false;
        recorded_miss.quality.structured_tool_agreement = 0.0;
        validate_receipt_semantics(&recorded_miss).expect("dense quality is never gated");
        let mut hidden_miss = recorded_miss.clone();
        hidden_miss.quality.needle_discriminating = true;
        assert!(validate_receipt_semantics(&hidden_miss)
            .unwrap_err()
            .contains("needle discrimination"));
        let mut unguarded = receipt.clone();
        unguarded.memory.admission.child_footprint_cap_bytes = 0;
        assert!(validate_receipt_semantics(&unguarded)
            .unwrap_err()
            .contains("runtime-guarded"));
        let mut over_cap_estimate = receipt.clone();
        over_cap_estimate
            .memory
            .admission
            .static_footprint_floor_bytes =
            over_cap_estimate.memory.admission.child_footprint_cap_bytes + 1;
        assert!(validate_receipt_semantics(&over_cap_estimate).is_err());
        // The estimate-plus-reserve decision is mandatory and self-consistent.
        for (label, mutate) in [
            (
                "another rule",
                (|a: &mut ReceiptAdmission| a.rule = "cap-plus-reserve".into())
                    as fn(&mut ReceiptAdmission),
            ),
            ("no estimate source", |a| a.estimate_source.clear()),
            ("estimate below the static floor", |a| {
                a.estimate_bytes = a.static_footprint_floor_bytes - 1
            }),
            ("estimate above the cap", |a| {
                a.estimate_bytes = a.child_footprint_cap_bytes + 1
            }),
            ("cap fallback below the cap", |a| {
                a.estimate_source = campaign_supervisor::CAP_FALLBACK_ESTIMATE_SOURCE.into()
            }),
        ] {
            let mut mutated = receipt.clone();
            mutate(&mut mutated.memory.admission);
            assert!(
                validate_receipt_semantics(&mutated)
                    .unwrap_err()
                    .contains(campaign_supervisor::ESTIMATE_PLUS_RESERVE_RULE),
                "{label}"
            );
        }
        // A record lacking any estimate field (a pre-v1 cap-plus-reserve admission) is refused.
        for field in ["rule", "estimateSource", "estimateBytes"] {
            let mut value = serde_json::to_value(&receipt.memory.admission).unwrap();
            value.as_object_mut().unwrap().remove(field);
            assert!(
                serde_json::from_value::<ReceiptAdmission>(value).is_err(),
                "{field}"
            );
        }
        // A receipt must carry a host measurement that recomputes and covers estimate plus
        // reserve: exactly reserve (1 GiB) + estimate (1 MiB) is admitted, one page less is not,
        // although both are below the former cap (1 GiB) plus reserve.
        let host_pages = |free| {
            campaign_supervisor::HostMemory::from_pages(
                16_384,
                campaign_supervisor::VmStatPages {
                    free,
                    ..Default::default()
                },
            )
        };
        let mut exact_host = receipt.clone();
        exact_host.memory.admission.host_memory_components =
            host_pages(((1 << 30) + (1 << 20)) / 16_384);
        validate_receipt_semantics(&exact_host).unwrap();
        let mut unmeasured = receipt.clone();
        unmeasured.memory.admission.host_memory_components = None;
        assert!(validate_receipt_semantics(&unmeasured)
            .unwrap_err()
            .contains("host memory components"));
        let mut tampered_host = receipt.clone();
        tampered_host
            .memory
            .admission
            .host_memory_components
            .as_mut()
            .unwrap()
            .reclaimable_file_pages = 90_000;
        assert!(validate_receipt_semantics(&tampered_host)
            .unwrap_err()
            .contains("recompute"));
        let mut short_host = receipt.clone();
        short_host.memory.admission.host_memory_components =
            host_pages(((1 << 30) + (1 << 20)) / 16_384 - 1);
        let short = validate_receipt_semantics(&short_host).unwrap_err();
        assert!(short.contains("is below reserve plus"), "{short}");
        // Artifacts: the compressed denominator is bound to the candidate's own inventory.
        let binding = |receipt: &Receipt, reference_inventory: &str| {
            let arm = |inventory: &str, session: &str| {
                serde_json::json!({
                    "coordinateInventorySha256": inventory,
                    "qualityInventorySha256": inventory,
                    "coordinateSessionId": session,
                    "qualitySessionId": session,
                    "operation": "single-shot-generation",
                    "operationOutputSha256": "a".repeat(64),
                    "operationEvidenceSha256": "a".repeat(64),
                    "coordinateEvidenceSha256": "a".repeat(64),
                    "qualityTranscriptSha256": "a".repeat(64),
                    "secondaryOperation": null,
                })
            };
            serde_json::to_vec(&serde_json::json!({ "binding": {
                "coordinate": "llama-short-single-single-shot-cold",
                "repeat": 1,
                "candidate": arm(
                    &receipt.provenance.model_file_sha256,
                    &receipt.provenance.campaign_session_id,
                ),
                "reference": arm(reference_inventory, &"b".repeat(64)),
            }}))
            .unwrap()
        };
        let bf16 = receipt.provenance.reference_model_sha256.clone();
        let own = receipt.provenance.model_file_sha256.clone();
        let check = |receipt: &Receipt, inventory: &str| {
            validate_fixture_binding(
                receipt,
                "long-context-needle",
                &binding(receipt, inventory),
                1,
            )
        };
        check(&receipt, &bf16).expect("dense rows characterize against bf16");
        assert!(check(&receipt, &own).is_err());
        check(&compressed, &own).expect("compressed denominator is the same weights");
        assert!(check(&compressed, &bf16)
            .unwrap_err()
            .contains("producer binding mismatch"));
        // Receipt discrimination is the AND over all repeats, each re-derived from its outcomes.
        let needle = |candidate: bool, reference: bool, outputs: bool, matches: u64, flag: bool| {
            serde_json::to_vec(&serde_json::json!({ "evidence": {
                "matches": matches, "total": 1, "candidateRecovered": candidate,
                "referenceRecovered": reference, "outputsMatch": outputs, "discriminating": flag,
            }}))
            .unwrap()
        };
        let tool = |candidate: bool, reference: bool, outputs: bool, flag: bool| {
            serde_json::to_vec(&serde_json::json!({ "evidence": {
                "matches": u64::from(outputs), "total": 1, "candidateValid": candidate,
                "referenceValid": reference, "outputsMatch": outputs, "discriminating": flag,
            }}))
            .unwrap()
        };
        let hit = needle(true, true, true, 1, true);
        // Contract v5: a shared miss records the candidate's own (failed) recovery, never the
        // v4 agreement with the dense miss text.
        let shared_miss = needle(false, false, true, 0, false);
        assert!(validate_repeat_discrimination(
            &compressed,
            [(
                "long-context-needle",
                needle(false, false, true, 1, false).as_slice()
            )],
        )
        .is_err());
        let valid_tool = tool(true, true, true, true);
        let shared_invalid_tool = tool(false, false, true, false);
        fn repeats<'a>(
            needles: &[&'a Vec<u8>],
            tools: &[&'a Vec<u8>],
        ) -> Vec<(&'static str, &'a [u8])> {
            needles
                .iter()
                .map(|bytes| ("long-context-needle", bytes.as_slice()))
                .chain(
                    tools
                        .iter()
                        .map(|bytes| ("structured-tool-call", bytes.as_slice())),
                )
                .collect()
        }
        let mut all_discriminating = compressed.clone();
        all_discriminating.quality.needle_discriminating = true;
        all_discriminating.quality.tool_discriminating = true;
        validate_repeat_discrimination(
            &all_discriminating,
            repeats(&[&hit, &hit], &[&valid_tool, &valid_tool]),
        )
        .unwrap();
        // One non-discriminating repeat makes the row non-discriminating; claiming otherwise fails.
        for (needles, tools) in [
            (
                repeats(&[&hit, &shared_miss], &[&valid_tool, &valid_tool]),
                "needle",
            ),
            (
                repeats(&[&hit, &hit], &[&valid_tool, &shared_invalid_tool]),
                "tool",
            ),
        ] {
            assert!(
                validate_repeat_discrimination(&all_discriminating, needles).is_err(),
                "{tools} repeat disagreement hidden by the receipt flag"
            );
        }
        assert!(validate_repeat_discrimination(
            &all_discriminating,
            repeats(&[&hit, &shared_miss], &[&valid_tool, &shared_invalid_tool]),
        )
        .is_err());
        let mut anded = all_discriminating.clone();
        anded.quality.needle_discriminating = false;
        anded.quality.tool_discriminating = false;
        validate_repeat_discrimination(
            &anded,
            repeats(&[&hit, &shared_miss], &[&valid_tool, &shared_invalid_tool]),
        )
        .unwrap();
        // A validly measured compressed miss (dense recovered / emitted the call, compressed did
        // not) is gate evidence, not a malformed artifact: its derivation is accepted here and the
        // miss is recorded by the quality gate.
        for (fixture, measured_miss) in [
            ("long-context-needle", needle(false, true, false, 0, true)),
            ("structured-tool-call", tool(true, true, false, true)),
        ] {
            validate_repeat_discrimination(
                &all_discriminating,
                [(fixture, measured_miss.as_slice())],
            )
            .unwrap_or_else(|error| panic!("{fixture} measured miss refused: {error}"));
        }
        // Per-artifact derivation: flag must follow the same-weights dense run and a
        // non-discriminating compressed repeat must match the dense output.
        for (fixture, forged, flag) in [
            // Shared miss whose outputs differ from dense, counted as a match.
            (
                "long-context-needle",
                needle(false, false, false, 1, false),
                false,
            ),
            // Dense recovered, yet flagged non-discriminating.
            (
                "long-context-needle",
                needle(true, true, true, 1, false),
                false,
            ),
            // Discriminating compressed repeat that lost the needle, counted as a match.
            (
                "long-context-needle",
                needle(false, true, false, 1, true),
                true,
            ),
            // Dense tool call invalid, yet flagged discriminating.
            ("structured-tool-call", tool(true, false, true, true), true),
        ] {
            // The receipt carries the artifact's own flag, so only the derivation can fail.
            let mut claimed = compressed.clone();
            claimed.quality.needle_discriminating = fixture != "long-context-needle" || flag;
            claimed.quality.tool_discriminating = fixture != "structured-tool-call" || flag;
            assert!(
                validate_repeat_discrimination(&claimed, [(fixture, forged.as_slice())]).is_err(),
                "{fixture} forgery accepted"
            );
        }
        // Dense rows derive discrimination from their own run and are never gated on outcomes.
        let mut dense_miss = receipt.clone();
        dense_miss.quality.needle_discriminating = false;
        dense_miss.quality.tool_discriminating = false;
        validate_repeat_discrimination(
            &dense_miss,
            repeats(
                &[&needle(false, true, false, 0, false)],
                &[&tool(false, true, false, false)],
            ),
        )
        .unwrap();
        // Admission must carry the captured policy's cap and reserve.
        let policy = CampaignSafetyPolicy {
            schema_version: 1,
            row_deadline_seconds: 10,
            poll_millis: 100,
            term_grace_millis: 500,
            host_free_reserve_bytes: receipt.memory.admission.host_free_reserve_bytes,
            child_footprint_cap_bytes: receipt.memory.admission.child_footprint_cap_bytes,
            max_context_tokens: 4096,
            max_request_tokens: 4096,
            stdout_cap_bytes: 4096,
            stderr_cap_bytes: 4096,
        };
        receipt.memory.admission.validate_against(&policy).unwrap();
        for other in [
            CampaignSafetyPolicy {
                child_footprint_cap_bytes: policy.child_footprint_cap_bytes + 1,
                ..policy.clone()
            },
            CampaignSafetyPolicy {
                host_free_reserve_bytes: policy.host_free_reserve_bytes + 1,
                ..policy.clone()
            },
        ] {
            assert!(receipt
                .memory
                .admission
                .validate_against(&other)
                .unwrap_err()
                .contains("captured safety policy"));
        }
        let mut lifecycle_tampered = receipt.clone();
        lifecycle_tampered.lifecycle.append = false;
        assert!(validate_receipt_semantics(&lifecycle_tampered).is_err());
        let mut phase_tampered = receipt.clone();
        phase_tampered.memory.phase_samples[3].source = "caller-authored".into();
        assert!(validate_receipt_semantics(&phase_tampered).is_err());
        let mut event_outside_phase = receipt.clone();
        event_outside_phase.memory.allocation_events[1].timestamp =
            "2026-01-01T00:00:03.100Z".into();
        assert!(validate_receipt_semantics(&event_outside_phase).is_err());
        let mut event_at_phase_start = receipt.clone();
        event_at_phase_start.memory.allocation_events[1].timestamp =
            event_at_phase_start.memory.phase_samples[2]
                .timestamp
                .clone();
        assert!(validate_receipt_semantics(&event_at_phase_start).is_err());
        let mut equivalent_fractional_phase_start = receipt.clone();
        equivalent_fractional_phase_start.memory.phase_samples[2].timestamp =
            "2026-01-01T00:00:02.0000Z".into();
        equivalent_fractional_phase_start.memory.allocation_events[1].timestamp =
            "2026-01-01T00:00:02.000Z".into();
        assert!(validate_receipt_semantics(&equivalent_fractional_phase_start).is_err());
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
        threshold_tampered.mode = "compressed".into();
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
        let mut sequential = receipt.clone();
        let release = sequential
            .memory
            .allocation_events
            .pop()
            .expect("release event");
        sequential.memory.allocation_events.extend([
            ReceiptAllocation {
                kind: "product-dense_concat_coexistence".into(),
                role: "output".into(),
                lifetime: "transient".into(),
                phase: "prefill-peak".into(),
                timestamp: "2026-01-01T00:00:02.300Z".into(),
                bytes: 3,
            },
            ReceiptAllocation {
                kind: "product-dense_concat_coexistence".into(),
                role: "output".into(),
                lifetime: "transient".into(),
                phase: "prefill-peak".into(),
                timestamp: "2026-01-01T00:00:02.400Z".into(),
                bytes: 2,
            },
            ReceiptAllocation {
                kind: "product-dense_concat_coexistence".into(),
                role: "output".into(),
                lifetime: "transient".into(),
                phase: "decode-steady".into(),
                timestamp: "2026-01-01T00:00:04.200Z".into(),
                bytes: 2,
            },
            release,
        ]);
        sequential.memory.transient_workspace_bytes = 3;
        sequential.memory.phase_samples[2].mlx.active_bytes = 10;
        for phase in &mut sequential.memory.phase_samples[2..] {
            phase.mlx.peak_bytes = 10;
        }
        assert!(validate_receipt_semantics(&sequential).is_ok());
        sequential.memory.transient_workspace_bytes = 2;
        assert!(validate_receipt_semantics(&sequential).is_err());
        let mut missing_release = receipt.clone();
        missing_release
            .memory
            .allocation_events
            .retain(|event| event.lifetime != "released");
        assert!(validate_receipt_semantics(&missing_release).is_err());
        let mut mismatched_release = receipt.clone();
        mismatched_release
            .memory
            .allocation_events
            .iter_mut()
            .find(|event| event.lifetime == "released")
            .unwrap()
            .bytes = 3;
        assert!(validate_receipt_semantics(&mismatched_release).is_err());
        assert!(dense_kv_bytes(u64::MAX, 2, 2, 2, 2, 2).is_err());
        let mut zero_context = receipt.clone();
        zero_context.geometry.context_window_tokens = 0;
        zero_context.geometry.context_target_tokens = 0;
        zero_context.geometry.context_payload_tokens = 0;
        assert!(validate_receipt_semantics(&zero_context).is_err());
        let mut non_material = receipt.clone();
        non_material.matrix.context_band = "memory-material".into();
        non_material.geometry.context_target_tokens = context_band_target(
            non_material.geometry.context_window_tokens,
            "memory-material",
        )
        .unwrap();
        non_material.geometry.context_payload_tokens =
            non_material.geometry.context_target_tokens / 2;
        assert!(validate_receipt_semantics(&non_material).is_err());
        // With its share recorded, the same memory-material row validates: a low dense-KV share
        // of the prefill footprint is flagged, never refused.
        let mut recorded = non_material.clone();
        (
            recorded.memory.dense_kv_share_bps,
            recorded.memory.below_memory_material_share,
        ) = dense_kv_share(
            recorded.memory.dense_theoretical_kv_bytes,
            recorded.memory.phase_samples[2].phys_footprint_bytes,
            "memory-material",
        )
        .unwrap();
        assert!(recorded.memory.below_memory_material_share);
        let coordinate = format!(
            "{}-{}-{}-{}-{}",
            recorded.matrix.family,
            recorded.matrix.context_band,
            recorded.matrix.request_mode,
            recorded.matrix.prefill_mode,
            recorded.matrix.process_temperature
        );
        for evidence in &mut recorded.timings.compile_attribution.probe_evidence {
            evidence.matrix_coordinate = coordinate.clone();
        }
        validate_receipt_semantics(&recorded).expect("a low share is recorded, not refused");
        let mut not_near_fit = receipt.clone();
        not_near_fit.matrix.context_band = "fit-boundary".into();
        not_near_fit.geometry.context_target_tokens =
            context_band_target(not_near_fit.geometry.context_window_tokens, "fit-boundary")
                .unwrap();
        not_near_fit.geometry.context_payload_tokens =
            not_near_fit.geometry.context_target_tokens / 2;
        assert!(validate_receipt_semantics(&not_near_fit).is_err());
        let mut split_prefill = receipt.clone();
        split_prefill.memory.phase_samples[2].mlx.active_bytes = 3;
        split_prefill.memory.phase_samples[2].mlx.peak_bytes = 4;
        assert!(validate_receipt_semantics(&split_prefill).is_err());
        let mut missing_prefill_workspace_peak = receipt.clone();
        missing_prefill_workspace_peak.memory.phase_samples[2]
            .mlx
            .active_bytes = 7;
        missing_prefill_workspace_peak.memory.phase_samples[2]
            .mlx
            .peak_bytes = 7;
        assert!(validate_receipt_semantics(&missing_prefill_workspace_peak).is_err());
        let mut missing_prefill_persistent_byte = receipt.clone();
        missing_prefill_persistent_byte.memory.phase_samples[2]
            .mlx
            .active_bytes = 6;
        missing_prefill_persistent_byte.memory.phase_samples[2]
            .mlx
            .peak_bytes = 8;
        assert!(validate_receipt_semantics(&missing_prefill_persistent_byte).is_err());
        let mut missing_prefill_cache_snapshot = receipt.clone();
        missing_prefill_cache_snapshot
            .memory
            .allocation_events
            .retain(|event| {
                !(event.phase == "prefill-peak"
                    && event.role == "cache"
                    && event.lifetime == "persistent")
            });
        assert!(validate_receipt_semantics(&missing_prefill_cache_snapshot).is_err());
        let mut stale_prefill_window = receipt.clone();
        stale_prefill_window
            .memory
            .prefill_peak_window
            .reset_peak_bytes = 8;
        assert!(
            validate_receipt_semantics(&stale_prefill_window).is_err(),
            "a session-global load or warmup peak must not masquerade as measured prefill workspace"
        );
        let mut late_prefill_window = receipt.clone();
        late_prefill_window.memory.prefill_peak_window.started_at =
            "2026-01-01T00:00:02.500Z".into();
        assert!(validate_receipt_semantics(&late_prefill_window).is_err());
        let mut missing_prefill_baseline = receipt.clone();
        missing_prefill_baseline
            .memory
            .prefill_peak_window
            .baseline_active_bytes = 2;
        assert!(validate_receipt_semantics(&missing_prefill_baseline).is_err());
        let mut released_prefill_workspace = receipt.clone();
        released_prefill_workspace.memory.phase_samples[2]
            .mlx
            .active_bytes = 7;
        released_prefill_workspace.memory.phase_samples[2]
            .mlx
            .peak_bytes = 8;
        assert!(
            validate_receipt_semantics(&released_prefill_workspace).is_ok(),
            "the post-prefill active sample need only retain weights and KV when peak memory proves the transient workspace"
        );
        let mut phase_local_peak_reset = receipt.clone();
        phase_local_peak_reset.memory.phase_samples[1]
            .mlx
            .peak_bytes = 99;
        assert!(
            validate_receipt_semantics(&phase_local_peak_reset).is_ok(),
            "the sealed prefill reset boundary permits the measured peak to fall below the earlier load high-water"
        );
        let mut shorter_prefill_cache = receipt.clone();
        shorter_prefill_cache
            .memory
            .allocation_events
            .iter_mut()
            .find(|event| {
                event.phase == "prefill-peak"
                    && event.role == "cache"
                    && event.lifetime == "persistent"
            })
            .unwrap()
            .bytes = 3;
        shorter_prefill_cache.memory.phase_samples[2]
            .mlx
            .active_bytes = 6;
        shorter_prefill_cache.memory.phase_samples[2].mlx.peak_bytes = 7;
        assert!(
            validate_receipt_semantics(&shorter_prefill_cache).is_ok(),
            "prefill containment must use the live prefill cache, not the larger later decode cache"
        );
        let mut phase_separated_allocations = shorter_prefill_cache.clone();
        let workspace = phase_separated_allocations
            .memory
            .allocation_events
            .iter_mut()
            .find(|event| event.role == "attention-workspace")
            .unwrap();
        workspace.phase = "decode-steady".into();
        workspace.timestamp = "2026-01-01T00:00:04.200Z".into();
        assert!(
            validate_receipt_semantics(&phase_separated_allocations).is_ok(),
            "zero prefill workspace and nonzero decode workspace must use their own phase floors"
        );
        let mut oversized_prefill_cache = receipt.clone();
        oversized_prefill_cache.memory.allocation_events[1].bytes = 5;
        assert!(validate_receipt_semantics(&oversized_prefill_cache).is_err());
        let mut stale_decode_cache = receipt.clone();
        stale_decode_cache.memory.allocation_events[3].bytes = 3;
        assert!(validate_receipt_semantics(&stale_decode_cache).is_err());
        let mut missing_decode_active = phase_separated_allocations.clone();
        missing_decode_active.memory.phase_samples[4]
            .mlx
            .active_bytes = 6;
        assert!(validate_receipt_semantics(&missing_decode_active).is_err());
        let mut missing_decode_peak = phase_separated_allocations.clone();
        missing_decode_peak.memory.phase_samples[4].mlx.peak_bytes = 7;
        assert!(validate_receipt_semantics(&missing_decode_peak).is_err());
        let mut cancellation_attribution = receipt.clone();
        cancellation_attribution.memory.allocation_events.extend([
            ReceiptAllocation {
                kind: "cancellation-cache-snapshot".into(),
                role: "cache".into(),
                lifetime: "persistent".into(),
                phase: "cancellation-cleanup".into(),
                timestamp: "2026-01-01T00:00:06.100Z".into(),
                bytes: 4,
            },
            ReceiptAllocation {
                kind: "cancellation-workspace".into(),
                role: "output".into(),
                lifetime: "transient".into(),
                phase: "cancellation-cleanup".into(),
                timestamp: "2026-01-01T00:00:06.200Z".into(),
                bytes: 2,
            },
            ReceiptAllocation {
                kind: "product-cache_release".into(),
                role: "cache".into(),
                lifetime: "released".into(),
                phase: "cancellation-cleanup".into(),
                timestamp: "2026-01-01T00:00:06.300Z".into(),
                bytes: 4,
            },
        ]);
        cancellation_attribution.memory.transient_workspace_bytes = 2;
        assert!(validate_receipt_semantics(&cancellation_attribution).is_err());
        cancellation_attribution.memory.phase_samples[6]
            .mlx
            .peak_bytes = 9;
        cancellation_attribution.memory.phase_samples[7]
            .mlx
            .peak_bytes = 9;
        assert!(validate_receipt_semantics(&cancellation_attribution).is_ok());
        let mut fixture_tampered = receipt.clone();
        fixture_tampered
            .quality
            .fixture_evidence
            .remove(REQUIRED_FIXTURES[0]);
        assert!(validate_receipt_semantics(&fixture_tampered).is_err());
        let mut partial = receipt.clone();
        partial.timings.samples.pop();
        assert!(validate_receipt_semantics(&partial).is_err());
        // Real product timing values exercise serde_json's float representation rather than the
        // integer-like values used by the synthetic builder fixture.
        receipt.timings.load_ms = 14.162040999999993;
        let bundle = assemble_artifacts(receipt).expect("receipt bytes must assemble");
        let mut serialized_core: serde_json::Value =
            serde_json::from_slice(&bundle.receipt).expect("receipt must decode");
        let embedded_seal = serialized_core["receiptSha256"]
            .as_str()
            .expect("receipt seal")
            .to_owned();
        serialized_core
            .as_object_mut()
            .expect("receipt object")
            .remove("receiptSha256");
        assert_eq!(
            embedded_seal,
            seal_bytes(&canonical_semantic_seal_bytes(&serialized_core).unwrap()),
            "the producer seal must survive exact receipt serialization and parsing"
        );
        assert!(std::str::from_utf8(&bundle.human)
            .unwrap()
            .contains("\n- Receipt hash: "));
        assert!(std::str::from_utf8(&bundle.human)
            .unwrap()
            .contains("\n- Released cache ownership bytes: 4\n"));
        assert!(!std::str::from_utf8(&bundle.human).unwrap().contains("\\n"));
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

    #[test]
    fn pmset_thermal_text_is_recorded_and_only_the_speed_limit_is_read() {
        assert_eq!(pmset_cpu_speed_limit(PMSET_NOMINAL_THERMAL).unwrap(), None);
        // Unknown note lines (a new macOS release) are tolerated and kept verbatim.
        assert_eq!(
            pmset_cpu_speed_limit(
                "Note: No thermal warning level has been recorded\nNote: something new\n"
            )
            .unwrap(),
            None
        );
        assert_eq!(
            pmset_cpu_speed_limit(
                "CPU_Scheduler_Limit \t= 100\nCPU_Available_CPUs \t= 12\nCPU_Speed_Limit \t= 70\n"
            )
            .unwrap(),
            Some(70)
        );
        for invalid in [
            "CPU_Speed_Limit = fast\n",
            "CPU_Speed_Limit = 100\nCPU_Speed_Limit = 90\n",
        ] {
            assert!(pmset_cpu_speed_limit(invalid).is_err(), "{invalid:?}");
        }
        assert!(!host_state_throttled("fair", Some(100)));
        assert!(host_state_throttled("nominal", Some(99)));
        assert!(host_state_throttled("serious", None));
        assert!(host_state_throttled("critical", Some(100)));
    }

    const PMSET_NOMINAL_THERMAL: &str = "Note: No thermal warning level has been recorded\n\
         Note: No performance warning level has been recorded\n\
         Note: No CPU power status has been recorded\n";

    fn pmset_in_use(settings: &str) -> String {
        format!(
            "System-wide power settings:\nCurrently in use:\n standby              1\n{settings} womp                 1\n"
        )
    }

    /// The recorded power mode is the ACTIVE energy mode, not the per-source profile dump.
    #[test]
    fn power_mode_is_the_active_pmset_setting() {
        for (settings, expected) in [
            (" powermode            0\n", "automatic"),
            (" powermode            1\n", "low-power"),
            (" powermode            2\n", "high-power"),
            (" lowpowermode         0\n", "automatic"),
            (" lowpowermode         1\n", "low-power"),
            (" highpowermode        1\n lowpowermode 0\n", "high-power"),
        ] {
            assert_eq!(
                normalize_pmset_power_mode(&pmset_in_use(settings)).unwrap(),
                expected,
                "{settings:?}"
            );
        }
        for invalid in [
            pmset_in_use(""),
            pmset_in_use(" powermode            3\n"),
            pmset_in_use(" powermode            0\n lowpowermode 1\n"),
            pmset_in_use(" lowpowermode 1\n highpowermode 1\n"),
            pmset_in_use(" powermode 0\n powermode 2\n"),
            // `pmset -g custom` lists every power source's profile, not the active mode.
            "Battery Power:\n powermode            1\nAC Power:\n powermode            0\n".into(),
        ] {
            assert!(
                normalize_pmset_power_mode(&invalid).is_err(),
                "{invalid:?} was accepted"
            );
        }
    }

    /// A throttled row START refuses the row; every later state is recorded and flagged.
    #[test]
    fn only_a_throttled_row_start_refuses_the_row() {
        let pmset = pmset_in_use(" powermode            0\n");
        let state = |boundary: &str, therm: &str, process: i64| {
            host_state_from_probes(
                boundary,
                "2026-01-01T00:00:00Z".into(),
                &pmset,
                therm,
                process,
            )
        };
        let nominal = state("row-start", PMSET_NOMINAL_THERMAL, 0).unwrap();
        assert_eq!(
            (
                nominal.power_mode.as_str(),
                nominal.thermal_state.as_str(),
                nominal.throttled
            ),
            ("automatic", "nominal", false)
        );
        assert_eq!(nominal.pmset_thermal_raw, PMSET_NOMINAL_THERMAL);
        refuse_throttled_row_start(&nominal).unwrap();
        let fair = state("row-start", PMSET_NOMINAL_THERMAL, 1).unwrap();
        refuse_throttled_row_start(&fair).expect("fair is not a throttle signal");
        for (therm, process) in [
            (PMSET_NOMINAL_THERMAL, 2),
            (PMSET_NOMINAL_THERMAL, 3),
            ("CPU_Speed_Limit \t= 80\n", 0),
        ] {
            let throttled = state("row-start", therm, process).unwrap();
            assert!(throttled.throttled);
            let error = refuse_throttled_row_start(&throttled).unwrap_err();
            assert!(error.contains("refused before it runs"), "{error}");
            // At row end the same observation is recorded, not refused.
            let end = state("row-end", therm, process).unwrap();
            assert_eq!(host_state_changes(&nominal, [&end]), (true, false));
        }
        let low_power = host_state_from_probes(
            "row-end",
            "t".into(),
            &pmset_in_use(" powermode            1\n"),
            PMSET_NOMINAL_THERMAL,
            0,
        )
        .unwrap();
        assert_eq!(host_state_changes(&nominal, [&low_power]), (false, true));
        assert_eq!(host_state_changes(&nominal, [&nominal]), (false, false));
        for (therm, process, boundary) in [
            (PMSET_NOMINAL_THERMAL, 4, "row-start"),
            (PMSET_NOMINAL_THERMAL, 0, "mid-row"),
            ("CPU_Speed_Limit = x\n", 0, "row-start"),
        ] {
            assert!(
                state(boundary, therm, process).is_err(),
                "{therm:?}/{process}/{boundary} was accepted"
            );
        }
    }

    /// Decode throughput is the repeat's fixed-length steady decode, whatever the coordinate's
    /// own phase deltas are; a short, EOS-terminated generation is not a steady-decode sample.
    #[test]
    fn decode_throughput_is_the_fixed_length_steady_decode_not_a_phase_delta() {
        let observation = ProductObservations {
            snapshot: SnapshotInventory {
                root: PathBuf::new(),
                files: Vec::new(),
                bytes: 1,
                sha256: "b".repeat(64),
            },
            geometry: ProductGeometry {
                query_heads: 2,
                kv_heads: 1,
                head_dimension: 64,
                layers: 1,
                element_bytes: 2,
            },
            phases: Vec::new(),
            // A chunked coordinate: `first-token` -> `decode-steady` is one token, 0.02 ms apart.
            phase_elapsed_ms: vec![0.0, 1.0, 3.0, 4.0, 4.02, 6.0, 7.0, 8.0],
            allocations: Vec::new(),
            prefill_logits: Vec::new(),
            token_probabilities: Vec::new(),
            forced_token_probabilities: Vec::new(),
            session_id: "session".into(),
            cache_state_version: 1,
            operations: Vec::new(),
            load_elapsed_ms: 10.0,
            prefill_peak_window: ReceiptPeakWindow {
                started_at: "2026-01-01T00:00:00Z".into(),
                baseline_active_bytes: 1,
                reset_peak_bytes: 0,
            },
            cache_live_tokens: 32,
            cache_capacity_tokens: 256,
            packed_evidence: Vec::new(),
            dense_fallbacks: Vec::new(),
            coordinate_scope: None,
        };
        let timing =
            timing_from_product_observation(&observation, &test_steady_decode(1_000.0), "cold")
                .unwrap();
        // 255 timed tokens in 1000 ms; the phase deltas still supply prefill/TTFT/first token.
        let expected = RawTiming {
            load_ms: 10.0,
            prefill_ms: 2.0,
            ttft_ms: 1.0,
            first_token_ms: 4.0,
            decode_tokens_per_second: 255.0,
            steady_decode: test_steady_decode(1_000.0),
            host_state: None,
        };
        assert_eq!(timing, expected);
        let eos_terminated = SteadyDecodeMeasurement {
            generated_tokens: 24,
            timed_tokens: 23,
            ..test_steady_decode(100.0)
        };
        assert!(timing_from_product_observation(&observation, &eos_terminated, "cold").is_err());
    }

    /// A phase stamp is the observer's product clock: synchronous memory sampling (the
    /// `proc_pid_rusage` read) is subtracted, so it can never be charged to a phase delta.
    #[test]
    fn phase_clock_excludes_the_observers_own_memory_sampling() {
        let pid = std::process::id();
        let sample = |bytes| MemorySample {
            captured_at: timestamp_now(),
            pid,
            current_bytes: bytes,
            peak_bytes: bytes,
            mlx_active_bytes: bytes,
            mlx_cache_bytes: 0,
            mlx_peak_bytes: bytes,
        };
        let mut observer = ProductObserver::new();
        observer.load_boundary = Some((sample(1), sample(2)));
        Observer::phase(&mut observer, "process-start");
        // A sampling interval no test run can reach: charged to a phase, it would dominate.
        observer.sampling_elapsed_ms += 1.0e12;
        Observer::phase(&mut observer, "weights-loaded");
        let [start, loaded] = observer.phase_elapsed_ms[..] else {
            panic!("two phases were stamped");
        };
        // Clock-free in substance: the injected interval exceeds any wall time this test can see,
        // so only a stamp that failed to subtract it could order these the other way.
        assert!(loaded < start, "{start} -> {loaded}");
    }

    /// Greedy agreement is teacher-forced: one argmax flip is one mismatch, not a cascade. The
    /// free-running first divergence is recorded beside it.
    #[test]
    fn greedy_agreement_is_teacher_forced_and_divergence_is_recorded() {
        let reference = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        // The candidate's own choice at every reference position: one flip at position 1.
        let forced_choices = [5, 99, 7, 8, 9, 10, 11, 12, 13, 14];
        assert_eq!(
            teacher_forced_agreement(&forced_choices, &reference, &[]),
            (9, 10)
        );
        // A shorter forced run counts the unreached positions as mismatches.
        assert_eq!(
            teacher_forced_agreement(&forced_choices[..4], &reference, &[]),
            (3, 10)
        );
        // Free running, the same flip cascades: every later token differs.
        let free_running = [5, 99, 1, 2, 3, 4, 0, 0, 0, 0];
        assert_eq!(
            teacher_forced_agreement(&free_running, &reference, &[]),
            (1, 10)
        );
        // The reference stream keeps its stop token: at that position the candidate agrees when
        // it also stops (with any stop token), and disagrees when it keeps generating.
        let stopped = [5, 6, 2];
        assert_eq!(
            teacher_forced_agreement(&[5, 6, 3], &stopped, &[2, 3]),
            (3, 3)
        );
        assert_eq!(
            teacher_forced_agreement(&[5, 6, 7], &stopped, &[2, 3]),
            (2, 3)
        );
        assert_eq!(teacher_forced_agreement(&[5, 6], &stopped, &[2, 3]), (2, 3));
        assert_eq!(first_divergence(&free_running, &reference), Some(1));
        assert_eq!(first_divergence(&reference, &reference), None);
        assert_eq!(first_divergence(&reference[..3], &reference), Some(3));
        assert_eq!(first_divergence(&[], &[1]), Some(0));
    }

    /// The forced pass runs once per distinct reference stream; its compressed arm must stay on
    /// the fused reader.
    #[test]
    fn forced_passes_are_deduplicated_and_must_stay_compressed() {
        let streams = vec![vec![1, 2], vec![1, 2], vec![3], vec![1, 2], vec![3]];
        let mut passes = Vec::new();
        let forced = force_distinct_streams::<Vec<i32>, String>(&streams, |stream| {
            passes.push(stream.to_vec());
            Ok(stream.iter().map(|token| token + 10).collect())
        })
        .unwrap();
        assert_eq!(passes, vec![vec![1, 2], vec![3]]);
        assert_eq!(
            forced,
            vec![vec![11, 12], vec![11, 12], vec![13], vec![11, 12], vec![13]]
        );
        let fused = crate::primitives::PackedCacheEvidence {
            accepted_direct_calls: 3,
            ..Default::default()
        };
        forced_pass_stayed_compressed(std::slice::from_ref(&fused), &[]).unwrap();
        let edits: [fn(&mut crate::primitives::PackedCacheEvidence); 5] = [
            |e| e.accepted_direct_calls = 0,
            |e| e.dense_active = true,
            |e| e.full_cache_dequantizations = 1,
            |e| e.failed_dispatches = 1,
            |e| {
                e.fallback_reasons
                    .push(("decode".into(), "additive mask".into()))
            },
        ];
        for edit in edits {
            let mut evidence = fused.clone();
            edit(&mut evidence);
            assert!(forced_pass_stayed_compressed(&[evidence], &[]).is_err());
        }
        assert!(forced_pass_stayed_compressed(&[], &[]).is_err());
        assert!(forced_pass_stayed_compressed(
            std::slice::from_ref(&fused),
            &[("chunked-prefix-seed".into(), "dense store".into())]
        )
        .is_err());
    }

    /// The receipt's greedy agreement is the minimum over the repeats; each is recorded.
    #[test]
    fn greedy_agreement_is_the_one_quality_measurement() {
        let mut receipt = builder_test_receipt();
        receipt.quality.greedy_token_agreement_by_repeat = vec![0.9];
        receipt.quality.greedy_token_agreement = 0.9;
        validate_receipt_semantics(&receipt).expect("the one measurement recorded");
        let mut not_it = receipt.clone();
        not_it.quality.greedy_token_agreement = 1.0;
        assert!(validate_receipt_semantics(&not_it)
            .unwrap_err()
            .contains("one recorded quality measurement"));
        // Contract v5: five quality repeats are no longer a valid receipt shape.
        let mut five = receipt.clone();
        five.quality.greedy_token_agreement_by_repeat = vec![0.9; TIMING_REPEATS];
        assert!(validate_receipt_semantics(&five).is_err());
        let mut none = receipt.clone();
        none.quality.greedy_token_agreement_by_repeat.pop();
        assert!(validate_receipt_semantics(&none).is_err());
        // The receipt records the v5 statistics block verbatim: a v4 block (quality re-measured
        // every repeat) is refused.
        let mut v4 = receipt.clone();
        v4.quality.statistics.quality_measured_once = false;
        let error = validate_receipt_semantics(&v4).unwrap_err();
        assert!(error.contains("statistics"), "{error}");
    }

    // Helpers keep the estimate test's assertions free of clock-shaped names (they read fixture
    // arithmetic, not a clock).
    fn epoch(value: &str) -> Option<f64> {
        utc_timestamp_seconds(value)
    }

    /// A synthetic completed Llama-3.2-3B-shaped row: `wall` total, `prefill` per sequence.
    fn completed_row(
        band: &str,
        request_mode: &str,
        prefill_mode: &str,
        temperature: &str,
        target: u64,
        wall: f64,
        prefill: f64,
    ) -> RowDurationBasis {
        RowDurationBasis {
            coordinate: format!("llama-{band}-{request_mode}-{prefill_mode}-{temperature}"),
            family: "llama".into(),
            context_window_tokens: 131_072,
            context_target_tokens: target,
            wall_seconds: wall,
            prefill_seconds: prefill,
            work: row_work(temperature, request_mode, prefill_mode),
            // 1.8 GB of 4.5-bit weights over 2 x 28 layers x 24 heads x 128.
            attention_crossover_tokens: 1.8e9 * 8.0 / 4.5 / (2.0 * 28.0 * 24.0 * 128.0),
        }
    }

    fn predicted(band: &str, history: &[RowDurationBasis]) -> Option<(u64, u64, u64, String)> {
        estimate_row_seconds(
            "llama",
            band,
            row_work("warm", "single", "single-shot"),
            history,
        )
        .map(|e| {
            (
                e.seconds.round() as u64,
                e.fixed_seconds.round() as u64,
                e.prefill_seconds.round() as u64,
                e.basis_coordinate,
            )
        })
    }

    /// The estimate scales only the measured full-context prefills of the nearest completed
    /// same-family row; fixed costs are not scaled by context. The 32 -> 1024 -> 32768 -> 130560
    /// progression.
    #[test]
    fn row_duration_estimate_scales_only_the_prefill_part() {
        assert_eq!(epoch("1970-01-01T00:00:00Z"), Some(0.0));
        assert_eq!(epoch("2026-01-01T00:00:00Z"), Some(1_767_225_600.0));
        assert_eq!(epoch("2024-02-29T12:30:15.250Z"), Some(1_709_209_815.25));
        assert_eq!(epoch("not a time"), None);
        assert_eq!(
            row_work("cold", "single", "chunked"),
            RowWork {
                units: 7,
                prefills: 27
            }
        );
        assert_eq!(
            row_work("warm", "supported-batch", "single-shot"),
            RowWork {
                units: 9,
                prefills: 45
            }
        );
        assert_eq!(predicted("memory-material", &[]), None);
        // After only the 32-token row a chunked prefill is a suffix, so it provides no basis;
        // the harness logs and never refuses.
        let medium = completed_row(
            "medium",
            "supported-batch",
            "single-shot",
            "warm",
            1_024,
            600.0,
            0.2,
        );
        // Medium: 45 x 0.2 s of prefill, the other 591 s fixed. Memory-material (32768):
        // fixed 591 s (same 9 units) + 31 prefills x 0.2 s x the scale ratio.
        let (total, fixed, prefill, basis) =
            predicted("memory-material", std::slice::from_ref(&medium)).unwrap();
        assert_eq!(basis, "llama-medium-supported-batch-single-shot-warm");
        assert_eq!(fixed, 591);
        let ratio = prefill_scale(32_768.0, medium.attention_crossover_tokens)
            / prefill_scale(1_024.0, medium.attention_crossover_tokens);
        assert_eq!(prefill, (31.0 * 0.2 * ratio).round() as u64);
        assert_eq!(total, fixed + prefill);
        // The rejected whole-row-per-token estimate would have been 600 / 1024 x 32768 = 19200 s.
        assert!(total < 4_000, "{total}");
        // With the 32768 row measured (9 s prefill, 29 min wall), fit-boundary scales from it.
        let material = completed_row(
            "memory-material",
            "single",
            "single-shot",
            "warm",
            32_768,
            1_740.0,
            9.0,
        );
        let (total, fixed, prefill, basis) =
            predicted("fit-boundary", &[medium, material.clone()]).unwrap();
        assert_eq!(basis, "llama-memory-material-single-single-shot-warm");
        assert_eq!(fixed, 1_461);
        let target = context_band_target(131_072, "fit-boundary").unwrap() as f64;
        let ratio = prefill_scale(target, material.attention_crossover_tokens)
            / prefill_scale(32_768.0, material.attention_crossover_tokens);
        assert_eq!(prefill, (31.0 * 9.0 * ratio).round() as u64);
        assert_eq!(total, fixed + prefill);
        // A qwen row is never a basis for llama.
        let mut qwen = completed_row("medium", "single", "single-shot", "warm", 1_024, 1.0, 0.001);
        qwen.family = "qwen".into();
        assert_eq!(predicted("memory-material", &[qwen]), None);
        // Refusal needs the margin over the deadline.
        let estimate = RowDurationEstimate {
            seconds: 12_000.0,
            fixed_seconds: 0.0,
            prefill_seconds: 12_000.0,
            basis_coordinate: String::new(),
        };
        assert!(!duration_estimate_refuses(&estimate, 10_000));
        assert!(duration_estimate_refuses(&estimate, 9_000));
        // A chunked receipt provides no basis; a single-shot one does.
        let receipt = builder_test_receipt();
        assert_eq!(
            row_duration_basis(&receipt).map(|row| row.work),
            Some(row_work("cold", "single", "single-shot"))
        );
        let mut chunked = receipt.clone();
        chunked.matrix.prefill_mode = "chunked".into();
        assert!(row_duration_basis(&chunked).is_none());
    }

    /// Between sessions MLX may keep a residual within the slack; a larger one still fails.
    #[test]
    fn session_quiesce_allows_recorded_slack_but_not_a_leak() {
        let expected = 4_000_000_000;
        let slack = post_release_mlx_slack_bytes(expected);
        quiesced_within_slack(expected, 0, expected).unwrap();
        quiesced_within_slack(expected - 1, 0, expected).unwrap();
        quiesced_within_slack(expected + slack, 0, expected).unwrap();
        quiesced_within_slack(expected, POST_RELEASE_MLX_SLACK_FLOOR_BYTES, expected).unwrap();
        assert!(quiesced_within_slack(expected + slack + 1, 0, expected).is_err());
        assert!(
            quiesced_within_slack(expected, POST_RELEASE_MLX_SLACK_FLOOR_BYTES + 1, expected)
                .is_err()
        );
    }

    /// The dense-KV share of the prefill footprint and host-state variation are recorded flags.
    #[test]
    fn dense_kv_share_and_campaign_host_variation_are_recorded_not_refused() {
        assert_eq!(
            dense_kv_share(100, 1_000, "memory-material").unwrap(),
            (1_000, false)
        );
        assert_eq!(
            dense_kv_share(99, 1_000, "memory-material").unwrap(),
            (990, true)
        );
        assert_eq!(dense_kv_share(99, 1_000, "short").unwrap(), (990, false));
        assert!(dense_kv_share(1, 0, "short").is_err());
        let receipt = builder_test_receipt();
        let mut flagged = receipt.clone();
        flagged.memory.below_memory_material_share = !flagged.memory.below_memory_material_share;
        assert!(validate_receipt_semantics(&flagged)
            .unwrap_err()
            .contains("dense KV share"));
        let mut share = receipt.clone();
        share.memory.dense_kv_share_bps += 1;
        assert!(validate_receipt_semantics(&share).is_err());

        let mut free_running = receipt.clone();
        free_running.quality.greedy_agreement_method = "free-running".into();
        assert!(validate_receipt_semantics(&free_running)
            .unwrap_err()
            .contains("teacher-forced"));
        assert!(!campaign_host_state_varied([&receipt, &receipt]));
        let mut low_power = receipt.clone();
        low_power.provenance.power_mode = "low-power".into();
        assert!(campaign_host_state_varied([&receipt, &low_power]));
        let mut changed = receipt.clone();
        changed.provenance.thermal_changed_during_row = true;
        assert!(campaign_host_state_varied([&changed]));
        // Power/thermal no longer split the campaign's global identity.
        assert_eq!(
            campaign_global_identity(&receipt).unwrap(),
            campaign_global_identity(&low_power).unwrap()
        );
    }

    #[derive(Debug, PartialEq)]
    struct CompileCostView {
        first: f64,
        steady: f64,
        excess: f64,
        samples: Option<Vec<f64>>,
        band: Option<f64>,
        resolved: Option<bool>,
        cost: Option<f64>,
        reason: Option<String>,
    }

    fn compile_cost_view(attribution: &ReceiptCompileAttribution) -> CompileCostView {
        CompileCostView {
            first: attribution.first_dispatch_ms,
            steady: attribution.steady_dispatch_ms,
            excess: attribution.first_dispatch_excess_ms,
            samples: attribution.noise_samples_ms.clone(),
            band: attribution.noise_band_ms,
            resolved: attribution.compile_cost_resolved,
            cost: attribution.compile_cost_ms,
            reason: attribution.compile_cost_unresolved_reason.clone(),
        }
    }

    #[test]
    fn compile_attribution_is_raw_recomputable_and_fail_closed() {
        let cold_matrix = ReceiptMatrix {
            family: "llama".into(),
            context_band: "short".into(),
            request_mode: "single".into(),
            prefill_mode: "single-shot".into(),
            process_temperature: "cold".into(),
        };
        let probes = |values: &[f64]| {
            values
                .iter()
                .map(|value| (0.0, *value, "a".repeat(64)))
                .collect::<Vec<_>>()
        };
        let cold = compile_attribution_from_probes(
            "single-shot-generation",
            &cold_matrix,
            probes(&[20.0, 4.0, 5.0, 6.0, 7.0]),
            &[20.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        // Recorded fixture values, read through a helper so no clock-shaped name is asserted on.
        assert_eq!(cold.method, COMPILE_ATTRIBUTION_METHOD);
        assert_eq!(cold.source, "measured-repeats");
        assert_eq!(
            compile_cost_view(&cold),
            CompileCostView {
                first: 20.0,
                steady: 5.5,
                excess: 14.5,
                samples: Some(vec![4.0, 5.0, 6.0, 7.0]),
                band: Some(3.0),
                resolved: Some(true),
                cost: Some(14.5),
                reason: None,
            }
        );
        assert_eq!(cold_compile_alias(&cold), Some(14.5));

        let tamper = |edit: fn(&mut ReceiptCompileAttribution)| {
            let mut tampered = cold.clone();
            edit(&mut tampered);
            validate_compile_attribution(&tampered, &cold_matrix).is_err()
        };
        let edits: [fn(&mut ReceiptCompileAttribution); 11] = [
            |a| a.probe_durations_ms[2] = 20.0,
            |a| a.probe_evidence[2].dispatch_ms = 20.0,
            |a| a.source = "warmup-suites".into(),
            |a| a.operation = "chunked-prefix-reuse".into(),
            |a| a.noise_samples_ms = Some(vec![5.0, 5.0, 6.0, 6.0]),
            |a| a.noise_samples_ms = None,
            |a| a.noise_band_ms = Some(30.0),
            |a| a.compile_cost_resolved = Some(false),
            |a| a.compile_cost_ms = None,
            |a| a.compile_cost_ms = Some(1.0),
            |a| a.compile_cost_unresolved_reason = Some(COMPILE_COST_WITHIN_NOISE.into()),
        ];
        for (index, edit) in edits.into_iter().enumerate() {
            assert!(tamper(edit), "tamper {index} was accepted");
        }

        // Below the band and negative are recorded as unresolved, never refused.
        let within = compile_attribution_from_probes(
            "single-shot-generation",
            &cold_matrix,
            probes(&[8.0, 4.0, 5.0, 6.0, 7.0]),
            &[8.0, 4.0, 5.0, 6.0, 7.0],
        )
        .unwrap();
        let view = compile_cost_view(&within);
        assert_eq!(view.excess, 2.5);
        assert_eq!((view.resolved, view.cost), (Some(false), None));
        assert_eq!(view.reason.as_deref(), Some(COMPILE_COST_WITHIN_NOISE));
        assert_eq!(cold_compile_alias(&within), None);
        let flat = compile_attribution_from_probes(
            "single-shot-generation",
            &cold_matrix,
            probes(&[1.0, 1.0, 1.0, 1.0, 1.0]),
            &[1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(
            flat.compile_cost_unresolved_reason.as_deref(),
            Some(COMPILE_COST_NOT_SLOWER)
        );

        let warm_matrix = ReceiptMatrix {
            family: "llama".into(),
            context_band: "short".into(),
            request_mode: "single".into(),
            prefill_mode: "chunked".into(),
            process_temperature: "warm".into(),
        };
        let measured = [9.0, 9.5, 9.2, 9.1, 9.4];
        let warm = compile_attribution_from_probes(
            "chunked-prefix-reuse",
            &warm_matrix,
            probes(&[12.0, 9.0]),
            &measured,
        )
        .unwrap();
        assert_eq!(warm.source, "warmup-suites");
        let view = compile_cost_view(&warm);
        assert_eq!(view.excess, 3.0);
        assert_eq!(view.samples, Some(measured.to_vec()));
        assert_eq!(view.cost, Some(3.0));
        let warmup_seal =
            warmup_probe_suite_sha256(&"7".repeat(64), 42, &warm.probe_evidence).unwrap();
        let mut rebound_warm = warm.clone();
        rebound_warm.probe_evidence[1].setup_ms += 1.0;
        assert_ne!(
            warmup_seal,
            warmup_probe_suite_sha256(&"7".repeat(64), 42, &rebound_warm.probe_evidence).unwrap()
        );
        // W1 row 3 (llama memory-material 32k warm): first 9017 ms, steady 10585 ms.
        let row3 = compile_attribution_from_probes(
            "single-shot-generation",
            &ReceiptMatrix {
                prefill_mode: "single-shot".into(),
                ..warm_matrix.clone()
            },
            probes(&[9017.437917, 10585.460208]),
            &[10510.0, 10590.0, 10555.0, 10620.0, 10575.0],
        )
        .unwrap();
        let view = compile_cost_view(&row3);
        assert!(view.excess < 0.0);
        assert_eq!(view.resolved, Some(false));
        assert_eq!(view.reason.as_deref(), Some(COMPILE_COST_NOT_SLOWER));
        let mut short_band = warm.clone();
        short_band.noise_samples_ms = Some(vec![9.0; 4]);
        assert!(validate_compile_attribution(&short_band, &warm_matrix).is_err());

        // Missing or invalid timing data still fails closed.
        for (values, measured) in [
            (vec![f64::NAN, 9.0], measured.to_vec()),
            (vec![12.0, 0.0], measured.to_vec()),
            (vec![12.0, 9.0], vec![9.0, f64::NAN, 9.2, 9.1, 9.4]),
            (vec![12.0, 9.0], vec![9.0, 9.2]),
            (vec![], measured.to_vec()),
        ] {
            assert!(compile_attribution_from_probes(
                "chunked-prefix-reuse",
                &warm_matrix,
                probes(&values),
                &measured,
            )
            .is_err());
        }

        // The retired v1 method (positive excess, no band) is no longer accepted.
        let mut retired = cold.clone();
        retired.method = "first-dispatch-minus-steady-v1".into();
        assert!(validate_compile_attribution(&retired, &cold_matrix).is_err());
    }

    #[test]
    fn repository_and_pmetal_lock_identities_are_exact() {
        for remote in [
            "git@github.com:SceneWorks/inference.git",
            "ssh://git@github.com/SceneWorks/inference.git",
            "https://github.com/SceneWorks/inference.git",
        ] {
            assert_eq!(
                canonical_github_repository(remote).unwrap(),
                INFERENCE_REPOSITORY
            );
        }
        for remote in [
            "git@evil.example:SceneWorks/inference.git",
            "https://github.com/not-SceneWorks/inference.git",
            "https://github.com/SceneWorks/other/inference.git",
        ] {
            assert!(canonical_github_repository(remote)
                .map(|identity| identity != INFERENCE_REPOSITORY)
                .unwrap_or(true));
        }
        let identity = locked_mlx_identity(include_bytes!("../../../../Cargo.lock")).unwrap();
        assert_eq!(identity.version, "0.25.8");
        assert_eq!(
            identity.revision,
            "105a72fd7840dd81b3139728a90073d871572125"
        );
        assert_eq!(
            identity.source,
            format!("git+{PMETAL_MLX_REPOSITORY}?rev={0}#{0}", identity.revision)
        );
    }

    /// A fake parent row set over a real resume directory: a row is "accepted" once its marker
    /// exists, exactly as the production loop accepts a sealed row directory from resume.
    struct FakeRows {
        root: PathBuf,
        ran: Vec<usize>,
        /// Simulates the operator touching the stop file while this row's worker is running.
        touch_stop_during: Option<usize>,
        /// Simulates a pre-spawn duration refusal of this row.
        refuse: Option<usize>,
        stop_file: PathBuf,
    }

    impl FakeRows {
        fn new(root: &Path, stop_file: &Path) -> Self {
            Self {
                root: root.to_path_buf(),
                ran: Vec::new(),
                touch_stop_during: None,
                refuse: None,
                stop_file: stop_file.to_path_buf(),
            }
        }

        fn drive(&mut self, slugs: &[String]) -> Result<Option<OperatorStop>, String> {
            self.drive_all(slugs).map(|outcome| outcome.stop)
        }

        fn drive_all(&mut self, slugs: &[String]) -> Result<DriveOutcome, String> {
            let (root, stop_file) = (self.root.clone(), self.stop_file.clone());
            drive_rows(
                slugs,
                &operator_stop_files(
                    &["--stop-file".to_string(), stop_file.display().to_string()],
                    &root,
                )
                .unwrap(),
                &root.join("logs"),
                "sc-20671-operator-stop",
                |index, step| {
                    let marker = root.join(&slugs[index]);
                    match step {
                        RowStep::Resume => Ok(marker.exists()),
                        RowStep::Run => {
                            if self.refuse == Some(index) {
                                return Ok(false);
                            }
                            if self.touch_stop_during == Some(index) {
                                fs::write(&stop_file, b"").unwrap();
                            }
                            self.ran.push(index);
                            fs::write(&marker, b"accepted").unwrap();
                            Ok(true)
                        }
                    }
                },
            )
        }
    }

    /// A pre-spawn refusal never aborts the campaign: later rows still run, and the loop reports
    /// the refused rows.
    #[test]
    fn a_refused_row_does_not_stop_later_rows() {
        let dir = tempfile::tempdir().unwrap();
        let stop_file = dir.path().join(OPERATOR_STOP_FILE_NAME);
        let mut rows = FakeRows::new(dir.path(), &stop_file);
        rows.refuse = Some(1);
        let outcome = rows.drive_all(&stop_slugs()).unwrap();
        assert!(outcome.stop.is_none());
        assert_eq!(outcome.refused, vec!["row-b".to_string()]);
        assert_eq!(rows.ran, vec![0, 2, 3]);
        assert_eq!(
            CampaignOutcome::IncompleteWithRefusals(outcome.refused).exit_code(),
            1
        );
    }

    fn stop_slugs() -> Vec<String> {
        ["row-a", "row-b", "row-c", "row-d"]
            .iter()
            .map(|slug| (*slug).to_string())
            .collect()
    }

    fn stop_record(stop: &OperatorStop) -> serde_json::Value {
        serde_json::from_slice(&fs::read(&stop.record).unwrap()).unwrap()
    }

    #[test]
    fn operator_stop_before_row_zero_spawns_no_worker_and_records_status() {
        let dir = tempfile::tempdir().unwrap();
        let stop_file = dir.path().join(OPERATOR_STOP_FILE_NAME);
        fs::write(&stop_file, b"").unwrap();
        let mut rows = FakeRows::new(dir.path(), &stop_file);
        let stop = rows.drive(&stop_slugs()).unwrap().expect("operator stop");
        assert!(rows.ran.is_empty(), "no worker may start after a stop");
        assert_eq!(
            (stop.before_row, stop.row.as_str(), stop.rows_total),
            (0, "row-a", 4)
        );
        let record = stop_record(&stop);
        assert_eq!(record["status"], "stopped-by-operator");
        assert_eq!(record["kind"], "sc-20671-operator-stop");
        assert_eq!(record["beforeRow"], 0);
        assert_eq!(record["rowsAccepted"], 0);
        assert!(stop.record.starts_with(dir.path().join("logs")));
        // The record is sealed by its own sidecar.
        let raw = fs::read(&stop.record).unwrap();
        let name = stop
            .record
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        assert_eq!(
            fs::read_to_string(dir.path().join("logs").join(format!("{name}.sha256"))).unwrap(),
            format!("{}  {name}\n", seal_bytes(&raw))
        );
        assert_eq!(
            CampaignOutcome::StoppedByOperator(stop).exit_code(),
            OPERATOR_STOP_EXIT_CODE
        );
        assert_eq!(CampaignOutcome::Completed.exit_code(), 0);
        assert_ne!(OPERATOR_STOP_EXIT_CODE, 0);
        assert_ne!(OPERATOR_STOP_EXIT_CODE, 1);
        assert_ne!(OPERATOR_STOP_EXIT_CODE, 2);
    }

    #[test]
    fn operator_stop_between_rows_finishes_the_running_row_then_stops() {
        let dir = tempfile::tempdir().unwrap();
        let stop_file = dir.path().join(OPERATOR_STOP_FILE_NAME);
        let mut rows = FakeRows::new(dir.path(), &stop_file);
        rows.touch_stop_during = Some(1);
        let stop = rows.drive(&stop_slugs()).unwrap().expect("operator stop");
        // The row running when the operator asked is completed, never interrupted.
        assert_eq!(rows.ran, vec![0, 1]);
        assert!(dir.path().join("row-b").exists());
        assert!(!dir.path().join("row-c").exists());
        assert_eq!((stop.before_row, stop.row.as_str()), (2, "row-c"));
        let record = stop_record(&stop);
        assert_eq!(record["beforeRow"], 2);
        assert_eq!(record["beforeRowSlug"], "row-c");
        assert_eq!(record["rowsAccepted"], 2);
    }

    #[test]
    fn resume_after_operator_stop_continues_at_the_stopped_row_and_completes() {
        let dir = tempfile::tempdir().unwrap();
        let stop_file = dir.path().join("elsewhere-stop");
        let mut rows = FakeRows::new(dir.path(), &stop_file);
        rows.touch_stop_during = Some(1);
        let first = rows.drive(&stop_slugs()).unwrap().expect("operator stop");
        // Still armed: a rerun stops again before the same row, with a second, distinct record.
        let mut again = FakeRows::new(dir.path(), &stop_file);
        let second = again.drive(&stop_slugs()).unwrap().expect("still stopped");
        assert!(again.ran.is_empty());
        assert_eq!(second.before_row, first.before_row);
        assert_ne!(second.record, first.record);
        assert!(first.record.exists());
        fs::remove_file(&stop_file).unwrap();
        let mut resumed = FakeRows::new(dir.path(), &stop_file);
        assert_eq!(resumed.drive(&stop_slugs()).unwrap(), None);
        assert_eq!(
            resumed.ran,
            vec![2, 3],
            "accepted rows resume; the rest run once"
        );
        assert!(stop_slugs()
            .iter()
            .all(|slug| dir.path().join(slug).exists()));
    }

    #[test]
    fn resume_directory_tolerates_the_stop_file_without_changing_its_identity() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("resume");
        let slugs = stop_slugs();
        let identity = serde_json::json!({
            "schemaVersion": 1,
            "kind": "fixture",
            "inferenceRevision": "a".repeat(40),
            "sceneWorksRevision": "b".repeat(40),
        });
        let sealed = prepare_resume_root(&root, &mut identity.clone()).unwrap();
        fs::create_dir(root.join("logs")).unwrap();
        fs::write(root.join(OPERATOR_STOP_FILE_NAME), b"").unwrap();
        fs::write(root.join("custom.stop"), b"").unwrap();
        let stops = vec![root.join(OPERATOR_STOP_FILE_NAME), root.join("custom.stop")];
        validate_resume_entries(&root, &stops, &slugs).unwrap();
        assert_eq!(
            prepare_resume_root(&root, &mut identity.clone()).unwrap(),
            sealed,
            "stop-file presence must not change the resume identity"
        );
        fs::write(root.join("stray.json"), b"{}").unwrap();
        assert!(validate_resume_entries(&root, &stops, &slugs).is_err());
    }

    #[test]
    fn stop_file_flag_adds_to_the_resume_directory_stop() {
        let resume = Path::new("/abs/resume");
        assert_eq!(
            operator_stop_files(&[], resume).unwrap(),
            vec![resume.join(OPERATOR_STOP_FILE_NAME)]
        );
        let args = ["parent", "--stop-file", "/abs/halt"].map(String::from);
        assert_eq!(
            operator_stop_files(&args, resume).unwrap(),
            vec![
                resume.join(OPERATOR_STOP_FILE_NAME),
                PathBuf::from("/abs/halt")
            ]
        );
        assert!(operator_stop_files(&["--stop-file".to_string()], resume).is_err());
    }

    #[test]
    fn both_stop_paths_stop_and_stat_errors_are_not_absence() {
        let dir = tempfile::tempdir().unwrap();
        let custom = dir.path().join("elsewhere").join("halt");
        let files = operator_stop_files(
            &["--stop-file".to_string(), custom.display().to_string()],
            dir.path(),
        )
        .unwrap();
        let logs = dir.path().join("logs");
        assert_eq!(
            operator_stop_before_row(&files, &logs, "k", 0, "row-a", 2).unwrap(),
            None
        );
        // <resume-dir>/STOP still stops a parent given --stop-file.
        fs::write(dir.path().join(OPERATOR_STOP_FILE_NAME), b"").unwrap();
        let stop = operator_stop_before_row(&files, &logs, "k", 1, "row-b", 2)
            .unwrap()
            .expect("default stop honoured");
        assert_eq!(
            stop_record(&stop)["stopFiles"][0],
            files[0].display().to_string()
        );
        fs::remove_file(dir.path().join(OPERATOR_STOP_FILE_NAME)).unwrap();
        // ...and so does the custom path alone.
        fs::create_dir(dir.path().join("elsewhere")).unwrap();
        fs::write(&custom, b"").unwrap();
        let stop = operator_stop_before_row(&files, &logs, "k", 1, "row-b", 2)
            .unwrap()
            .expect("custom stop honoured");
        assert_eq!(
            stop_record(&stop)["stopFiles"][0],
            custom.display().to_string()
        );
        // A stat failure other than NotFound (here ENOTDIR) is an error, not "no stop".
        let blocked = vec![custom.join("STOP")];
        assert!(operator_stop_before_row(&blocked, &logs, "k", 0, "row-a", 2).is_err());
    }
}

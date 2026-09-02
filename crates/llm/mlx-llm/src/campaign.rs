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

use core_llm::{
    Message, Role, Sampling, StreamEvent, TextLlmOutput, TextLlmRequest, ThinkingMode, ToolSpec,
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
pub const QUALITY_CONTRACT_HASH: &str =
    "03c44b0f12caf79c1560e29fcfe536e2d7fd57153add4f3958697057b10116d2";
/// Frozen compressed-domain parity contract, shared by SC-20671 and the SC-20676 product proof.
pub const COMPRESSED_PARITY_MAX_ERROR: f64 = 0.0001;
pub const COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN: f64 = 0.999;

pub const CONTEXT_BANDS: [&str; 4] = ["short", "medium", "memory-material", "fit-boundary"];
pub const MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS: u64 = 1_000;
pub const FIT_BOUNDARY_MIN_CONTEXT_BPS: u64 = 9_000;
/// Fixed allowance for process-resident Metal/JIT runtime pages after MLX live tensors and its
/// allocator cache have returned exactly to the loaded-model boundary. This is deliberately not a
/// tensor or cache tolerance: both MLX release tolerances remain zero.
pub const POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES: u64 = 512 * 1024 * 1024;
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
    pub power_mode: String,
    pub thermal_state: String,
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
}

impl CampaignSession {
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
    pub mlx_active_tolerance_bytes: u64,
    pub mlx_cache_tolerance_bytes: u64,
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
    pub first_dispatch_excess_ms: f64,
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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
#[serde(deny_unknown_fields)]
pub struct ReceiptQualityStatistics {
    pub repeats: u64,
    pub warmups: u64,
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
    pub session_id: String,
    pub cache_state_version: u64,
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
        self.template.schema_version = 4;
        self.template.harness_version = "sc-20671-kv-baseline-v4".into();
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
            cold_compile_ms: self.compile_attribution.first_dispatch_excess_ms,
            warm_compile_ms: self.compile_attribution.steady_dispatch_ms,
            compile_attribution: self.compile_attribution,
            samples: self
                .timings
                .into_iter()
                .map(|t| ReceiptTimingSample {
                    load_ms: t.load_ms,
                    prefill_ms: t.prefill_ms,
                    ttft_ms: t.ttft_ms,
                    first_token_ms: t.first_token_ms,
                    decode_tokens_per_second: t.decode_tokens_per_second,
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
    if receipt.schema_version != 4
        || receipt.harness_version != "sc-20671-kv-baseline-v4"
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
        g.context_window_tokens,
        g.context_target_tokens,
        g.context_payload_tokens,
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
                    || p.source != "footprint -p"
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
    if receipt.matrix.context_band == "memory-material"
        && u128::from(dense).saturating_mul(10_000)
            < u128::from(prefill_footprint)
                .saturating_mul(u128::from(MEMORY_MATERIAL_MIN_DENSE_SHARE_BPS))
    {
        return Err("memory-material dense KV is below the frozen process-footprint share".into());
    }
    if receipt.matrix.context_band == "fit-boundary"
        && (receipt.geometry.capacity > receipt.geometry.context_window_tokens
            || u128::from(receipt.geometry.capacity).saturating_mul(10_000)
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
        && receipt.memory.persistent_kv_bytes.abs_diff(dense)
            > receipt.memory.reconciliation.tolerance_bytes
    {
        return Err(format!(
            "dense KV exceeds tolerance: observed={}, theoretical={}, tolerance={}",
            receipt.memory.persistent_kv_bytes,
            dense,
            receipt.memory.reconciliation.tolerance_bytes,
        ));
    }
    if receipt.memory.release.phys_footprint_tolerance_bytes
        != POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES
        || receipt.memory.release.mlx_active_tolerance_bytes != 0
        || receipt.memory.release.mlx_cache_tolerance_bytes != 0
    {
        return Err("release tolerances differ from the frozen platform contract".into());
    }
    let end = receipt.memory.phase_samples.last().unwrap();
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
    if receipt.timings.samples.len() != 5
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
    if (receipt.timings.cold_compile_ms
        - receipt.timings.compile_attribution.first_dispatch_excess_ms)
        .abs()
        > 1e-9
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
    if receipt.mode == "compressed"
        && (receipt.quality.parity_max_error > COMPRESSED_PARITY_MAX_ERROR
            || receipt.quality.perplexity_delta > 0.01)
    {
        return Err(format!(
            "quality thresholds failed: parityMaxError={}, perplexityDelta={}, greedyTokenAgreement={}, structuredToolAgreement={}, needleRetrieval={}, multiTurnPromptCache={}",
            receipt.quality.parity_max_error,
            receipt.quality.perplexity_delta,
            receipt.quality.greedy_token_agreement,
            receipt.quality.structured_tool_agreement,
            receipt.quality.needle_retrieval,
            receipt.quality.multi_turn_prompt_cache,
        ));
    }
    if receipt.quality.fixture_evidence.len() != 4
        || receipt.quality.statistics.repeats != 5
        || receipt.quality.statistics.warmups != 2
    {
        return Err("quality contract evidence incomplete".into());
    }
    if (receipt.mode == "compressed"
        && (receipt.quality.greedy_token_agreement < COMPRESSED_GREEDY_TOKEN_AGREEMENT_MIN
            || receipt.quality.structured_tool_agreement < 1.0
            || receipt.quality.needle_retrieval < 1.0
            || receipt.quality.multi_turn_prompt_cache < 1.0))
        || REQUIRED_FIXTURES.iter().any(|name| {
            receipt.quality.fixture_evidence.get(*name).is_none_or(|f| {
                !f.passed
                    || f.artifact_name != format!("fixtures/{name}.json")
                    || f.artifact_sha256.len() != 64
                    || !f
                        .artifact_sha256
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
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
    let human = format!(
        "# {} KV receipt\n\n- Run: {}\n- Mode: {}\n- Released cache ownership bytes: {}\n- Compile attribution: {} / {} / {}\n- Compile probes: {} ms\n- First dispatch excess: {} ms\n- Steady dispatch: {} ms\n- Receipt hash: {}\n",
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
        receipt.timings.cold_compile_ms,
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
    if bundle.fixtures.len() != REQUIRED_FIXTURES.len() * 5 {
        return Err("complete product evidence requires every fixture for every repeat".into());
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
    if timings["samples"].as_array().is_none_or(|v| v.len() != 5) {
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
    for repeat in 0..5 {
        for fixture in REQUIRED_FIXTURES {
            let artifact_name = fixture_artifact_name(fixture, repeat);
            let artifact = bundle
                .fixtures
                .iter()
                .find(|artifact| artifact.name == artifact_name)
                .ok_or_else(|| format!("missing repeat-bound fixture {artifact_name}"))?;
            validate_fixture_binding(&typed, fixture, &artifact.bytes, repeat)?;
        }
    }
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
    let reference_inventory = receipt.provenance.reference_model_sha256.as_str();
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
        "powerMode": provenance.power_mode,
        "thermalState": provenance.thermal_state,
        "commandTemplate": provenance.command_template,
    }))
    .map_err(|error| error.to_string())
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
    let mut global_identity = None;
    let mut family_identities = std::collections::BTreeMap::new();
    for item in prepared {
        validate_artifact_bundle(&item.bundle)?;
        let receipt: Receipt = serde_json::from_slice(&item.bundle.receipt)
            .map_err(|e| format!("prepared receipt is not decodable: {e}"))?;
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

/// The one SC-20671 coordinate SC-20676 may bind for a family. It is deliberately the frozen
/// memory-material long-context row, not the campaign's near-fit calibration row.
pub const SC20676_BASELINE_CONTEXT_BAND: &str = "memory-material";

/// A sealed candidate baseline selected from a complete SC-20671 64-coordinate publication.
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
/// selected row is trusted only after all 64 receipt/human/fixture bytes, sidecars, identities,
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
    if value
        .get("schemaVersion")
        .and_then(serde_json::Value::as_u64)
        != Some(1)
        || value.get("kind").and_then(serde_json::Value::as_str)
            != Some("sc-20671-complete-coordinate-set")
    {
        return Err("SC-20671 campaign manifest identity is invalid".into());
    }
    let rows = value
        .get("coordinates")
        .and_then(serde_json::Value::as_array)
        .ok_or("SC-20671 campaign manifest has no coordinate rows")?;
    let schedule = required_schedule();
    if rows.len() != schedule.len() {
        return Err("SC-20671 campaign manifest is not a complete 64-coordinate set".into());
    }
    let mut prepared = Vec::with_capacity(schedule.len());
    let mut seen = std::collections::BTreeSet::new();
    let mut global_identity = None;
    let mut family_identities = std::collections::BTreeMap::new();
    let mut outcomes = Vec::with_capacity(schedule.len());
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
        prepared.push(item);
    }
    if seen.len() != schedule.len() || family_identities.len() != 2 {
        return Err("SC-20671 campaign omits a scheduled coordinate or family identity".into());
    }
    validate_schedule_outcomes(&schedule, &outcomes)?;
    Ok(prepared)
}

/// Load and validate a whole published SC-20671 campaign before selecting the one exact long-
/// context candidate row for `family`.  This never accepts a standalone receipt: all 64 immutable
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
    let mut fixtures = Vec::with_capacity(REQUIRED_FIXTURES.len() * 5);
    for repeat in 0..5 {
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
        &launch.llama_fp32_reference_snapshot,
        &launch.qwen_fp32_reference_snapshot,
        &launch.prompt_file,
    ] {
        if !path.exists() {
            return Err(format!(
                "required campaign input is absent: {}",
                path.display()
            ));
        }
    }
    // Validate every role before a single worker is spawned.  The worker repeats the family-local
    // check before loading, so neither command boundary can smuggle in an arbitrary snapshot.
    validate_benchmark_snapshot(&launch.llama_snapshot, &LLAMA_CANDIDATE)?;
    validate_benchmark_snapshot(&launch.qwen_snapshot, &QWEN_CANDIDATE)?;
    validate_benchmark_snapshot(&launch.llama_fp32_reference_snapshot, &LLAMA_REFERENCE)?;
    validate_benchmark_snapshot(&launch.qwen_fp32_reference_snapshot, &QWEN_REFERENCE)?;
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
            let reference_snapshot = if row.coordinate.family == "llama" {
                &launch.llama_fp32_reference_snapshot
            } else {
                &launch.qwen_fp32_reference_snapshot
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
                .arg(reference_snapshot)
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

/// CLI entrypoint used by the standalone `sc20671-kv-baseline` binary.  The child accepts no
/// caller-authored evidence: identity, geometry, timing, quality, cache, and fixtures are bound
/// from the loaded product session before receipt publication.
pub fn sc20671_cli(args: &[String]) -> Result<(), String> {
    let Some(mode) = args.first().map(String::as_str) else {
        return Err("usage: sc20671-kv-baseline parent|worker [options]".into());
    };
    match mode {
        "parent" => launch_complete_campaign(&CampaignLaunch {
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
            validate_coordinate_model_contract(&row.coordinate, &snapshot, &reference_snapshot)?;
            let candidate_session = CampaignSession::load(&snapshot)
                .map_err(|e| format!("load candidate campaign session: {e}"))?;
            let candidate_baseline = candidate_session.load_start_sample.mlx_active_bytes;
            // Candidate lifecycle evidence is collected before the reference model exists in the
            // process. This keeps every global MLX phase sample attributable to the candidate.
            let mut candidate_warmups = Vec::new();
            if row.coordinate.process_temperature == "warm" {
                for warmup_index in 0..2 {
                    let half = run_product_fixture_half_on_session(
                        &candidate_session,
                        &prompt,
                        &row.coordinate,
                    )
                    .map_err(|e| {
                        format!(
                            "product warmup {warmup_index} for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        )
                    })?;
                    candidate_warmups.push(half);
                }
            }
            let warmup_cache_state_version =
                (!candidate_warmups.is_empty()).then(|| candidate_session.cache_state_version());
            let mut candidate_repeats = Vec::with_capacity(5);
            for repeat in 0..5 {
                let half = run_product_fixture_half_on_session(
                    &candidate_session,
                    &prompt,
                    &row.coordinate,
                )
                .map_err(|e| {
                    format!(
                        "product fixture repeat {repeat} for {}: {e}",
                        coordinate_slug(&row.coordinate)
                    )
                })?;
                candidate_repeats.push(half);
            }
            drop(candidate_session);
            quiesce_campaign_active_memory(candidate_baseline).map_err(|e| {
                format!(
                    "candidate session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let reference_session = CampaignSession::load(&reference_snapshot)
                .map_err(|e| format!("load reference campaign session: {e}"))?;
            let reference_baseline = reference_session.load_start_sample.mlx_active_bytes;
            let mut reference_warmups = Vec::with_capacity(candidate_warmups.len());
            for warmup_index in 0..candidate_warmups.len() {
                reference_warmups.push(
                    run_product_fixture_half_on_session(
                        &reference_session,
                        &prompt,
                        &row.coordinate,
                    )
                    .map_err(|e| {
                        format!(
                            "reference warmup {warmup_index} for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        )
                    })?,
                );
            }
            let mut reference_repeats = Vec::with_capacity(5);
            for repeat in 0..5 {
                reference_repeats.push(
                    run_product_fixture_half_on_session(
                        &reference_session,
                        &prompt,
                        &row.coordinate,
                    )
                    .map_err(|e| {
                        format!(
                            "reference fixture repeat {repeat} for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        )
                    })?,
                );
            }
            drop(reference_session);
            quiesce_campaign_active_memory(reference_baseline).map_err(|e| {
                format!(
                    "reference session release for {}: {e}",
                    coordinate_slug(&row.coordinate)
                )
            })?;

            let warmup_suites = candidate_warmups
                .into_iter()
                .zip(reference_warmups)
                .enumerate()
                .map(|(warmup_index, (candidate, reference))| {
                    let suite = pair_product_fixture_halves(candidate, reference)?;
                    suite.quality().map_err(|e| {
                        core_llm::Error::InvalidRequest(format!(
                            "product warmup {warmup_index} quality for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        ))
                    })?;
                    Ok(suite)
                })
                .collect::<core_llm::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            let suites = candidate_repeats
                .into_iter()
                .zip(reference_repeats)
                .enumerate()
                .map(|(repeat, (candidate, reference))| {
                    let suite = pair_product_fixture_halves(candidate, reference)?;
                    suite.quality().map_err(|e| {
                        core_llm::Error::InvalidRequest(format!(
                            "product fixture repeat {repeat} quality for {}: {e}",
                            coordinate_slug(&row.coordinate)
                        ))
                    })?;
                    Ok(suite)
                })
                .collect::<core_llm::Result<Vec<_>>>()
                .map_err(|e| e.to_string())?;
            let kernel_runs = suites
                .iter()
                .map(|suite| &suite.kernel_candidate)
                .collect::<Vec<_>>();
            let warmup_runs = warmup_suites
                .iter()
                .map(|suite| &suite.kernel_candidate)
                .collect::<Vec<_>>();
            let timings = timing_samples_from_product_repeats(
                &kernel_runs,
                &row.coordinate,
                row.coordinate.process_temperature,
                &warmup_runs,
            )?;
            let executable = std::env::current_exe().map_err(|e| e.to_string())?;
            let fixtures = sealed_product_fixture_artifacts(&suites, &row.coordinate)?;
            let receipt = product_receipt(
                &row.coordinate,
                &suites[0],
                timings,
                &executable,
                &fixtures,
                warmup_cache_state_version,
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
) -> Result<(), String> {
    let candidate = benchmark_model(coordinate.family, false)?;
    let reference = benchmark_model(coordinate.family, true)?;
    if snapshot == reference_snapshot {
        return Err("candidate and higher-precision reference paths must be distinct".into());
    }
    validate_benchmark_snapshot(snapshot, candidate)?;
    validate_benchmark_snapshot(reference_snapshot, reference)?;
    Ok(())
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
            }),
            &suite.kernel_candidate,
            &suite.kernel_reference,
        ),
        "structured-tool-call" => (
            serde_json::json!({ "matches": quality.tool_matches, "total": quality.tool_total }),
            &suite.tool_candidate,
            &suite.tool_reference,
        ),
        "long-context-needle" => (
            serde_json::json!({ "matches": quality.needle_matches, "total": quality.needle_total }),
            &suite.needle_candidate,
            &suite.needle_reference,
        ),
        "multi-turn-prompt-cache" => (
            serde_json::json!({ "matches": quality.cache_matches, "total": quality.cache_total }),
            &suite.cache_candidate,
            &suite.cache_reference,
        ),
        _ => return Err(format!("unknown fixture {name}")),
    };
    let metrics = compute_quality(&quality)?;
    let value = serde_json::json!({
        "fixture": name,
        "independentReference": fixture_independent_reference(name, &reference.quality_observation.snapshot.sha256),
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

fn fixture_independent_reference(name: &str, reference_inventory: &str) -> String {
    if name == "kernel-fp32-reference" {
        format!("host-fp32-dense-attention-v1;bf16-model:{reference_inventory}")
    } else {
        format!("bf16-model:{reference_inventory}")
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
    if suites.len() != 5 {
        return Err("complete product evidence requires exactly five repeats".into());
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
pub trait Observer {
    fn phase(&mut self, name: &'static str);
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
    /// sequence capacity, and the actual MLX array scalar width so repeated append events are not
    /// summed and the receipt cannot confuse model-weight dtype with cache dtype.
    fn cache_snapshot(&mut self, bytes: u64, _tokens: u64, _element_bytes: u64) {
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
    session_id: Option<String>,
    cache_state_version: Option<u64>,
    operations: Vec<String>,
    load_elapsed_ms: Option<f64>,
    load_boundary: Option<(MemorySample, MemorySample)>,
    prefill_peak_window: Option<ReceiptPeakWindow>,
    cache_capacity_tokens: u64,
    sampling_elapsed_ms: f64,
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
            cache_capacity_tokens: 0,
            sampling_elapsed_ms: 0.0,
        }
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
        let prefill_peak_window = self
            .prefill_peak_window
            .ok_or("product observer did not reset the MLX prefill peak window")?;
        if self.cache_capacity_tokens == 0 {
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
            cache_capacity_tokens: self.cache_capacity_tokens,
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
    pub session_id: String,
    pub cache_state_version: u64,
    pub operations: Vec<String>,
    pub load_elapsed_ms: f64,
    pub prefill_peak_window: ReceiptPeakWindow,
    pub cache_capacity_tokens: u64,
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
        match sample_memory(self.pid) {
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

    fn cache_snapshot(&mut self, bytes: u64, tokens: u64, element_bytes: u64) {
        if bytes == 0 || tokens == 0 || element_bytes == 0 {
            self.error = Some(
                "product cache snapshot must have positive bytes, tokens, and element width".into(),
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
        self.cache_capacity_tokens = self.cache_capacity_tokens.max(tokens);
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

fn quiesce_campaign_active_memory(expected_active_bytes: u64) -> core_llm::Result<()> {
    mlx_rs::memory::clear_cache();
    let active = mlx_rs::memory::get_active_memory() as u64;
    let cache = mlx_rs::memory::get_cache_memory() as u64;
    if active != expected_active_bytes || cache != 0 {
        return Err(core_llm::Error::Load(format!(
            "campaign session did not quiesce before the next model load: active={active}, expected={expected_active_bytes}, cache={cache}"
        )));
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
                observer.operation("supported-batch");
            }
            if coordinate.prefill_mode == "chunked" {
                provider.campaign_prefix_reuse(prefix_prompt)?;
                observer.operation("chunked-prefix-reuse");
            }
        }
        observer.begin_prefill_memory_window();
        observer.phase("weights-loaded");
        observer.allocation("weights", "persistent", session.model_weights_bytes());
        let mut saw_token = false;
        let output = provider.generate_observed(
            &request,
            &mut |event| saw_token |= matches!(event, StreamEvent::Token { .. }),
            observer,
        )?;
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
        provider.campaign_cancel_after_first_token(observer)?;
        output
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

pub struct CoordinateOperationEvidence {
    pub observation: ProductObservations,
    pub operation: String,
    pub generated_tokens: u64,
    pub prompt_tokens: u64,
    pub output_sha256: String,
    pub compile_setup_ms: f64,
    pub compile_dispatch_ms: f64,
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
    let mut observer = ProductObserver::new();
    let output = run_dense_lifecycle_request_on_session(
        session,
        prefix_prompt,
        request,
        None,
        &mut observer,
    )?;
    let quality_observation = observer.finish().map_err(core_llm::Error::InvalidRequest)?;
    Ok(ProductFixtureResult {
        observation: primary.observation,
        quality_observation,
        output,
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
/// real phase boundaries; this wrapper prevents `/usr/bin/footprint` latency from being relabeled
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
        provider.campaign_seed_prefix_reuse(prefix_prompt)?
    } else {
        0.0
    };
    observer.begin_prefill_memory_window();
    observer.phase("weights-loaded");
    observer.allocation("weights", "persistent", session.model_weights_bytes());

    let (operation, generated_tokens, prompt_tokens, output_sha256, observed_dispatch_ms) =
        if operation == "supported-batch" {
            let ((outputs, prompt_tokens), dispatch_elapsed_ms) =
                measure_product_dispatch(&mut observer, |observer| {
                    provider.campaign_supported_batch_observed(prefix_prompt, 2, observer)
                })?;
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
            let ((output, hits, prompt_tokens), dispatch_elapsed_ms) =
                measure_product_dispatch(&mut observer, |observer| {
                    provider.campaign_prefix_reuse_observed(prefix_prompt, observer)
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
    // Prompt-cache reuse and deliberate cancellation are additional lifecycle facts.  They do not
    // replace the coordinate operation measured above.
    provider.campaign_prefix_reuse(prefix_prompt)?;
    observer.cache_state(session.advance_cache_state());
    observer.phase("prompt-cache-reuse");
    provider.campaign_cancel_after_first_token(&mut observer)?;
    provider.campaign_release_cache_state();
    mlx_rs::memory::clear_cache();
    observer.phase("post-run-release");
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
    let actual = output.as_slice::<f32>();
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
/// [`ReceiptBuilder`].  This helper intentionally rejects a fixture that did not execute its own
/// required behavior (tool parsing, needle recovery, or prefix result equality), rather than
/// converting a missing capability into a green ratio.
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
) -> Result<QualityObservation, String> {
    let parity_errors = dense_kernel_fp32_parity_errors()?;
    let (greedy_matches, greedy_total) = token_agreement(
        &kernel_candidate.quality_observation.token_probabilities,
        &kernel_reference.quality_observation.token_probabilities,
    );
    let (cache_matches, cache_total) = token_agreement(
        &cache_candidate.quality_observation.token_probabilities,
        &cache_reference.quality_observation.token_probabilities,
    );
    let tool_ok = tool_candidate.output.tool_calls == tool_reference.output.tool_calls
        && matches!(tool_candidate.output.tool_calls.as_slice(), [call]
            if call.name == "record_baseline_fact"
                && call.arguments.get("fact").and_then(serde_json::Value::as_str)
                    == Some("SC20671 structured fixture"));
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
            &kernel_reference.quality_observation.token_probabilities,
        )?,
        candidate_perplexity: negative_log_likelihood(
            &kernel_candidate.quality_observation.token_probabilities,
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
    pub context_window_tokens: u64,
    pub context_target_tokens: u64,
    pub context_payload_tokens: u64,
}

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
        max_new_tokens: 64,
        seed: Some(0),
        ..Default::default()
    }
}

fn needle_fixture_prompt(prompt: &str, band_payload: &str, needle: &str) -> String {
    format!(
        "{prompt}\nRemember this harmless passphrase exactly: {needle}\n\
         BEGIN LONG CONTEXT\n{band_payload}\nEND LONG CONTEXT\n\
         What harmless passphrase were you instructed to remember? Reply with only the passphrase."
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
    pair_product_fixture_halves(candidate_half, reference_half)
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
}

fn run_product_fixture_half_on_session(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureHalf> {
    session.validate_coordinate_family(coordinate)?;
    let context_window_tokens = session.provider.campaign_context_window()?;
    let (band_payload, context_target_tokens, context_payload_tokens) = session
        .provider
        .campaign_context_band_measurement(coordinate.context_band)?;
    let needle = "SC20671-NUMERIC-NEEDLE-9b7a2e".to_string();
    let kernel_prompt = format!("{prompt}\n{band_payload}\nReturn a concise deterministic answer.");
    let tool_prompt = format!(
        "{prompt}\n{band_payload}\nCall record_baseline_fact with fact exactly `SC20671 structured fixture`."
    );
    let needle_prompt = needle_fixture_prompt(prompt, &band_payload, &needle);
    let cache_prompt = format!("{prompt}\n{band_payload}\nRepeat the stable baseline fact.");
    let fixture_prompt_tokens = [
        session.provider.campaign_prompt_tokens(&kernel_prompt)?,
        session.provider.campaign_prompt_tokens(&tool_prompt)?,
        session.provider.campaign_prompt_tokens(&needle_prompt)?,
        session.provider.campaign_prompt_tokens(&cache_prompt)?,
    ];
    for tokens in fixture_prompt_tokens {
        if tokens > context_window_tokens.saturating_sub(256) {
            return Err(core_llm::Error::InvalidRequest(format!(
                "fixture prompt tokenization exceeds the loaded context: {tokens}/{context_window_tokens}"
            )));
        }
    }
    let kernel = run_product_fixture_on_session(
        session,
        &kernel_prompt,
        fixture_request(kernel_prompt.clone(), Vec::new()),
        coordinate,
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
    let cache = run_product_fixture_on_session(
        session,
        &cache_prompt,
        fixture_request(cache_prompt.clone(), Vec::new()),
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
    })
}

fn pair_product_fixture_halves(
    candidate: ProductFixtureHalf,
    reference: ProductFixtureHalf,
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

/// Derive one timing sample from product phase boundaries plus the wall-clock snapshot/provider
/// load that created this session. Compile attribution is finalized across the cold dispatch or
/// two real warmups by [`timing_samples_from_product_repeats`].
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
    let load_ms = observation.load_elapsed_ms;
    if !load_ms.is_finite() || load_ms <= 0.0 {
        return Err("product snapshot load duration is not positive".into());
    }
    let prefill_ms = positive_delta(2, 1)?;
    let ttft_ms = positive_delta(3, 2)?;
    let first_token_ms = positive_delta(3, 0)?;
    let decode_ms = positive_delta(4, 3)?;
    let throughput = generated_tokens as f64 * 1_000.0 / decode_ms;
    if !throughput.is_finite() || throughput <= 0.0 {
        return Err("invalid product decode throughput".into());
    }
    if !matches!(process_temperature, "cold" | "warm") {
        return Err("unknown process temperature".into());
    }
    Ok(RawTiming {
        load_ms,
        prefill_ms,
        ttft_ms,
        first_token_ms,
        decode_tokens_per_second: throughput,
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
    if attribution.method != "first-dispatch-minus-steady-v1"
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
    if excess <= 0.0
        || !excess.is_finite()
        || (attribution.first_dispatch_ms - first).abs() > 1e-9
        || (attribution.steady_dispatch_ms - steady).abs() > 1e-9
        || (attribution.first_dispatch_excess_ms - excess).abs() > 1e-9
    {
        return Err(format!(
            "compile attribution does not prove a positive first-dispatch excess: first={first:.6}ms steady={steady:.6}ms excess={excess:.6}ms"
        ));
    }
    Ok(())
}

fn compile_attribution_from_probes(
    operation: &str,
    matrix: &ReceiptMatrix,
    probes: Vec<(f64, f64, String)>,
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
        method: "first-dispatch-minus-steady-v1".into(),
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
    attribution.first_dispatch_excess_ms =
        attribution.first_dispatch_ms - attribution.steady_dispatch_ms;
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
/// warm worker uses its two real pre-measurement warmups. Prefix-reuse probes add only the seed and
/// observed-hit product calls, excluding allocator reset and footprint instrumentation.
pub fn timing_samples_from_product_repeats(
    runs: &[&ProductFixtureResult],
    coordinate: &Coordinate,
    process_temperature: &str,
    warmups: &[&ProductFixtureResult],
) -> Result<ProductTimingMeasurements, String> {
    if runs.len() != 5 {
        return Err("receipt requires exactly five product repeats".into());
    }
    let samples = runs
        .iter()
        .map(|run| {
            timing_from_product_observation(
                &run.observation,
                run.coordinate_generated_tokens as usize,
                process_temperature,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let operation = runs[0].coordinate_operation.as_str();
    if runs.iter().any(|run| run.coordinate_operation != operation)
        || warmups
            .iter()
            .any(|run| run.coordinate_operation != operation)
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
                run.compile_setup_ms,
                run.compile_dispatch_ms,
                primary_operation_evidence_digest(run),
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
    let compile_attribution = compile_attribution_from_probes(operation, &matrix, probes)?;
    Ok(ProductTimingMeasurements {
        samples,
        compile_attribution,
    })
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

/// `pmset -g therm` is verbose diagnostic text, never a receipt state.  Accept only an explicit
/// no-throttling/no-pressure observation and normalize it to the schema's semantic `nominal`.
pub fn normalize_pmset_thermal(value: &str) -> Result<String, String> {
    let normalized = value.to_ascii_lowercase();
    if normalized.contains("not nominal")
        || normalized.contains("throttl")
        || normalized.contains("critical")
    {
        return Err("pmset thermal probe reports throttling or a contradictory state".into());
    }
    let lines = normalized
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    let nominal_no_history = [
        "note: no thermal warning level has been recorded",
        "note: no performance warning level has been recorded",
        "note: no cpu power status has been recorded",
    ];
    if lines.len() == nominal_no_history.len()
        && nominal_no_history
            .iter()
            .all(|expected| lines.contains(expected))
    {
        return Ok("nominal".into());
    }
    let mut saw_zero = false;
    for line in lines {
        let Some((name, raw_value)) = line.split_once(':') else {
            continue;
        };
        if !matches!(name.trim(), "thermal pressure" | "thermal level") {
            continue;
        }
        let value = raw_value.trim();
        let digits = value
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect::<String>();
        if digits.is_empty() || digits != "0" {
            return Err("pmset thermal probe reports nonzero thermal pressure".into());
        }
        saw_zero = true;
    }
    if saw_zero {
        return Ok("nominal".into());
    }
    Err("pmset thermal probe did not prove nominal thermal state".into())
}

fn product_receipt(
    coordinate: &Coordinate,
    suite: &ProductFixtureSuite,
    timings: ProductTimingMeasurements,
    executable: &Path,
    fixtures: &[SealedFixtureArtifact],
    warmup_cache_state_version: Option<u64>,
) -> Result<Receipt, String> {
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
    let quality = suite.quality()?;
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
    let reference = &suite.kernel_reference.observation.snapshot;
    let model = &observation.snapshot;
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
    let scene_works_revision = checked_git_revision(&scene_works_root)?;
    let inference_revision = checked_git_revision(inference_root)?;
    let hardware = probed_command("sysctl", &["-n", "hw.model"], "hardware")?;
    let xcode = probed_command("xcodebuild", &["-version"], "xcode")?;
    let power_mode = probed_command("pmset", &["-g", "custom"], "power mode")?;
    let thermal_state = probed_command("pmset", &["-g", "therm"], "thermal state")?;
    let normalized_thermal_state = normalize_pmset_thermal(&thermal_state)?;
    let mlx = locked_mlx_identity(include_bytes!("../../../../Cargo.lock"))?;
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
                independent_reference: fixture_independent_reference(name, &reference.sha256),
            },
        );
    }
    let template = Receipt {
        schema_version: 4, harness_version: "sc-20671-kv-baseline-v4".into(), run_id: seal_bytes(format!("{}:{}:{}", coordinate_slug(coordinate), model.sha256, seal_bytes(transcript.as_bytes())).as_bytes()), captured_at: release.timestamp.clone(), mode: "dense".into(), status: "complete".into(), contract_hash: QUALITY_CONTRACT_HASH.into(), receipt_sha256: String::new(),
        provenance: ReceiptProvenance { scene_works_repository, inference_repository, scene_works_revision, inference_revision, mlx_version: mlx.version, mlx_source: mlx.source, mlx_revision: mlx.revision, dependency_lock_sha256: seal_bytes(include_bytes!("../../../../Cargo.lock")), os: std::env::consts::OS.into(), xcode, hardware, model_id: format!("{}@{};architecture={};inventory={}", candidate_contract.repository, candidate_contract.revision, candidate_contract.architecture, model.sha256), model_file_sha256: model.sha256.clone(), model_file_bytes: model.bytes, reference_model_id: format!("{}@{};architecture={};inventory={}", reference_contract.repository, reference_contract.revision, reference_contract.architecture, reference.sha256), reference_model_sha256: reference.sha256.clone(), reference_model_bytes: reference.bytes, power_mode, thermal_state: normalized_thermal_state, command_template: "sc20671-kv-baseline --mode {mode}".into(), command: "sc20671-kv-baseline --mode dense".into(), campaign_session_id: observation.session_id.clone(), campaign_cache_state_version: observation.cache_state_version, coordinate_operation_sha256: coordinate_operation_digest(&suite.kernel_candidate) },
        matrix: ReceiptMatrix { family: coordinate.family.into(), context_band: coordinate.context_band.into(), request_mode: coordinate.request_mode.into(), prefill_mode: coordinate.prefill_mode.into(), process_temperature: coordinate.process_temperature.into() },
        geometry: ReceiptGeometry { batch: if coordinate.request_mode == "single" {1} else {2}, query_heads: observation.geometry.query_heads, kv_heads: observation.geometry.kv_heads, head_dimension: observation.geometry.head_dimension, query_length: suite.kernel_candidate.coordinate_prompt_tokens, kv_length: observation.cache_capacity_tokens, layers: observation.geometry.layers, element_bytes: observation.geometry.element_bytes, capacity: observation.cache_capacity_tokens, context_window_tokens: suite.context_window_tokens, context_target_tokens: suite.context_target_tokens, context_payload_tokens: suite.context_payload_tokens },
        memory: ReceiptMemory { model_weights_bytes, persistent_kv_bytes: cache_bytes, transient_workspace_bytes: workspace, dense_theoretical_kv_bytes: 0, prefill_peak_window: observation.prefill_peak_window.clone(), phase_samples: vec![], allocation_events: vec![], reconciliation: ReceiptReconciliation { expected_dense_kv_bytes: 0, observed_persistent_kv_bytes: 0, tolerance_bytes: 0 }, release: ReceiptRelease { verified: release.phys_footprint_bytes <= weights_loaded.phys_footprint_bytes.saturating_add(POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES) && release.mlx.active_bytes <= weights_loaded.mlx.active_bytes && release.mlx.cache_bytes <= weights_loaded.mlx.cache_bytes, phys_footprint_tolerance_bytes: POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES, mlx_active_tolerance_bytes: 0, mlx_cache_tolerance_bytes: 0 } },
        timings: ReceiptTimings { load_ms: 0.0,prefill_ms:0.0,ttft_ms:0.0,first_token_ms:0.0,decode_tokens_per_second:0.0,cold_compile_ms:0.0,warm_compile_ms:0.0,compile_attribution:compile_attribution.clone(),samples:vec![],summary:ReceiptTimingSummary{decode_tokens_per_second_mean:0.0,decode_tokens_per_second_p95:0.0,decode_tokens_per_second_variance:0.0,decode_tokens_per_second_coefficient_of_variation:0.0,confidence_interval_low:0.0,confidence_interval_high:0.0}},
        quality: ReceiptQuality { parity_max_error:0.0,perplexity_delta:0.0,greedy_token_agreement:0.0,structured_tool_agreement:0.0,needle_retrieval:0.0,multi_turn_prompt_cache:0.0,statistics:ReceiptQualityStatistics{repeats:5,warmups:2,confidence_interval:"95% bootstrap".into(),outlier_policy:"report all samples; no silent deletion".into(),variance_policy:"all raw repeats retained; decode throughput coefficient of variation must stay within the frozen maximum".into(),max_coefficient_of_variation:0.05},fixture_evidence}, lifecycle: ReceiptLifecycle { append:true,chunked_prefill:true,single_shot_prefill:true,prompt_cache_reuse:true,trim:false,rollback:false,clear:false,cancel:true,clone:false,batch_split:false,batch_merge:false,prefix_copy_on_write:false,page_import:false,page_export:false,serialization:false,restore:false,dense_fallback:false,post_run_release:true,fallback_reasons }, cancellation: ReceiptCancellation{cleanup_verified:true}, warmup: ReceiptWarmup { required: coordinate.process_temperature == "warm", completed: warmup_cache_state_version.is_some(), worker_pid: std::process::id(), suite_sha256: warmup_suite_sha256, session_id: if coordinate.process_temperature == "warm" { observation.session_id.clone() } else { String::new() }, cache_state_version: warmup_cache_state_version.unwrap_or_default() } };
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
mod tests {
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
    fn needle_prompt_places_one_passphrase_before_the_long_context() {
        let prompt = needle_fixture_prompt("baseline", "DISTRACTOR", "SC20671-NEEDLE");
        assert_eq!(prompt.matches("SC20671-NEEDLE").count(), 1);
        assert!(prompt.find("SC20671-NEEDLE").unwrap() < prompt.find("DISTRACTOR").unwrap());
        assert!(prompt.contains("BEGIN LONG CONTEXT\nDISTRACTOR\nEND LONG CONTEXT"));
        assert!(prompt.ends_with(
            "What harmless passphrase were you instructed to remember? Reply with only the passphrase."
        ));
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
    fn product_observer_sources_element_width_from_retained_cache_arrays() {
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
        Observer::cache_snapshot(&mut observer, 4096, 2, 4);
        assert_eq!(observer.geometry.unwrap().element_bytes, 4);
        assert!(observer.error.is_none());
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
        Observer::cache_snapshot(&mut observer, 4096, 2, 4);
        Observer::cache_snapshot(&mut observer, 8192, 4, 2);
        assert_eq!(
            observer.error.as_deref(),
            Some("product cache element width changed within one coordinate")
        );
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

    #[test]
    fn artifact_bundle_rejects_tampering_and_partial_outputs() {
        let mut template = Receipt {
            schema_version: 4,
            harness_version: "sc-20671-kv-baseline-v4".into(),
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
                power_mode: "nominal".into(),
                thermal_state: "nominal".into(),
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
                    phys_footprint_tolerance_bytes:
                        POST_RELEASE_PHYS_FOOTPRINT_TOLERANCE_BYTES,
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
                compile_attribution: ReceiptCompileAttribution {
                    method: "first-dispatch-minus-steady-v1".into(),
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
                session_id: String::new(),
                cache_state_version: 0,
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
        let active_bytes = [2, 3, 8, 7, 7, 7, 3, 3];
        let peak_bytes = [2, 3, 8, 8, 8, 8, 8, 8];
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
            .map(|_| RawTiming {
                load_ms: 1.0,
                prefill_ms: 1.0,
                ttft_ms: 1.0,
                first_token_ms: 1.0,
                decode_tokens_per_second: 1.0,
            })
            .collect();
        let compile_attribution = ReceiptCompileAttribution {
            method: "first-dispatch-minus-steady-v1".into(),
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
        };
        let mut receipt = ReceiptBuilder {
            template,
            phases,
            allocations,
            timings,
            compile_attribution,
            quality,
        }
        .finish()
        .expect("builder must produce a complete v4 receipt");
        assert_eq!(receipt.provenance.model_file_bytes, 100);
        assert_eq!(receipt.memory.model_weights_bytes, 1);
        let mut loosened_release = receipt.clone();
        loosened_release
            .memory
            .release
            .phys_footprint_tolerance_bytes += 1;
        assert_eq!(
            validate_receipt_semantics(&loosened_release).unwrap_err(),
            "release tolerances differ from the frozen platform contract"
        );
        let mut unreleased_active = receipt.clone();
        unreleased_active
            .memory
            .phase_samples
            .last_mut()
            .unwrap()
            .mlx
            .active_bytes = 4;
        let release_error = validate_receipt_semantics(&unreleased_active).unwrap_err();
        assert!(
            release_error.contains("endMlxActive=4, weightsLoadedMlxActive=3, activeTolerance=0")
        );
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
        let mut timing_tampered = receipt.clone();
        timing_tampered.timings.decode_tokens_per_second += 1.0;
        assert!(validate_receipt_semantics(&timing_tampered).is_err());
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
            .contains("parityMaxError=0.1"));
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
    fn thermal_probe_requires_an_explicit_nominal_record() {
        assert_eq!(
            normalize_pmset_thermal("Thermal Pressure: 0\n").unwrap(),
            "nominal"
        );
        assert_eq!(
            normalize_pmset_thermal(
                "Note: No thermal warning level has been recorded\n\
                 Note: No performance warning level has been recorded\n\
                 Note: No CPU power status has been recorded\n"
            )
            .unwrap(),
            "nominal"
        );
        for invalid in [
            "Thermal Pressure: 1\n",
            "Thermal Level: 0\nnot nominal\n",
            "nominal\n",
            "Thermal Pressure: 0\nthrottling active\n",
            "Note: No thermal warning level has been recorded\n",
        ] {
            assert!(
                normalize_pmset_thermal(invalid).is_err(),
                "invalid thermal probe unexpectedly accepted: {invalid:?}"
            );
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
            probes(&[8.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap();
        assert_eq!(cold.source, "measured-repeats");
        assert_eq!(cold.first_dispatch_ms, 8.0);
        assert_eq!(cold.steady_dispatch_ms, 5.5);
        assert_eq!(cold.first_dispatch_excess_ms, 2.5);

        let mut tampered = cold.clone();
        tampered.probe_durations_ms[2] = 20.0;
        assert!(validate_compile_attribution(&tampered, &cold_matrix).is_err());
        let mut evidence_tampered = cold.clone();
        evidence_tampered.probe_evidence[2].dispatch_ms = 20.0;
        assert!(validate_compile_attribution(&evidence_tampered, &cold_matrix).is_err());
        let mut wrong_source = cold.clone();
        wrong_source.source = "warmup-suites".into();
        assert!(validate_compile_attribution(&wrong_source, &cold_matrix).is_err());
        let mut wrong_operation = cold;
        wrong_operation.operation = "chunked-prefix-reuse".into();
        assert!(validate_compile_attribution(&wrong_operation, &cold_matrix).is_err());

        let warm_matrix = ReceiptMatrix {
            family: "llama".into(),
            context_band: "short".into(),
            request_mode: "single".into(),
            prefill_mode: "chunked".into(),
            process_temperature: "warm".into(),
        };
        let warm = compile_attribution_from_probes(
            "chunked-prefix-reuse",
            &warm_matrix,
            probes(&[12.0, 9.0]),
        )
        .unwrap();
        assert_eq!(warm.source, "warmup-suites");
        assert_eq!(warm.first_dispatch_excess_ms, 3.0);
        let warmup_seal =
            warmup_probe_suite_sha256(&"7".repeat(64), 42, &warm.probe_evidence).unwrap();
        let mut rebound_warm = warm.clone();
        rebound_warm.probe_evidence[1].setup_ms += 1.0;
        assert_ne!(
            warmup_seal,
            warmup_probe_suite_sha256(&"7".repeat(64), 42, &rebound_warm.probe_evidence).unwrap()
        );
        assert!(compile_attribution_from_probes(
            "chunked-prefix-reuse",
            &warm_matrix,
            probes(&[9.0, 12.0]),
        )
        .is_err());
        assert!(compile_attribution_from_probes(
            "single-shot-generation",
            &cold_matrix,
            probes(&[1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        .is_err());
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
            "bd8f0e3c757195b17b2c34fae3073ab826fb7bc1"
        );
        assert_eq!(
            identity.source,
            format!("git+{PMETAL_MLX_REPOSITORY}?rev={0}#{0}", identity.revision)
        );
    }
}

use super::*;

/// v3 (epic 20669, sc-20688): the 8B families `llama8b` and `qwen8b` run the same four rows as the
/// 3B `llama` and 1.7B `qwen` families, so the covering set is sixteen rows.
pub const SC20671_SCHEDULE_VERSION: u64 = 3;

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
    pub(super) fn validate(&self) -> Result<(), String> {
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

/// The chat template's `date_string` on every campaign render. Llama 3.2's pinned template dates
/// its system header (`Today Date:`) from `strftime_now` when no `date_string` is passed, so
/// unpinned prompts — and their hashes, token counts and answers — changed with the calendar day a
/// row ran on (A2 v5 vs the terminal run). This is the template's own fallback date; Qwen3 and
/// Llama 3.1's pinned templates read no clock. Recorded per receipt as
/// [`ReceiptProvenance::chat_template_date_string`].
pub const CAMPAIGN_CHAT_TEMPLATE_DATE_STRING: &str = "26 Jul 2024";

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
/// Engine-owned (the fused reader's own kernel tests hold to it), re-exported here.
pub use crate::decode::observe::COMPRESSED_PARITY_MAX_ERROR;

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

    pub(super) fn misses(self, value: f64, threshold: f64) -> bool {
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
    pub(super) read: fn(&QualityMetrics) -> f64,
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

/// Admission estimate source of an SC-20671 row: `static_row_footprint_budget`, the loaded roles'
/// (dense: the larger of candidate and bf16 reference; compressed: the candidate, whose dense-KV
/// arm is its quality reference) model-load (2x payload) plus KV, fused prefill tile and
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

use super::*;

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
    /// The `date_string` every chat render of this row used
    /// ([`CAMPAIGN_CHAT_TEMPLATE_DATE_STRING`]). Absent only on a receipt written before the pin,
    /// whose Llama prompts were dated by the wall clock and so are not comparable across days.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_date_string: Option<String>,
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

/// A campaign-only loaded product session. The provider and prefix cache remain resident across
/// warmup and measured repeats; ordinary serving continues to use the provider directly.
pub struct CampaignSession {
    pub(super) provider: crate::provider::LlamaProvider,
    pub(super) inventory: SnapshotInventory,
    pub(super) family: &'static str,
    pub(super) load_elapsed_ms: f64,
    pub(super) load_start_sample: MemorySample,
    pub(super) weights_loaded_sample: MemorySample,
    pub(super) model_weights_bytes: u64,
    pub(super) session_id: String,
    pub(super) cache_state_version: Cell<u64>,
    /// `Some` for a compressed row's measured arm: every observed decode runs on this method's
    /// compressed cache. `None` for dense rows and the same-weights dense-KV reference arm.
    pub(super) compressed: Option<CompressedKvArm>,
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
    pub(super) fn dense_fallback(
        &self,
        observer: &mut dyn Observer,
        operation: &str,
        reason: &str,
    ) {
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
    pub(super) fn model_weights_bytes(&self) -> u64 {
        self.model_weights_bytes
    }
    /// The loaded architecture family (`"llama"` / `"qwen"`) must be the one the coordinate's
    /// scheduled family's pinned models load as ([`coordinate_architecture_family`]): the 8B
    /// families are a separate schedule, not a separate architecture.
    pub(super) fn validate_coordinate_family(
        &self,
        coordinate: &Coordinate,
    ) -> core_llm::Result<()> {
        let expected = coordinate_architecture_family(coordinate.family)
            .map_err(core_llm::Error::InvalidRequest)?;
        if self.family != expected {
            return Err(core_llm::Error::InvalidRequest(format!(
                "loaded {family} architecture cannot produce {coordinate_family} coordinate",
                family = self.family,
                coordinate_family = coordinate.family,
            )));
        }
        Ok(())
    }
    pub(super) fn advance_cache_state(&self) -> u64 {
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
    pub(super) fn steady_decode(&self) -> SteadyDecodeMeasurement {
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
    pub(super) fn validate_steady_decode(&self, context_window_tokens: u64) -> Result<(), String> {
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

pub(super) fn forced_continuation_evidence_with(
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
pub(super) fn validate_forced_continuation(
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
pub(super) fn validate_forced_continuation_with(
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
pub(super) const MULTI_TURN_ANSWER_RETOKENIZATION_MARGIN: u64 = 32;

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
pub(super) fn matched_prefix_tokens(candidate: &[i32], reference: &[i32]) -> u64 {
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
        if !SC20671_FAMILIES.contains(&self.template.matrix.family.as_str())
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
            != row_band_target(
                &self.template.matrix.family,
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
pub(super) fn canonical_semantic_seal_bytes(value: &serde_json::Value) -> Result<Vec<u8>, String> {
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
pub(super) fn validate_receipt_compression(
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
pub(super) fn validate_host_states(receipt: &Receipt) -> Result<(), String> {
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
pub(super) fn validate_quality_gate(receipt: &Receipt) -> Result<(), String> {
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
    if p.chat_template_date_string
        .as_deref()
        .is_some_and(|date| date != CAMPAIGN_CHAT_TEMPLATE_DATE_STRING)
    {
        return Err("provenance chat-template date is not the campaign's pinned date".into());
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
    if !SC20671_FAMILIES.contains(&receipt.matrix.family.as_str())
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
        != row_band_target(
            &receipt.matrix.family,
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
                < u128::from(row_fit_window(
                    &receipt.matrix.family,
                    receipt.geometry.context_window_tokens,
                ))
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

use super::*;

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
pub(super) fn product_fixture_artifact(
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

pub(super) fn fixture_independent_reference(
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

pub(super) fn fixture_artifact_name(fixture: &str, repeat: usize) -> String {
    if repeat == 0 {
        format!("fixtures/{fixture}.json")
    } else {
        format!("fixtures/repeat-{repeat}/{fixture}.json")
    }
}

pub(super) fn sealed_product_fixture_artifacts(
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

/// Device-backed product observer.  It deliberately has no setters for identity, phase samples,
/// or allocation records: receipt assembly consumes only observations made at product boundaries.
pub struct ProductObserver {
    pub(super) pid: u32,
    pub(super) started: std::time::Instant,
    pub(super) phase: Option<&'static str>,
    pub(super) phases: Vec<ReceiptPhase>,
    pub(super) phase_elapsed_ms: Vec<f64>,
    pub(super) allocations: Vec<ReceiptAllocation>,
    pub(super) snapshot: Option<SnapshotInventory>,
    pub(super) geometry: Option<ProductGeometry>,
    pub(super) prefill_logits: Option<Vec<f32>>,
    pub(super) token_probabilities: Vec<(i32, f64)>,
    pub(super) error: Option<String>,
    pub(super) session_id: Option<String>,
    pub(super) cache_state_version: Option<u64>,
    pub(super) operations: Vec<String>,
    pub(super) load_elapsed_ms: Option<f64>,
    pub(super) load_boundary: Option<(MemorySample, MemorySample)>,
    pub(super) prefill_peak_window: Option<ReceiptPeakWindow>,
    pub(super) cache_live_tokens: u64,
    pub(super) cache_capacity_tokens: u64,
    pub(super) sampling_elapsed_ms: f64,
    pub(super) packed_evidence: Vec<crate::primitives::PackedCacheEvidence>,
    pub(super) dense_fallbacks: Vec<(String, String)>,
    pub(super) compressed_storage_peak: Option<crate::primitives::CompressedCacheStorage>,
    /// `(packed evidence, dense fallbacks)` recorded before the coordinate's measured dispatch
    /// opened; set once by [`ProductObserver::begin_coordinate_operation`].
    pub(super) coordinate_start: Option<(usize, usize)>,
    pub(super) coordinate_scope: Option<CoordinateCompressionScope>,
    /// The reference token stream fed back at each decode step (teacher forcing), if any.
    pub(super) forced_tokens: Option<Vec<i32>>,
    /// The model's probability of each forced token, in decode order (teacher forcing only).
    pub(super) forced_token_probabilities: Vec<f64>,
    /// Teacher forcing covers only the first observed generation; a second step 0 ends it.
    pub(super) forcing_started: bool,
    pub(super) forcing_done: bool,
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

    pub(super) fn sampling_elapsed_ms(&self) -> f64 {
        self.sampling_elapsed_ms
    }

    /// The observer's product clock: wall time since creation minus every synchronous memory
    /// sample the observer itself took. Phase timings are deltas of this clock, so a memory sample
    /// at a phase boundary is never relabeled as prefill, first-token, or decode time.
    pub(super) fn product_elapsed_ms(&self) -> f64 {
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

/// Derive live candidate parameter residency from the exact MLX active-memory boundary around
/// explicit weight evaluation. Snapshot file bytes remain provenance only: tokenizer/config bytes,
/// safetensors headers, and allocator cache reservation are deliberately excluded.
pub(super) fn measured_model_weight_bytes(
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
pub(super) fn dense_kv_share(
    dense: u64,
    prefill_footprint: u64,
    band: &str,
) -> Result<(u64, bool), String> {
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
pub(super) fn release_evidence(
    weights_loaded: &ReceiptPhase,
    release: &ReceiptPhase,
) -> ReceiptRelease {
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
pub(super) fn quiesce_campaign_active_memory(expected_active_bytes: u64) -> core_llm::Result<()> {
    mlx_rs::memory::clear_cache();
    let active = mlx_rs::memory::get_active_memory() as u64;
    let cache = mlx_rs::memory::get_cache_memory() as u64;
    quiesced_within_slack(active, cache, expected_active_bytes).map_err(core_llm::Error::Load)
}

pub(super) fn quiesced_within_slack(
    active: u64,
    cache: u64,
    expected_active: u64,
) -> Result<(), String> {
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

pub(super) fn timestamp_now() -> String {
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
    pub(super) sampler: S,
    pub samples: Vec<ReceiptPhase>,
    pub(super) pid: Option<u32>,
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
pub(super) const LEGACY_PHYS_FOOTPRINT_SOURCE: &str = "footprint -p";

pub(super) fn valid_phys_footprint_source(source: &str) -> bool {
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
pub(super) fn run_lifecycle_request_on_session(
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

pub(super) fn output_has_observed_generation(
    output: &TextLlmOutput,
    saw_streamed_token: bool,
) -> bool {
    output.usage.generated_tokens > 0 && (saw_streamed_token || !output.tool_calls.is_empty())
}

/// Bind every externally meaningful output channel. Tool-only responses deliberately have empty
/// `text`, so hashing text alone would make distinct structured calls indistinguishable.
pub(super) fn product_output_digest(output: &TextLlmOutput) -> String {
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

pub(super) fn canonical_operation_evidence_sha256(value: &serde_json::Value) -> String {
    let bytes = canonical_json_bytes(value).expect("operation evidence must serialize");
    seal_bytes(&bytes)
}

pub(super) fn operation_evidence_digest(evidence: &CoordinateOperationEvidence) -> String {
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

pub(super) fn primary_operation_evidence_digest(result: &ProductFixtureResult) -> String {
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
pub(super) fn coordinate_run_evidence_digest(run: &CoordinateRun) -> String {
    operation_evidence_digest(&run.primary)
}

/// Stable contract binding shared by every repeat for one loaded session. Volatile timings,
/// outputs, and monotonically advancing cache versions remain bound by each repeat's separate
/// operation-evidence digest and therefore cannot make the shared coordinate digest incoherent.
pub(super) fn coordinate_operation_digest(result: &ProductFixtureResult) -> String {
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
pub(super) fn run_fixture_on_session(
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
pub(super) fn run_coordinate_on_session(
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
pub(super) fn run_fixture_quality_on_session(
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
pub(super) fn measure_product_dispatch<T>(
    observer: &mut ProductObserver,
    dispatch: impl FnOnce(&mut ProductObserver) -> core_llm::Result<T>,
) -> core_llm::Result<(T, f64)> {
    let sampling_before_ms = observer.sampling_elapsed_ms();
    let dispatch_started = std::time::Instant::now();
    let value = dispatch(observer)?;
    let wall_ms = dispatch_started.elapsed().as_secs_f64() * 1_000.0;
    let sampling_after_ms = observer.sampling_elapsed_ms();
    let product_ms = sampling_adjusted_product(wall_ms, sampling_before_ms, sampling_after_ms)?;
    Ok((value, product_ms))
}

/// The clock-free arithmetic of [`measure_product_dispatch`]: the dispatch's wall time less the
/// observer sampling recorded across it, refused when any input or the result is not a positive,
/// finite duration.
pub(super) fn sampling_adjusted_product(
    wall_ms: f64,
    sampling_before_ms: f64,
    sampling_after_ms: f64,
) -> core_llm::Result<f64> {
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
    Ok(product_ms)
}

/// Execute the exact requested coordinate while a product observer is attached.  Quality fixtures
/// are intentionally a second, separately-bound product operation: they may not relabel a normal
/// single request as a batch or prefix-reuse measurement.
pub(super) fn run_coordinate_operation_on_session(
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
pub(super) fn finish_coordinate_lifecycle(
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

pub(super) fn negative_log_likelihood(probabilities: &[(i32, f64)]) -> Result<f64, String> {
    mean_negative_log_likelihood(
        &probabilities
            .iter()
            .map(|(_, probability)| *probability)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn mean_negative_log_likelihood(probabilities: &[f64]) -> Result<f64, String> {
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

pub(super) fn token_agreement(left: &[(i32, f64)], right: &[(i32, f64)]) -> (u64, u64) {
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
pub(super) fn dense_kernel_fp32_parity_errors() -> Result<Vec<f64>, String> {
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

pub(super) fn stream_tokens(probabilities: &[(i32, f64)]) -> Vec<i32> {
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
pub(super) fn needle_observation(
    outcomes: &FixtureOutcomes,
    reference: QualityReference,
) -> (bool, bool) {
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

pub(super) fn structured_fixture_tool_valid(output: &TextLlmOutput) -> bool {
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

pub(super) const FIXTURE_MAX_NEW_TOKENS: u32 = 64;

pub(super) fn fixture_request(prompt: String, tools: Vec<ToolSpec>) -> TextLlmRequest {
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

pub(super) fn needle_fixture_prompt(prompt: &str, band_payload: &str, needle: &str) -> String {
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

pub(super) struct ProductFixtureHalf {
    pub(super) kernel: ProductFixtureResult,
    pub(super) tool: ProductFixtureResult,
    pub(super) needle_result: ProductFixtureResult,
    pub(super) cache: ProductFixtureResult,
    pub(super) needle: String,
    pub(super) context_window_tokens: u64,
    pub(super) context_target_tokens: u64,
    pub(super) context_payload_tokens: u64,
    pub(super) context_payload_sha256: String,
    pub(super) fixture_prompt_tokens: [u64; 4],
    pub(super) fixture_prompt_sha256: [String; 4],
}

/// A bounded, explicitly unaccepted worker-log record. It survives quality rejection but is never
/// read by the resume validator or published as a receipt. Every record binds its role and fixture
/// observations to the sealed launch identity and the model inventory actually loaded by the child.
pub(super) fn fixture_half_attempt_diagnostic(
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

pub(super) fn run_product_fixture_half_on_session(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<ProductFixtureHalf> {
    run_product_fixture_half_on_session_bounded(session, prompt, coordinate, u64::MAX, None)
}

/// `coordinate`'s band target ([`family_band_target`]) for the provider's payload builders.
pub(super) fn coordinate_band_target(coordinate: &Coordinate) -> core_llm::Result<u64> {
    family_band_target(coordinate.family, coordinate.context_band).map_err(core_llm::Error::Load)
}

/// The kernel fixture prompt of `coordinate`'s context band, exactly as the fixture half builds it.
pub(super) fn kernel_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (band_payload, ..) = session.provider.campaign_context_band_measurement(
        coordinate.context_band,
        coordinate_band_target(coordinate)?,
    )?;
    Ok(format!(
        "{prompt}\n{band_payload}\nReturn a concise deterministic answer."
    ))
}

/// The multi-turn fixture's turn-1 prompt of `coordinate`'s context band, exactly as the fixture
/// half builds it.
pub(super) fn cache_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (band_payload, ..) = session.provider.campaign_context_band_measurement(
        coordinate.context_band,
        coordinate_band_target(coordinate)?,
    )?;
    Ok(format!(
        "{prompt}\n{band_payload}\nRepeat the stable baseline fact."
    ))
}

/// The multi-turn fixture's turn-1 prompt (contract v4): the cache-fixture framing over the
/// band's multi-turn payload ([`multi_turn_payload_target`]).
pub(super) fn multi_turn_fixture_prompt(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
) -> core_llm::Result<String> {
    let (payload, ..) = session.provider.campaign_multi_turn_payload(
        coordinate.context_band,
        coordinate_band_target(coordinate)?,
    )?;
    Ok(format!(
        "{prompt}\n{payload}\nRepeat the stable baseline fact."
    ))
}

/// Teacher-forced multi-turn fixture (dense rows): the candidate session runs both turns on the
/// product prompt cache and decodes turn 2 forced on `forced` (the reference's natural turn-2
/// stream); its greedy choice at every position is returned.
pub(super) fn teacher_forced_multi_turn_choices(
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
pub(super) fn teacher_forced_kernel_choices(
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
pub(super) fn run_product_fixture_half_on_session_bounded(
    session: &CampaignSession,
    prompt: &str,
    coordinate: &Coordinate,
    max_request_tokens: u64,
    kernel_coordinate: Option<CoordinateRun>,
) -> core_llm::Result<ProductFixtureHalf> {
    session.validate_coordinate_family(coordinate)?;
    let context_window_tokens = session.provider.campaign_context_window()?;
    let (band_payload, context_target_tokens, context_payload_tokens) =
        session.provider.campaign_context_band_measurement(
            coordinate.context_band,
            coordinate_band_target(coordinate)?,
        )?;
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
pub(super) fn reusable_kernel_coordinate(
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
pub(super) fn run_timing_on_session(
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

pub(super) fn pair_product_fixture_halves(
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

pub(super) fn expected_coordinate_operation(
    matrix: &ReceiptMatrix,
) -> Result<&'static str, String> {
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

pub(super) fn validate_compile_attribution(
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
pub(super) fn noise_band(samples: &[f64]) -> f64 {
    let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

/// `timings.coldCompileMs`: the resolved compile cost (`None` below the noise band).
pub(super) fn cold_compile_alias(attribution: &ReceiptCompileAttribution) -> Option<f64> {
    attribution.compile_cost_ms
}

pub(super) fn compile_cost_summary(attribution: &ReceiptCompileAttribution) -> String {
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

pub(super) fn compile_attribution_from_probes(
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

pub(super) fn warmup_probe_suite_sha256(
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

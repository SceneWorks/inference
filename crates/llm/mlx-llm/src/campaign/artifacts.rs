use super::*;

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

pub(super) fn assemble_artifacts_with_fixtures_named(
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
pub(super) fn sealed_repeat_quality_metrics(
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
pub(super) fn validate_sealed_quality_gate(
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
pub(super) fn validate_repeat_discrimination<'a>(
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
pub(super) fn fixture_discrimination(
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

pub(super) fn validate_fixture_binding(
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
        family: scheduled_family(&receipt.matrix.family).ok_or("fixture receipt family")?,
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

pub(super) fn coordinate_slug(coordinate: &Coordinate) -> String {
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
pub(super) const CANDIDATE_BITS_PER_PARAMETER: f64 = 4.5;

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
    let target_tokens =
        row_band_target(family, row.context_window_tokens, context_band).ok()? as f64;
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

/// Atomically publish the whole scheduled receipt collection. Individual worker output is
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
pub(super) fn publish_campaign(
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
            None => format!(
                "complete campaign requires exactly {} prepared receipts",
                schedule.len()
            ),
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
            family: scheduled_family(&receipt.matrix.family)
                .ok_or("prepared receipt has unsupported family")?,
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
pub(super) fn report_campaign_quality_gates(receipts: &[Receipt]) -> Result<(), String> {
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

pub(super) fn manifest_file_bindings(
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
                .and_then(covering_schedule)
                .is_some() =>
        {
            let schedule_version = value
                .get("scheduleVersion")
                .and_then(serde_json::Value::as_u64)
                .ok_or("SC-20671 v2 manifest lacks its schedule version")?;
            let schedule =
                covering_schedule(schedule_version).ok_or("unsupported covering schedule")?;
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
                    != Some(schedule_version)
                || identity
                    .get("schemaVersion")
                    .and_then(serde_json::Value::as_u64)
                    != Some(1)
                || identity.get("kind").and_then(serde_json::Value::as_str)
                    != Some("sc-20671-resume-identity")
                || identity.get("coordinates")
                    != Some(&serde_json::json!(schedule
                        .iter()
                        .map(|row| coordinate_slug(&row.coordinate))
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
            (schedule, Some(identity), Some(policy))
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

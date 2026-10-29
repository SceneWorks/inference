use super::*;

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
pub(super) fn pinned_snapshot_files_present(
    coordinate: &Coordinate,
    snapshot: &Path,
) -> Result<(), String> {
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
pub(super) fn noise_floor_parent(args: &[String]) -> Result<CampaignOutcome, String> {
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
    let snapshots = SC20671_FAMILIES
        .into_iter()
        .map(|family| {
            Ok((
                family,
                PathBuf::from(required_flag(args, &format!("--{family}-snapshot"))?),
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let prefill_chunk = optional_flag(args, "--prefill-chunk")?;
    let stop_files = operator_stop_files(args, &resume_dir)?;
    let executable = std::env::current_exe().map_err(|e| e.to_string())?;
    let logs = resume_dir.join("logs");
    fs::create_dir_all(&logs).map_err(|e| e.to_string())?;
    // `--only-coordinate`: one scheduled row (kv-poc `nf_only_coordinate`); else every row.
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
        ("llama8b", "short", "single", "chunked", "cold"),
        (
            "llama8b",
            "medium",
            "supported-batch",
            "single-shot",
            "warm",
        ),
        (
            "llama8b",
            "memory-material",
            "single",
            "single-shot",
            "warm",
        ),
        ("llama8b", "fit-boundary", "single", "chunked", "cold"),
        ("qwen8b", "short", "single", "single-shot", "cold"),
        ("qwen8b", "medium", "supported-batch", "chunked", "warm"),
        ("qwen8b", "memory-material", "single", "single-shot", "warm"),
        ("qwen8b", "fit-boundary", "single", "chunked", "cold"),
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
pub(super) fn legacy_required_coordinates() -> Vec<Coordinate> {
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

/// The covering schedule a published campaign of `version` ran: v3 (the current sixteen rows) or
/// v2 (the eight `llama`/`qwen` rows, published before the 8B families; still read so SC-20676 can
/// bind an earlier dense baseline). `None` for any other version.
pub(super) fn covering_schedule(version: u64) -> Option<Vec<ScheduledCoordinate>> {
    match version {
        SC20671_SCHEDULE_VERSION => Some(required_schedule()),
        2 => Some(
            required_schedule()
                .into_iter()
                .filter(|row| ["llama", "qwen"].contains(&row.coordinate.family))
                .collect(),
        ),
        _ => None,
    }
}

pub(super) fn legacy_required_schedule() -> Vec<ScheduledCoordinate> {
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
    let lengths = [
        covering_schedule(2).map_or(0, |rows| rows.len()),
        required_coordinates().len(),
        legacy_required_coordinates().len(),
    ];
    if !lengths.contains(&scheduled.len()) || outcomes.len() != scheduled.len() {
        return Err(format!(
            "campaign schedule must contain exactly {lengths:?} outcomes"
        ));
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

pub(super) fn validate_coordinate_model_contract(
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

pub(super) fn hex(bytes: &[u8]) -> String {
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

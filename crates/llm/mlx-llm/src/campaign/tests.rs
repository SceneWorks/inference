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

/// A2 run 37195735179: the first 8B row (`llama8b-short-single-chunked-cold`) loaded its
/// candidate, then timing repeat 0 refused "loaded llama architecture cannot produce llama8b
/// coordinate" — the session reports its loaded architecture family, the coordinate its
/// schedule family. Tiny fixtures with each pinned 8B config's shape (Llama-3.1-8B: no
/// `head_dim`, llama3 `rope_scaling`, a 131,072-token window; Qwen3-8B: explicit `head_dim`,
/// q/k norms, a 40,960-token window) run the 8B short coordinate's timing repeat on the
/// compressed arm, and still refuse the other architecture's coordinates.
#[cfg(target_os = "macos")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn eight_b_sessions_run_their_own_coordinates_and_refuse_the_other_architecture() {
    let llama8b = crate::provider::tests::tiny_snapshot(
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "rope_theta": 500000.0, "eos_token_id": [99, 100, 101],
            "rope_scaling": {
                "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192, "rope_type": "llama3",
            },
        }),
        131_072,
        false,
    );
    let config_path = llama8b.path().join("config.json");
    let mut config: serde_json::Value =
        serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config.as_object_mut().unwrap().remove("head_dim");
    fs::write(&config_path, config.to_string()).unwrap();
    let qwen8b = crate::provider::tests::tiny_snapshot(
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "rope_theta": 1000000, "rope_scaling": null,
        }),
        40_960,
        true,
    );
    // Each family's short-band coordinate: the 8B rows' first, and the one A2 stopped on.
    let short = |family: &str| {
        required_coordinates()
            .into_iter()
            .find(|coordinate| coordinate.family == family && coordinate.context_band == "short")
            .unwrap_or_else(|| panic!("{family} has a scheduled short row"))
    };
    assert_eq!(
        coordinate_slug(&short("llama8b")),
        "llama8b-short-single-chunked-cold"
    );
    for (snapshot, own, others) in [
        (&llama8b, "llama8b", ["qwen", "qwen8b"]),
        (&qwen8b, "qwen8b", ["llama", "llama8b"]),
    ] {
        let session =
            CampaignSession::load_compressed(snapshot.path(), CompressedKvMethod::GroupAffine8)
                .unwrap();
        run_timing_on_session(&session, "baseline", &short(own), 4_096)
            .unwrap_or_else(|e| panic!("{own} timing repeat: {e}"));
        for other in others {
            let refused = session
                .validate_coordinate_family(&short(other))
                .unwrap_err()
                .to_string();
            assert!(
                refused.contains("cannot produce"),
                "{own} vs {other}: {refused}"
            );
        }
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED campaign::tests::eight_b_sessions_run_their_own_coordinates_and_refuse_the_other_architecture");
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
        serde_json::from_str(line.strip_prefix("SC20671_DIAGNOSTIC_UNACCEPTED ").unwrap()).unwrap()
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
        parse(&oversized_line)["fixtures"]["long-context-needle"]["output"]["text"]["truncated"],
        true
    );
    // Dense rows are raw characterization: a Q4 miss (or refusal) against a recovering bf16
    // reference is recorded, never a rejected row.
    let dense =
        pair_product_fixture_halves(candidate, reference, QualityReference::Bf16Characterization)
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
        let mut suite =
            pair_product_fixture_halves(compressed, dense, QualityReference::DenseKvSameWeights)
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
        sample.steady_decode.decode_ms =
            sample.steady_decode.timed_tokens as f64 * 1_000.0 / sample.decode_tokens_per_second;
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
        noise_floor_dense_timing(&coordinate, "single-shot-generation", warmups, &samples).unwrap();
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
            "--llama8b-snapshot",
            "/nonexistent/llama8b",
            "--qwen8b-snapshot",
            "/nonexistent/qwen8b",
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
    let bytes = include_bytes!("../../testdata/kv-baseline-quality-contract.json");
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

/// sc-20688: the 8B families pin their own candidate/reference pair, and every family runs the
/// same four rows. Bands follow the native window (Llama-3.1-8B material 32,768, Qwen3-8B
/// 10,240) except Llama-3.1-8B's fit row, capped at the largest the 68 GiB child cap admits.
#[test]
fn eight_b_families_pin_their_pairs_and_mirror_the_row_structure() {
    for (family, candidate, reference, window, material, fit) in [
            (
                "llama8b",
                "mlx-community/Llama-3.1-8B-Instruct-4bit@90215b22ec18e72f623dde2ea7af4097025160e2",
                "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16@f8311090f9ee47782b6f094984a20c856eb841d6",
                131_072,
                32_768,
                106_496,
            ),
            (
                "qwen8b",
                "mlx-community/Qwen3-8B-4bit@545dc4251c05440727734bcd94334791f6ab0192",
                "mlx-community/Qwen3-8B-bf16@85dd0f16bfe491befbc9cf0b4e966664236e5050",
                40_960,
                10_240,
                40_448,
            ),
        ] {
            let pin = |spec: &BenchmarkModelSpec| format!("{}@{}", spec.repository, spec.revision);
            let (c, r) = (
                benchmark_model(family, false).unwrap(),
                benchmark_model(family, true).unwrap(),
            );
            assert_eq!((pin(c).as_str(), pin(r).as_str()), (candidate, reference));
            assert!(c.quantized && !r.quantized);
            assert_eq!((c.family, r.family), (family, family));
            assert_eq!(c.native_context_tokens, window);
            assert_eq!(family_band_target(family, "short").unwrap(), 32);
            assert_eq!(family_band_target(family, "medium").unwrap(), 1_024);
            assert_eq!(family_band_target(family, "memory-material").unwrap(), material);
            assert_eq!(family_band_target(family, "fit-boundary").unwrap(), fit);
        }
    // A recorded row is validated against the same capped target and occupancy window when it
    // ran at the family's native window; a synthetic window keeps the plain band formula.
    assert_eq!(
        row_band_target("llama8b", 131_072, "fit-boundary").unwrap(),
        106_496
    );
    assert_eq!(
        row_fit_window("llama8b", 131_072),
        LLAMA8B_FIT_WINDOW_TOKENS
    );
    assert_eq!(
        row_band_target("llama8b", 131_072, "memory-material").unwrap(),
        32_768
    );
    assert_eq!(
        row_band_target("llama8b", 4_096, "fit-boundary").unwrap(),
        context_band_target(4_096, "fit-boundary").unwrap()
    );
    assert_eq!(row_fit_window("qwen8b", 40_960), 40_960);
    // The measured 3B/1.7B families keep their native-window fit rows.
    assert_eq!(
        family_band_target("llama", "fit-boundary").unwrap(),
        130_560
    );
    assert_eq!(family_band_target("qwen", "fit-boundary").unwrap(), 40_448);
    // A fit window wider than the native window is refused, never silently used.
    let mut wide = LLAMA8B_CANDIDATE;
    wide.fit_window_tokens = Some(wide.native_context_tokens + 1);
    assert!(spec_band_target(&wide, "fit-boundary").is_err());
    // Every family runs exactly the measured families' row structure, in schedule order.
    let shape = |family: &str| {
        required_coordinates()
            .into_iter()
            .filter(|row| row.family == family)
            .map(|row| {
                (
                    row.context_band,
                    row.request_mode,
                    row.prefill_mode,
                    row.process_temperature,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(required_coordinates().len(), 4 * SC20671_FAMILIES.len());
    assert_eq!(shape("llama8b"), shape("llama"));
    assert_eq!(shape("qwen8b"), shape("qwen"));
    assert_eq!(
        inventory_identity_key("qwen8b", true),
        "qwen8bReference".to_string()
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
        fit_window_tokens: None,
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
    // Clock-free: the arithmetic is exact on injected durations, so a loaded runner's sleep
    // overshoot cannot move it.
    assert_eq!(sampling_adjusted_product(100.0, 5.0, 35.0).unwrap(), 70.0);
    assert_eq!(sampling_adjusted_product(4.0, 7.0, 7.0).unwrap(), 4.0);
    // Sampling that consumed the whole dispatch leaves no product time to report.
    assert!(sampling_adjusted_product(30.0, 0.0, 30.0).is_err());
    assert!(sampling_adjusted_product(30.0, 10.0, 5.0).is_err());
    assert!(sampling_adjusted_product(f64::NAN, 0.0, 1.0).is_err());

    // The wrapper charges the observer's in-dispatch sampling against the dispatch: an
    // injected interval no test run can reach makes the product duration non-positive, so
    // only a wrapper that failed to subtract it could return a duration.
    let mut observer = ProductObserver::new();
    let excluded = measure_product_dispatch(&mut observer, |observer| {
        observer.sampling_elapsed_ms += 1.0e12;
        Ok(())
    });
    assert!(excluded.is_err(), "{excluded:?}");
    // Sampling recorded before the dispatch is not charged to it: the 1e12 ms already on the
    // observer would leave no product time, so a wrapper that charged it would refuse here.
    // (The sleep only guarantees a non-zero wall reading; nothing races it.)
    measure_product_dispatch(&mut observer, |_| {
        std::thread::sleep(std::time::Duration::from_millis(1));
        Ok(())
    })
    .expect("prior sampling must not be charged to this dispatch");
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
    let parsed: serde_json::Value = serde_json::from_str(r#"{"value":91.01400000000001}"#).unwrap();
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
fn required_covering_set_is_exactly_sixteen_coordinates() {
    let coordinates = required_coordinates();
    assert_eq!(coordinates.len(), 16);
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
            "llama8b-short-single-chunked-cold",
            "llama8b-medium-supported-batch-single-shot-warm",
            "llama8b-memory-material-single-single-shot-warm",
            "llama8b-fit-boundary-single-chunked-cold",
            "qwen8b-short-single-single-shot-cold",
            "qwen8b-medium-supported-batch-chunked-warm",
            "qwen8b-memory-material-single-single-shot-warm",
            "qwen8b-fit-boundary-single-chunked-cold",
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
        (0..16).collect::<Vec<_>>()
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

/// sc-20688: the complete-campaign loader still reads a published schedule-v2 campaign (the
/// eight `llama`/`qwen` rows, SC-20676's dense baseline) as well as v3, routes each to its own
/// schedule and resume-identity coordinates, and refuses any other schedule version or a v2
/// manifest bound to the v3 coordinate list.
#[test]
fn complete_campaign_loader_reads_schedule_v2_and_v3_by_their_own_schedule() {
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
    let slugs = |version: u64| {
        covering_schedule(version)
            .unwrap()
            .iter()
            .map(|row| coordinate_slug(&row.coordinate))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        slugs(2),
        required_coordinates()[..8]
            .iter()
            .map(coordinate_slug)
            .collect::<Vec<_>>()
    );
    assert_eq!(slugs(3).len(), 16);
    assert!(covering_schedule(1).is_none() && covering_schedule(4).is_none());
    let publish_as = |schedule_version: u64, identity_version: u64, coordinates: Vec<String>| {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path();
        let identity = serde_json::json!({
            "schemaVersion": 1, "kind": "sc-20671-resume-identity",
            "scheduleVersion": identity_version, "coordinates": coordinates,
            "policySha256": policy.seal().unwrap(),
        });
        let (identity_bytes, identity_sha) = seal_json(&identity).unwrap();
        fs::write(dir.join("resume-identity.json"), &identity_bytes).unwrap();
        fs::write(
            dir.join("resume-identity.json.sha256"),
            format!("{identity_sha}  resume-identity.json\n"),
        )
        .unwrap();
        let (policy_bytes, _) = seal_json(&serde_json::to_value(&policy).unwrap()).unwrap();
        fs::write(dir.join("safety-policy.json"), &policy_bytes).unwrap();
        let manifest = canonical_json_bytes(&serde_json::json!({
            "schemaVersion": 2, "kind": SC20671_CAMPAIGN_KIND,
            "scheduleVersion": schedule_version,
            "policySha256": policy.seal().unwrap(),
            "resumeIdentitySha256": identity_sha,
            "coordinates": [],
        }))
        .unwrap();
        fs::write(dir.join("campaign.json"), &manifest).unwrap();
        fs::write(
            dir.join("campaign.json.sha256"),
            format!("{}  campaign.json\n", seal_bytes(&manifest)),
        )
        .unwrap();
        load_validated_complete_campaign(dir).unwrap_err()
    };
    let publish =
        |version: u64, coordinates: Vec<String>| publish_as(version, version, coordinates);
    // Both versions pass the identity gate and reach their own (here empty) row set.
    for version in [2, 3] {
        assert_eq!(
            publish(version, slugs(version)),
            "SC-20671 campaign manifest is not a complete scheduled set",
            "schedule v{version}"
        );
    }
    assert_eq!(
        publish(2, slugs(3)),
        "SC-20671 published resume identity semantics drifted"
    );
    // A v2 manifest whose resume identity claims another schedule version is refused.
    assert_eq!(
        publish_as(2, 3, slugs(2)),
        "SC-20671 published resume identity semantics drifted"
    );
    assert_eq!(
        publish(4, slugs(3)),
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
        "complete campaign requires exactly 16 prepared receipts"
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
        "schemaVersion": 1, "kind": "sc-20671-resume-identity", "scheduleVersion": 3,
        "coordinates": required_coordinates().iter().map(coordinate_slug).collect::<Vec<_>>(),
        "inferenceRevision": "a".repeat(40), "sceneWorksRevision": "b".repeat(40),
        "executableSha256": "c".repeat(64), "promptSha256": "d".repeat(64),
        "policySha256": policy.seal().unwrap(),
        "llamaCandidate": {"sha256": "e".repeat(64), "bytes": 101},
        "qwenCandidate": {"sha256": "f".repeat(64), "bytes": 102},
        "llamaReference": {"sha256": "1".repeat(64), "bytes": 103},
        "qwenReference": {"sha256": "2".repeat(64), "bytes": 104},
        "llama8bCandidate": {"sha256": "3".repeat(64), "bytes": 105},
        "qwen8bCandidate": {"sha256": "4".repeat(64), "bytes": 106},
        "llama8bReference": {"sha256": "5".repeat(64), "bytes": 107},
        "qwen8bReference": {"sha256": "6".repeat(64), "bytes": 108},
    });
    assert_eq!(
        seal_json(&identity).unwrap().1,
        "501113212265652a9652fd27c48962708d15d8f705cd46f6e9a5458b182518d1"
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
    for coordinate in required_coordinates()
        .into_iter()
        .filter(|coordinate| ["memory-material", "fit-boundary"].contains(&coordinate.context_band))
    {
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
        fn child_footprint_bytes(&mut self, _: u32, _: std::time::Instant) -> std::io::Result<u64> {
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
    let tripped: serde_json::Value = serde_json::from_slice(&fs::read(tripped).unwrap()).unwrap();
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

pub(crate) fn write_stub_dtype_snapshot(root: &Path, spec: &BenchmarkModelSpec, scale_dtype: &str) {
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
    let refusal =
        static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy, false)
            .unwrap_err();
    assert!(refusal.contains("child footprint cap 2048 refuses before spawn"));
    policy.child_footprint_cap_bytes = u64::MAX - policy.host_free_reserve_bytes;
    let budget =
        static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy, false).unwrap();
    assert!(budget > 12_000_000_000); // BF16 reference dominates the sequential roles.
}

/// sc-20688: a config without `head_dim` (Llama 3.1) is priced at the product loader's
/// default, hidden_size / num_attention_heads, instead of refusing the row; an indivisible
/// pair is refused.
#[test]
fn static_floor_defaults_a_missing_head_dim_like_the_loader() {
    let temporary = tempfile::tempdir().unwrap();
    let snapshot = temporary.path().join("candidate");
    write_stub_dtype_snapshot(&snapshot, &LLAMA_CANDIDATE, "F16");
    let explicit = static_role_footprint_budget(&LLAMA_CANDIDATE, &snapshot, 425, 318).unwrap();
    let config = |text: &str| fs::write(snapshot.join("config.json"), text).unwrap();
    config(
        r#"{"torch_dtype":"bfloat16","num_hidden_layers":28,"num_key_value_heads":8,"num_attention_heads":24,"hidden_size":3072,"intermediate_size":8192,"vocab_size":128256}"#,
    );
    assert_eq!(
        static_role_footprint_budget(&LLAMA_CANDIDATE, &snapshot, 425, 318).unwrap(),
        explicit
    );
    config(
        r#"{"torch_dtype":"bfloat16","num_hidden_layers":28,"num_key_value_heads":8,"num_attention_heads":25,"hidden_size":3072,"intermediate_size":8192,"vocab_size":128256}"#,
    );
    assert!(static_role_footprint_budget(&LLAMA_CANDIDATE, &snapshot, 425, 318).is_err());
}

/// sc-20688: the scaled compressed estimate covers the measured A2 v5 peaks of the 3B and 1.7B
/// fit rows (run 37004025116: 56,136,371,440 B and 17,960,978,184 B) at their recorded live and
/// request tokens, so admission never prices a compressed row below what it was seen to hold.
#[test]
fn compressed_estimate_covers_the_measured_fit_row_peaks() {
    let temporary = tempfile::tempdir().unwrap();
    let policy = CampaignSafetyPolicy {
        schema_version: 1,
        row_deadline_seconds: 10,
        poll_millis: 100,
        term_grace_millis: 500,
        host_free_reserve_bytes: 1,
        child_footprint_cap_bytes: u64::MAX - 1,
        max_context_tokens: 1 << 20,
        max_request_tokens: 1 << 20,
        stdout_cap_bytes: 4_096,
        stderr_cap_bytes: 4_096,
    };
    let qwen_config = br#"{"torch_dtype":"bfloat16","num_hidden_layers":28,"num_key_value_heads":8,"head_dim":128,"num_attention_heads":16,"hidden_size":2048,"intermediate_size":6144,"vocab_size":151936}"#;
    for (family, candidate_spec, reference_spec, scale, config, live, request, peak) in [
        (
            "llama",
            &LLAMA_CANDIDATE,
            &LLAMA_REFERENCE,
            "F16",
            None,
            261_483,
            130_846,
            56_136_371_440_u64,
        ),
        (
            "qwen",
            &QWEN_CANDIDATE,
            &QWEN_REFERENCE,
            "BF16",
            Some(&qwen_config[..]),
            81_231,
            40_700,
            17_960_978_184,
        ),
    ] {
        let candidate = temporary.path().join(format!("{family}-candidate"));
        let reference = temporary.path().join(format!("{family}-reference"));
        write_stub_dtype_snapshot(&candidate, candidate_spec, scale);
        write_stub_dtype_snapshot(&reference, reference_spec, "BF16");
        if let Some(config) = config {
            fs::write(candidate.join("config.json"), config).unwrap();
            fs::write(reference.join("config.json"), config).unwrap();
        }
        let row = required_coordinates()
            .into_iter()
            .find(|row| row.family == family && row.context_band == "fit-boundary")
            .unwrap();
        let unscaled =
            static_role_footprint_budget(candidate_spec, &candidate, live, request).unwrap();
        assert!(
            unscaled < peak,
            "{family}: the unscaled estimate is the under-price"
        );
        let priced =
            static_row_footprint_budget(&row, &candidate, &reference, live, request, &policy, true)
                .unwrap();
        assert!(priced >= peak, "{family}: {priced} < measured {peak}");
    }
}

/// sc-20688: a compressed row loads only the candidate (its quality reference is the dense-KV
/// arm on the same weights), so its floor is the candidate's alone; a dense row still prices
/// the larger bf16 reference it loads.
#[test]
fn compressed_rows_price_only_the_candidate_they_load() {
    let temporary = tempfile::tempdir().unwrap();
    let candidate = temporary.path().join("candidate");
    let reference = temporary.path().join("reference");
    write_stub_dtype_snapshot(&candidate, &LLAMA_CANDIDATE, "F16");
    write_stub_dtype_snapshot(&reference, &LLAMA_REFERENCE, "BF16");
    let policy = CampaignSafetyPolicy {
        schema_version: 1,
        row_deadline_seconds: 10,
        poll_millis: 100,
        term_grace_millis: 500,
        host_free_reserve_bytes: 1,
        child_footprint_cap_bytes: u64::MAX - 1,
        max_context_tokens: 4_096,
        max_request_tokens: 4_096,
        stdout_cap_bytes: 4_096,
        stderr_cap_bytes: 4_096,
    };
    let row = &required_coordinates()[0];
    let candidate_only =
        static_role_footprint_budget(&LLAMA_CANDIDATE, &candidate, 425, 318).unwrap();
    let reference_only =
        static_role_footprint_budget(&LLAMA_REFERENCE, &reference, 425, 318).unwrap();
    assert!(reference_only > candidate_only);
    assert_eq!(
        static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy, true).unwrap(),
        (candidate_only * COMPRESSED_MEASURED_PEAK_SCALE_BPS).div_ceil(10_000)
    );
    assert_eq!(
        static_row_footprint_budget(row, &candidate, &reference, 425, 318, &policy, false).unwrap(),
        reference_only
    );
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
    let floor = static_role_footprint_budget(&LLAMA_REFERENCE, &reference, total, request).unwrap();
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
        fit_window_tokens: None,
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
    assert_eq!(schedule.len(), 16);
    assert_eq!(
        schedule
            .iter()
            .filter(|row| row.discipline == ProcessDiscipline::FreshChild)
            .count(),
        8
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
    assert_eq!(seen.len(), 16);
    assert_eq!(outcomes.len(), 16);
    assert_eq!(
        seen.iter()
            .filter(|(_, discipline)| *discipline == ProcessDiscipline::ReusedWarmWorker)
            .count(),
        8
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
            chat_template_date_string: Some(CAMPAIGN_CHAT_TEMPLATE_DATE_STRING.into()),
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
    let path =
        |kernel: &str, selection: &str, calls: usize| crate::primitives::PackedKernelPathEvidence {
            kernel: kernel.into(),
            selection: selection.into(),
            reason: "test".into(),
            query_dtype: "bfloat16".into(),
            calls: calls as u64,
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
    quality.greedy_token_agreement_by_repeat = vec![continuation.agreement; QUALITY_MEASUREMENTS];
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
        fit_window_tokens: None,
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
        let mut value: serde_json::Value = serde_json::from_slice(pairs[cache_index].1).unwrap();
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
        &|v| v["evidence"]["turns"]["candidate"]["turn2"]["cacheHit"] = serde_json::json!(false),
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
    assert!(human.contains("needleRetrieval repeat 0 = 0 (minimum 1, fixture long-context-needle)"));

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
        error.contains("kernel parity failed: metric=parityMaxError value=0.5 threshold=0.0001"),
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
    assert!(
        compressed_receipt_block(CompressedKvMethod::GroupAffine, &dense_only, &[&dense_only])
            .unwrap_err()
            .contains("never executed the fused")
    );
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
        compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary]).unwrap();
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
        compressed_receipt_block(CompressedKvMethod::GroupAffine, &primary, &[&primary]).unwrap();
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
        validate_repeat_discrimination(&all_discriminating, [(fixture, measured_miss.as_slice())])
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
    event_outside_phase.memory.allocation_events[1].timestamp = "2026-01-01T00:00:03.100Z".into();
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
    non_material.geometry.context_payload_tokens = non_material.geometry.context_target_tokens / 2;
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
        context_band_target(not_near_fit.geometry.context_window_tokens, "fit-boundary").unwrap();
    not_near_fit.geometry.context_payload_tokens = not_near_fit.geometry.context_target_tokens / 2;
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
    late_prefill_window.memory.prefill_peak_window.started_at = "2026-01-01T00:00:02.500Z".into();
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
            event.phase == "prefill-peak" && event.role == "cache" && event.lifetime == "persistent"
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
        quiesced_within_slack(expected, POST_RELEASE_MLX_SLACK_FLOOR_BYTES + 1, expected).is_err()
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
    let warmup_seal = warmup_probe_suite_sha256(&"7".repeat(64), 42, &warm.probe_evidence).unwrap();
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
    // Parse a controlled lock entry so an ordinary dependency update does not
    // turn this parser regression into a repository-pin currency requirement.
    let lock = br#"[[package]]
name = "pmetal-mlx-rs"
version = "0.25.8"
source = "git+https://github.com/michaeltrefry/mlx-rs?rev=355e2ae4e81c13032233a4244e956bc3547f49e4#355e2ae4e81c13032233a4244e956bc3547f49e4"
"#;
    let identity = locked_mlx_identity(lock).unwrap();
    assert_eq!(identity.version, "0.25.8");
    assert_eq!(
        identity.revision,
        "355e2ae4e81c13032233a4244e956bc3547f49e4"
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

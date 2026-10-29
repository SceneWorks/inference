use super::*;

/// The receipt quality of the row's quality measurement (contract v5: exactly one per arm, since
/// every fixture is deterministic in-process); its fixture artifacts record its own values.
pub(super) fn receipt_quality_over_repeats(
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
pub(super) fn compressed_arm_observations<'a>(
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
pub(super) fn product_receipt(
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
    let mlx = locked_mlx_identity(include_bytes!("../../../../../Cargo.lock"))?;
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
            dependency_lock_sha256: seal_bytes(include_bytes!("../../../../../Cargo.lock")),
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
            chat_template_date_string: Some(CAMPAIGN_CHAT_TEMPLATE_DATE_STRING.into()),
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

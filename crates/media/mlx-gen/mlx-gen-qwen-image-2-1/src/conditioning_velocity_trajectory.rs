//! Additive ignored current-donor trajectory observer. No production changes or acceptance.
use super::*;
use mlx_gen::gen_core::sampling::TimestepConvention;
use mlx_gen::{run_flow_sampler_with_latent_hook, CancelFlag};
#[path = "conditioning_trajectory_math.rs"]
mod trajectory_math;

const ENDPOINTS: [&str; 4] = [
    "b0eda194bc37877a9ee186d51a7945d388ff331628f2c11b9f5cdb35a4645ef4",
    "5a2a5ac175fa9ba63549b34bd40e2c8e7571409ad5efae0d2e5590643654136b",
    "34617ede5c4635bf40cb59e441b11075381b42b8459c888c42b771d453d37e14",
    "c15998cea7be72d8685ffce248ad3c34483512883869c0833e425b0e7189f6d5",
];

#[test]
#[ignore = "four fixed current-donor trajectories/32 forwards/four endpoint identity decodes; never acceptance"]
fn diagnostic_current_failed_adapter_actual_dense_q4_trajectory() {
    run().unwrap();
}

fn read_vector(out: &Path, row: &Value, cache: &mut math::Cache) -> Result<Tensor> {
    let bytes = row["bytes"].as_u64().ok_or("vector byte count missing")?;
    assert_eq!(bytes, 147456 * 4);
    cache.reserve(bytes * 2)?;
    let file = row["file"].as_str().ok_or("vector file missing")?;
    assert!(!Path::new(file).is_absolute() && !file.split('/').any(|p| p == ".."));
    let path = out.join(file);
    assert_eq!(evidence::sha256_file(&path), row["sha256"]);
    let raw = std::fs::read(path)?;
    assert_eq!(raw.len() as u64, bytes);
    let values = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert!(values.iter().all(|v| v.is_finite()));
    drop(raw);
    cache.release(bytes);
    Ok(Tensor {
        shape: vec![1, 2304, 64],
        values,
        facts: row.clone(),
    })
}

fn record_tensor(array: &Array, name: &str, out: &Path, cache: &mut math::Cache) -> Result<Value> {
    assert_eq!(array.shape(), [1, 2304, 64]);
    let copied = copy_tensor(array, name, out, cache)?;
    let facts = copied.facts.clone();
    let bytes = copied.bytes();
    drop(copied);
    cache.release(bytes);
    Ok(facts)
}

fn run() -> Result<()> {
    assert_eq!(
        std::env::var("QWEN_IMAGE_2_1_LORA_PHASE").as_deref(),
        Ok("current-trajectory")
    );
    let binding = Binding::CurrentFailedNative768;
    let prepared_binding = prepare_binding(binding); // tensor-free immutable current bytes first
    let out = evidence::out_dir().join("current-q4-trajectory");
    assert!(!out.exists(), "fresh trajectory output required");
    std::fs::create_dir_all(&out)?;
    let src = evidence::edit_source(99, 768);
    let target = evidence::edit_transform(&src);
    let key = evidence::edit_key(512);
    for (name, img, expected) in [
        ("source99", &src, binding.reference_hashes()[0]),
        ("expected", &target, binding.target_hash()),
        ("palette", &key, binding.reference_hashes()[1]),
    ] {
        let path = out.join(format!("{name}.png"));
        img.save(&path).map_err(|e| e.to_string())?;
        assert_eq!(evidence::sha256_file(&path), expected);
    }
    let _owner = crate::q4_diagnostic::production_owner_scope();
    assert!(mlx_rs::task_local_default_stream().is_none());
    assert!(mlx_rs::Stream::new() == mlx_rs::Stream::gpu());
    let source = &prepared_binding.source_candidate;
    let manifest_path = &prepared_binding.manifest_path;
    let manifest = &prepared_binding.manifest;
    let donor = &prepared_binding.donor;
    let (donor_sha, _) = binding.adapter_identity();
    let (training_sha, _) = binding.training_identity();
    let dense_spec = evidence::tier_spec("bf16", None);
    let q4_spec = evidence::tier_spec("q4", Some(Quant::Q4));
    let dense = crate::loader::snapshot_root(&dense_spec.weights)?;
    let q4 = crate::loader::snapshot_root(&q4_spec.weights)?;
    let base_closure = seal_closure(binding, dense, q4, donor, &out);
    // The complete prior dense overlay/construction/VAE closure remains reserved.
    // Add the production request transient (denoise, reference encode and bounded decode),
    // sampler/copy workspace and fixed CPU scratch before any native construction.
    let request_transient = crate::memory_strategy::derived::request_transient_budget_bytes(
        768,
        768,
        crate::memory_strategy::derived::TABLE_CONDITIONING_TOKENS,
        2,
        false,
        None,
    );
    let sampler_workspace = 64 * 147456 * 4;
    let active_limit = base_closure
        .checked_add(request_transient)
        .and_then(|n| n.checked_add(sampler_workspace))
        .and_then(|n| n.checked_add(trajectory_math::CPU_SCRATCH))
        .ok_or("trajectory envelope overflow")?;
    evidence::write_json(
        &out,
        "trajectory-phase-closure",
        &json!({
        "sealedBeforeNativeConstruction":true,"priorFullClosureBytes":base_closure,
        "productionRequestTransientBytes":request_transient,"samplerWorkspaceBytes":sampler_workspace,
        "cpuScratchBytes":trajectory_math::CPU_SCRATCH,"activeEnvelopeBytes":active_limit,
        "frozenFreeCacheAllowanceBytes":math::FREE_CACHE,"endpointDecodeCount":4,
        "scope":"prior full closure plus entire production request transient and bounded observer workspace"}),
    );
    let previous_limits = crate::memory_strategy::AllocatorBounds::current();
    let guard = evidence::Footprint::start(&out);
    let cache_grant = guard.admit_numeric_scoped(active_limit, math::FREE_CACHE);
    guard.begin();
    let baseline_active = mlx_rs::memory::get_active_memory() as u64;
    let bounds = crate::memory_strategy::AllocatorBounds::enter(
        active_limit - math::FREE_CACHE,
        math::FREE_CACHE,
    );
    let retire = RetireOnDrop; // native locals retire before bounds/cache grant/watchdog on all exits
    let mut cache = math::Cache::default();
    cache.reserve(2 * 1024 * 1024 + trajectory_math::CPU_SCRATCH)?;
    let mut trace = Vec::new();
    let mut receipt = json!({"kind":"DIAGNOSTIC_ONLY","acceptanceEvidence":false,"accepted":false,
        "qualityAcceptance":null,"donorAccepted":false,"status":"RUNNING",
        "sourceCandidate":source,"sourceBase":binding.source_base(),"sceneWorksFbdCommit":manifest["sceneWorksFbdCommit"],
        "runId":std::env::var("GITHUB_RUN_ID").unwrap().parse::<u64>().unwrap(),"runAttempt":std::env::var("GITHUB_RUN_ATTEMPT").unwrap().parse::<u64>().unwrap(),
        "inputManifestSha256":evidence::sha256_file(manifest_path),"trainingProvenance":manifest["trainingProvenance"],
        "adapterSha256":donor_sha,"trainingReceiptSha256":training_sha,
        "protocolReceiptSha256":binding.protocol_identity(),"trainingSteps":0,"renderCount":0,"endpointDecodeCount":0,
        "forwardCount":0,"trajectoryCount":4,"steps":8,"seed":24163,"guidance":1,"negativeBranch":false,
        "sampler":"Euler","arithmeticBoundVerdict":math::ARITHMETIC,"trajectories":[],
        "limitations":["autonomous inputs differ after step zero; no later same-input adapter sign attribution",
        "CPU f64 metrics are observations, not a native arithmetic bound or image acceptance",
        "CPU capture adds x/v materialization before Euler update and can alter lazy fusion; endpoint hashes qualify identity"],
        "historicalIdentityScope":"four endpoint PNG hashes only; historical intermediate tensors were not exported",
        "extraMaterialization":{"stepInputCpuCopies":32,"stepVelocityCpuCopiesBeforeEulerUpdate":32,"finalLatentCpuCopies":4,"productionSolverUnchanged":true}});
    evidence::write_json(&out, "receipt", &receipt);
    let req = GenerationRequest {
        prompt: binding.caption().to_owned(),
        width: 768,
        height: 768,
        steps: Some(8),
        seed: Some(24163),
        ..Default::default()
    };
    let scheduler = crate::loader::load_scheduler_config(dense)?;
    let params = crate::model::resolve_run_params(&scheduler, &req)?;
    assert!(!params.use_negative && params.true_cfg == 1.0);
    trajectory_math::validate_schedule(&params.sigmas)?;
    let vision = crate::loader::load_vision_config(dense)?.expect("fixed vision required");
    let image_bytes = (src.as_raw().len() + target.as_raw().len() + key.as_raw().len()) as u64;
    cache.reserve(image_bytes)?;
    let vae = crate::loader::load_vae(dense)?;
    let x0_array = crate::training::diagnostic_encode_latents(&vae, &evidence::to_image(target))?;
    let x0 = copy_tensor(&x0_array, "cpu-tensors/x0", &out, &mut cache)?;
    assert_eq!(x0.shape, binding.shape());
    drop(x0_array);
    let mut references = Vec::new();
    let mut reference_hashes = Vec::new();
    for (index, img) in [src, key].into_iter().enumerate() {
        let source_path = out.join(format!("prepared-source-{index}.png"));
        img.save(&source_path).unwrap();
        reference_hashes.push(evidence::sha256_file(&source_path));
        assert_eq!(reference_hashes[index], binding.reference_hashes()[index]);
        let rgb = evidence::to_image(img);
        let rgba = mlx_gen::RgbaImage {
            width: rgb.width,
            height: rgb.height,
            pixels: rgb
                .pixels
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255])
                .collect(),
        };
        let mut prepared = crate::reference::prepare_references(&[rgba], &vision)?;
        let reference = prepared.remove(0);
        assert_eq!(reference.size, (1024, 1024));
        let latents_array =
            crate::pipeline::encode_references(&vae, std::slice::from_ref(&reference))?.remove(0);
        let latents = copy_tensor(
            &latents_array,
            &format!("cpu-tensors/ref-{index}-latents"),
            &out,
            &mut cache,
        )?;
        assert_eq!(latents.shape, [1, 4096, 64]);
        drop(latents_array);
        let pixels = copy_tensor(
            &reference.pixel_values,
            &format!("cpu-tensors/ref-{index}-pixels"),
            &out,
            &mut cache,
        )?;
        references.push(Reference {
            pixels,
            grid: reference.grid_thw,
            size: reference.size,
            latents,
        });
        drop(reference);
        drop(prepared);
        crate::q4_diagnostic::retirement_boundary();
        phase_trace(
            &out,
            &format!("vae-reference-{index}"),
            active_limit,
            &cache,
            &mut trace,
        );
    }
    binding.validate_reference_order(&reference_hashes);
    assert_eq!(references.len(), 2);
    drop(vae);
    cache.release(image_bytes);
    crate::q4_diagnostic::retirement_boundary();
    assert_retired(baseline_active);
    phase_trace(&out, "vae-retired", active_limit, &cache, &mut trace);
    let noise_array = crate::pipeline::create_noise(24163, 768, 768, 64)?;
    let noise = copy_tensor(&noise_array, "cpu-tensors/noise", &out, &mut cache)?;
    assert_eq!(noise.shape, binding.shape());
    drop(noise_array);
    crate::q4_diagnostic::retirement_boundary();
    let tokenizer = crate::loader::load_tokenizer(dense)?;
    let drop_count = crate::text_encoder::system_prompt_drop_count(&tokenizer)?;
    let mut conditioning = Vec::new();
    let mut layouts = Vec::new();
    let mut masks = Vec::new();
    for (index, root) in [dense, q4].into_iter().enumerate() {
        let te = crate::loader::diagnostic_language_with_fixed_vision(root, dense)?;
        let prepared = references
            .iter()
            .map(Reference::prepared)
            .collect::<Vec<_>>();
        let encoded =
            te.encode_conditioning(&tokenizer, binding.caption(), drop_count, &prepared)?;
        let branch = crate::pipeline::joint_branch(&encoded, &prepared, 768, 768)?;
        assert_eq!(branch.text.dtype(), Dtype::Float32);
        conditioning.push(copy_tensor(
            &branch.text,
            &format!("cpu-tensors/conditioning-{index}"),
            &out,
            &mut cache,
        )?);
        layouts.push(branch.layout.clone());
        masks.push(encoded.image_pad_mask.clone());
        drop(branch);
        drop(encoded);
        drop(prepared);
        drop(te);
        crate::q4_diagnostic::retirement_boundary();
        assert_retired(baseline_active);
        phase_trace(
            &out,
            &format!("conditioning-{index}-retired"),
            active_limit,
            &cache,
            &mut trace,
        );
    }
    assert_eq!(layouts[0], layouts[1]);
    assert_eq!(masks[0], masks[1]);
    assert_eq!(conditioning[0].shape, conditioning[1].shape);
    let (conditioning_mae, conditioning_rms, conditioning_max) =
        math::difference(&conditioning[0].values, &conditioning[1].values)?;
    let positions = layouts[0].position_ids();
    receipt["inputs"] = json!({"caption":binding.caption(),"referenceOrder":["source99","palette"],"fit":[[1024,1024],[1024,1024]],
        "x0":x0.facts,"noise":noise.facts,"sigmas":params.sigmas,
        "references":references.iter().map(|r|json!({"pixels":r.pixels.facts,"latents":r.latents.facts,"grid":r.grid,"size":r.size})).collect::<Vec<_>>(),
        "conditioning":conditioning.iter().map(|c|&c.facts).collect::<Vec<_>>(),"imagePadMask":masks[0],
        "layout":format!("{:?}",layouts[0]),"positionIdsSha256":sha(&serde_json::to_vec(&positions).unwrap()),
        "fixedVisionSource":dense,"fixedVAESource":dense,"conditioningRoots":[dense,q4]});
    {
        receipt["inputs"]["conditioningDenseVsQ4"] = json!({
            "meanAbsoluteDifference":conditioning_mae,
            "rootMeanSquareDifference":conditioning_rms,
            "maxAbsoluteDifference":conditioning_max
        });
    }
    for r in &mut references {
        cache.release(r.pixels.bytes());
        r.pixels.values.clear();
        r.pixels.values.shrink_to_fit();
    }

    let mut trajectories = Vec::new();
    let mut forward_count = 0usize;
    let cancel = CancelFlag::new();
    for (trajectory, (tier, adapted)) in [(0, false), (0, true), (1, false), (1, true)]
        .into_iter()
        .enumerate()
    {
        let root = if tier == 0 { dense } else { q4 };
        let mut model = crate::loader::load_transformer(root)?;
        assert_eq!(
            model.compute_dtype(),
            if tier == 0 {
                Dtype::Bfloat16
            } else {
                Dtype::Float32
            }
        );
        for path in model.adaptable_paths() {
            assert!(model
                .adaptable_mut(&path.split('.').collect::<Vec<_>>())
                .unwrap()
                .adapters()
                .is_empty());
        }
        if adapted {
            install_retired(&mut model, donor, donor_sha, &mut cache)?;
        }
        let text = conditioning[tier].array();
        let latent_refs = references
            .iter()
            .map(|r| r.latents.array())
            .collect::<Vec<_>>();
        let mut observed = Vec::new();
        let final_native = run_flow_sampler_with_latent_hook(
            None,
            TimestepConvention::Sigma,
            &params.sigmas,
            noise.array(),
            24163,
            &cancel,
            &mut |_| {},
            |_, _| {},
            |x, sigma| {
                let step = observed.len();
                assert!(step < trajectory_math::STEPS);
                assert_eq!(sigma.to_bits(), params.sigmas[step].to_bits());
                let images = crate::pipeline::joint_images(&latent_refs, x);
                let v = model.forward_joint(&text, &images, sigma, &layouts[tier])?;
                assert_eq!(v.dtype(), Dtype::Float32);
                let x_row = record_tensor(
                    x,
                    &format!("vectors/t{trajectory}-s{step}-x"),
                    &out,
                    &mut cache,
                )?;
                let v_row = record_tensor(
                    &v,
                    &format!("vectors/t{trajectory}-s{step}-v"),
                    &out,
                    &mut cache,
                )?;
                observed.push(json!({"step":step,"sigma":sigma,"nextSigma":params.sigmas[step+1],"x":x_row,"velocity":v_row}));
                forward_count += 1;
                phase_trace(
                    &out,
                    &format!("t{trajectory}-step{step}"),
                    active_limit,
                    &cache,
                    &mut trace,
                );
                Ok(v) // return the original velocity; shared solver alone integrates it
            },
        )?;
        assert_eq!(observed.len(), trajectory_math::STEPS);
        let final_row = record_tensor(
            &final_native,
            &format!("vectors/t{trajectory}-final"),
            &out,
            &mut cache,
        )?;
        drop(final_native);
        drop(text);
        drop(latent_refs);
        drop(model);
        crate::q4_diagnostic::retirement_boundary();
        assert_retired(baseline_active);
        let final_cpu = read_vector(&out, &final_row, &mut cache)?;
        let final_error = trajectory_math::mean_squared_difference(&final_cpu.values, &x0.values)?;
        let final_array = final_cpu.array();
        let vae = crate::loader::load_vae(dense)?;
        let tiling = crate::pipeline::decode_tiling(&req);
        let endpoint = crate::pipeline::decode_rgb(
            &vae,
            &final_array,
            768,
            768,
            tiling.as_ref(),
            Some(&cancel),
        )?;
        let endpoint_path = out.join(format!("endpoint-{trajectory}.png"));
        image::save_buffer(
            &endpoint_path,
            &endpoint.pixels,
            endpoint.width,
            endpoint.height,
            image::ColorType::Rgb8,
        )
        .map_err(|e| e.to_string())?;
        let endpoint_sha = evidence::sha256_file(&endpoint_path);
        let endpoint_matches = endpoint_sha == ENDPOINTS[trajectory];
        drop(endpoint);
        drop(vae);
        drop(final_array);
        let final_bytes = final_cpu.bytes();
        drop(final_cpu);
        cache.release(final_bytes);
        crate::q4_diagnostic::retirement_boundary();
        assert_retired(baseline_active);
        let mut metrics = Vec::new();
        for step in 0..trajectory_math::STEPS {
            let x = read_vector(&out, &observed[step]["x"], &mut cache)?;
            let v = read_vector(&out, &observed[step]["velocity"], &mut cache)?;
            let next_row = if step + 1 == trajectory_math::STEPS {
                &final_row
            } else {
                &observed[step + 1]["x"]
            };
            let next = read_vector(&out, next_row, &mut cache)?;
            let m = trajectory_math::step_metrics(
                &x.values,
                &v.values,
                &x0.values,
                &noise.values,
                params.sigmas[step],
                params.sigmas[step + 1],
            )?;
            let update_residual = trajectory_math::update_residual(
                &x.values,
                &v.values,
                &next.values,
                params.sigmas[step + 1] - params.sigmas[step],
            )?;
            metrics.push(json!({"step":step,"targetPathError":m.target_path_error,"denoisedEstimateError":m.estimate_error,
                "updateToTargetProjection":m.update_projection,"updateNorm2":m.update_norm2,"eulerUpdateResidualMax":update_residual,
                "nativeArithmeticPass":false}));
            for t in [x, v, next] {
                let bytes = t.bytes();
                drop(t);
                cache.release(bytes);
            }
        }
        trajectories.push(json!({"trajectory":trajectory,"tier":if tier==0 {"denseBF16"} else {"Q4"},"adapted":adapted,
            "rawMasterVerifiedTensors":if adapted {672} else {0},"steps":observed,"metrics":metrics,"finalLatent":final_row,
            "finalLatentTargetError":final_error,"endpoint":{"file":format!("endpoint-{trajectory}.png"),"sha256":endpoint_sha,
            "expectedSha256":ENDPOINTS[trajectory],"exactHistoricalPngMatch":endpoint_matches}}));
        receipt["trajectories"] = json!(trajectories);
        receipt["forwardCount"] = json!(forward_count);
        receipt["endpointDecodeCount"] = json!(trajectories.len());
        evidence::write_json(&out, "receipt", &receipt);
        phase_trace(
            &out,
            &format!("t{trajectory}-retired"),
            active_limit,
            &cache,
            &mut trace,
        );
    }
    let mut comparisons = Vec::new();
    for step in 0..=trajectory_math::STEPS {
        let mut values = Vec::new();
        for t in &trajectories {
            let row = if step == trajectory_math::STEPS {
                &t["finalLatent"]
            } else {
                &t["steps"][step]["x"]
            };
            values.push(read_vector(&out, row, &mut cache)?);
        }
        comparisons.push(json!({"step":step,
            "denseDenoisedEstimateErrorGain":if step<trajectory_math::STEPS {Some(trajectories[0]["metrics"][step]["denoisedEstimateError"].as_f64().unwrap()-trajectories[1]["metrics"][step]["denoisedEstimateError"].as_f64().unwrap())} else {None},
            "q4DenoisedEstimateErrorGain":if step<trajectory_math::STEPS {Some(trajectories[2]["metrics"][step]["denoisedEstimateError"].as_f64().unwrap()-trajectories[3]["metrics"][step]["denoisedEstimateError"].as_f64().unwrap())} else {None},
            "denseQ4BaseTrajectoryDriftMse":trajectory_math::mean_squared_difference(&values[0].values,&values[2].values)?,
            "denseQ4AdaptedTrajectoryDriftMse":trajectory_math::mean_squared_difference(&values[1].values,&values[3].values)?,
            "denseAdapterTrajectoryDifferenceMse":trajectory_math::mean_squared_difference(&values[0].values,&values[1].values)?,
            "q4AdapterTrajectoryDifferenceMse":trajectory_math::mean_squared_difference(&values[2].values,&values[3].values)?,
            "denseTargetErrorGain":trajectory_math::mean_squared_difference(&values[0].values,&x0.values)?-trajectory_math::mean_squared_difference(&values[1].values,&x0.values)?,
            "q4TargetErrorGain":trajectory_math::mean_squared_difference(&values[2].values,&x0.values)?-trajectory_math::mean_squared_difference(&values[3].values,&x0.values)?,
            "comparisonScope":"matched sigma, own autonomous inputs; not later same-input adapter gain"}));
        for t in values {
            let bytes = t.bytes();
            drop(t);
            cache.release(bytes);
        }
    }
    // CPU vectors are released before the retirement/bounds/cache restoration receipt.
    for t in conditioning {
        let bytes = t.bytes();
        drop(t);
        cache.release(bytes);
    }
    for r in references {
        cache.release(r.latents.bytes());
        drop(r);
    }
    cache.release(x0.bytes());
    drop(x0);
    cache.release(noise.bytes());
    drop(noise);
    drop(retire);
    assert_retired(baseline_active);
    let (peak, physical) = guard.end();
    assert!(peak <= active_limit);
    drop(bounds);
    drop(cache_grant);
    let restored_limits = crate::memory_strategy::AllocatorBounds::current();
    assert_eq!(
        restored_limits, previous_limits,
        "actual post-Drop allocator readback"
    );
    drop(guard); // watchdog joins before any complete receipt
    assert_eq!(
        forward_count,
        trajectory_math::STEPS * trajectory_math::TRAJECTORIES
    );
    assert_eq!(trajectories.len(), trajectory_math::TRAJECTORIES);
    let historical_identity = trajectories
        .iter()
        .all(|t| t["endpoint"]["exactHistoricalPngMatch"] == true);
    receipt["status"] = json!("DIAGNOSTIC_COMPLETED");
    receipt["comparisons"] = json!(comparisons);
    receipt["historicalEndpointIdentity"] = json!(historical_identity);
    receipt["trajectoryQualification"] = json!(if historical_identity {
        "EXACT_FOUR_PRIOR_PNG_ENDPOINTS"
    } else {
        "SOURCE_EQUIVALENT_RECONSTRUCTION_ONLY_ENDPOINT_MISMATCH"
    });
    receipt["activePeakBytes"] = json!(peak);
    receipt["physicalPeakBytes"] = json!(physical);
    receipt["activeEnvelopeBytes"] = json!(active_limit);
    receipt["cpuCachePeakBytes"] = json!(cache.peak);
    receipt["cleanup"] = json!({"nativeRetired":true,"watchdogJoined":true,"postDropReadback":true,
        "previousMemoryLimitBytes":previous_limits.0,"previousCacheLimitBytes":previous_limits.1,
        "restoredMemoryLimitBytes":restored_limits.0,"restoredCacheLimitBytes":restored_limits.1});
    receipt["renderCount"] = json!(4);
    evidence::write_json(&out, "receipt", &receipt);
    Ok(())
}

//! MOSS-only complete-state persistence. The generic legacy trainer checkpoint format is unchanged.
use super::*;
use candle_gen::train::policy::{self, JobTrainingPolicy, TrainingProfile};
use contract::recovery::{bundle, DataPlan, RunState};
use sha2::{Digest, Sha256};

const PRIVATE_PROFILE_KEY: &str = "_candle_execution_profile";
const PRIVATE_PROFILE_DOMAIN: &str = "moss-native-private-execution-v1";

// The exported identity is never mutated. Only the native snapshot's private
// effective controls and digest bind the execution profile. Current is exactly
// the historical representation, including its original digest and JSON bytes.
fn private_identity(
    identity: &contract::recovery::RunIdentity,
    profile: TrainingProfile,
) -> Result<contract::recovery::RunIdentity> {
    let binding = policy::bind_state_identity(PRIVATE_PROFILE_DOMAIN, profile);
    if binding == PRIVATE_PROFILE_DOMAIN {
        return Ok(identity.clone());
    }
    let mut controls: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&identity.effective_config)
            .map_err(|e| CandleError::Msg(format!("MOSS private resume controls: {e}")))?;
    if controls
        .insert(
            PRIVATE_PROFILE_KEY.into(),
            serde_json::Value::String(binding),
        )
        .is_some()
    {
        return Err(CandleError::Msg(
            "MOSS expected an unbound export identity".into(),
        ));
    }
    let mut private = identity.clone();
    private.effective_config =
        serde_json::to_string(&controls).map_err(|e| CandleError::Msg(e.to_string()))?;
    private.digest = format!(
        "{:x}",
        Sha256::digest(format!(
            "{PRIVATE_PROFILE_DOMAIN}:{}:{}",
            identity.digest, private.effective_config
        ))
    );
    Ok(private)
}

fn saved_profile(identity: &contract::recovery::RunIdentity) -> Result<TrainingProfile> {
    let controls: std::collections::BTreeMap<String, serde_json::Value> =
        serde_json::from_str(&identity.effective_config)
            .map_err(|e| CandleError::Msg(format!("MOSS private resume controls: {e}")))?;
    let Some(binding) = controls.get(PRIVATE_PROFILE_KEY) else {
        return Ok(TrainingProfile::Current);
    };
    let binding = binding
        .as_str()
        .ok_or_else(|| CandleError::Msg("MOSS malformed private profile".into()))?;
    if !binding.starts_with(&format!("{PRIVATE_PROFILE_DOMAIN};execution_profile=")) {
        return Err(CandleError::Msg(
            "MOSS unknown private profile domain".into(),
        ));
    }
    let profile = policy::profile_from_state_identity(binding)?;
    if profile == TrainingProfile::Current {
        return Err(CandleError::Msg(
            "MOSS noncanonical tagged Current state".into(),
        ));
    }
    Ok(profile)
}

pub(super) fn job_policy(
    model: &str,
    cuda: bool,
    req: &TrainingRequest,
) -> Result<JobTrainingPolicy> {
    let mut job = JobTrainingPolicy::resolve(model, cuda, req)?;
    // Only the measured fixed-square Base LoRA recipe may select a new arm.
    // This guard must precede finite resume-profile selection below.
    if model != "moss_image_base"
        || req.config.resolution != 256
        || !req.config.resolution_buckets.is_empty()
        || !req.config.aspect_ratio_buckets.is_empty()
        || req.config.full_finetune
        || req.config.network_type != gen_core::NetworkType::Lora
    {
        job = job.for_unmeasured_mode();
    }
    if req.config.resume {
        // The native reader verifies the complete seal/member hashes, not a bare
        // untrusted run.json or the unrelated generic resume filename format.
        let state = contract::recovery::resume_state(&req.output_dir, &req.file_name)?;
        job = job.with_native_resume_profile(saved_profile(&state.effective_identity)?)?;
    }
    Ok(job)
}

pub(super) fn validate_resume_plan(
    req: &TrainingRequest,
    source: &str,
    plan: &DataPlan,
) -> Result<()> {
    if !req.config.resume {
        return Ok(());
    }
    let identity = contract::recovery::effective_identity(req, source, plan)?;
    let stem = checkpoint::file_stem(&req.file_name);
    let state = bundle::read(&bundle::latest(&req.output_dir, stem)?)?;
    checked_state(&state, req, plan, &identity, policy::active_profile())?;
    Ok(())
}

fn state_for_profile(state: &RunState, profile: TrainingProfile) -> Result<RunState> {
    let mut private = state.clone();
    private.effective_identity = private_identity(&state.effective_identity, profile)?;
    Ok(private)
}

fn checked_state(
    state: &RunState,
    req: &TrainingRequest,
    plan: &DataPlan,
    export_identity: &contract::recovery::RunIdentity,
    profile: TrainingProfile,
) -> Result<RunState> {
    state.validate(req, plan, &private_identity(export_identity, profile)?)?;
    let mut restored = state.clone();
    restored.effective_identity = export_identity.clone();
    Ok(restored)
}

pub(super) fn save_complete(
    req: &TrainingRequest,
    stem: &str,
    state: &RunState,
    opt: &TrainOptimizer,
    set: &LoraSet,
    window: &GradientWindow,
) -> Result<()> {
    let private = state_for_profile(state, policy::active_profile())?;
    bundle::publish(&req.output_dir, stem, &private, |dir| {
        let mut tensors = HashMap::new();
        for (name, var) in set.named_vars() {
            tensors.insert(format!("factor.{name}"), var.as_tensor().clone());
            if state.accumulated_tokens > 0 {
                let gradient = window
                    .gradients
                    .as_ref()
                    .and_then(|g| g.get(var.as_tensor()))
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!(
                            "MOSS checkpoint missing pending gradient {name}"
                        ))
                    })?;
                crate::adapters::finite(gradient, "pending gradient")?;
                tensors.insert(format!("pending.{name}"), gradient.clone());
            }
        }
        safetensors::serialize_to_file(tensors.iter(), None, &dir.join("tensors.safetensors"))
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let (optimizer, metadata) = opt.snapshot()?;
        safetensors::serialize_to_file(
            optimizer.iter(),
            Some(metadata),
            &dir.join("optimizer.safetensors"),
        )
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        Ok(())
    })?;
    Ok(())
}

pub(super) fn load_complete(
    req: &TrainingRequest,
    stem: &str,
    plan: &DataPlan,
    identity: &contract::recovery::RunIdentity,
    opt: &mut TrainOptimizer,
    set: &LoraSet,
) -> Result<(RunState, GradientWindow)> {
    let path = bundle::latest(&req.output_dir, stem)?;
    let state = checked_state(
        &bundle::read(&path)?,
        req,
        plan,
        identity,
        policy::active_profile(),
    )?;
    // Cursor/RNG/data and all payload checks are unchanged. Return the original
    // identity view to the training loop; only persistence owns the private tag.
    let tensors =
        candle_gen::candle_core::safetensors::load(path.join("tensors.safetensors"), &Device::Cpu)?;
    let mut factors = HashMap::new();
    let mut pending = HashMap::new();
    for (name, value) in tensors {
        crate::adapters::finite(&value, "resume tensor")?;
        if value.dtype() != DType::F32 {
            return Err(CandleError::Msg(
                "MOSS raw resume tensors must be FP32".into(),
            ));
        }
        if let Some(name) = name.strip_prefix("factor.") {
            factors.insert(name.to_owned(), value);
        } else if let Some(name) = name.strip_prefix("pending.") {
            pending.insert(name.to_owned(), value);
        } else {
            return Err(CandleError::Msg(
                "MOSS resume contains an unknown tensor".into(),
            ));
        }
    }
    let named = set.named_vars();
    let expected = if state.accumulated_tokens > 0 {
        named.len()
    } else {
        0
    };
    if pending.len() != expected {
        return Err(CandleError::Msg(
            "MOSS pending gradient population differs".into(),
        ));
    }
    let mut gradients = GradStore::default();
    for (name, var) in &named {
        if state.accumulated_tokens > 0 {
            let value = pending
                .get(name)
                .ok_or_else(|| CandleError::Msg(format!("MOSS missing pending gradient {name}")))?;
            if value.dims() != var.dims() {
                return Err(CandleError::Msg(
                    "MOSS pending gradient shape differs".into(),
                ));
            }
            gradients.insert(var.as_tensor(), value.to_device(var.device())?);
        }
    }
    let optimizer_path = path.join("optimizer.safetensors");
    let bytes = std::fs::read(&optimizer_path).map_err(|e| CandleError::Msg(e.to_string()))?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(e.to_string()))?;
    let metadata = header
        .metadata()
        .clone()
        .ok_or_else(|| CandleError::Msg("MOSS missing optimizer metadata".into()))?;
    let optimizer = candle_gen::candle_core::safetensors::load(&optimizer_path, &Device::Cpu)?;
    for value in optimizer.values() {
        crate::adapters::finite(value, "optimizer state")?;
    }
    let mut expected_optimizer = std::collections::HashMap::<String, Vec<usize>>::new();
    match opt.kind_tag() {
        "adam" | "adamw" => {
            let saved = metadata
                .get("optim.step")
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or_else(|| {
                    CandleError::Msg("MOSS missing/malformed optimizer update cursor".into())
                })?;
            if saved != state.update {
                return Err(CandleError::Msg(
                    "MOSS optimizer update cursor differs".into(),
                ));
            }
            for (index, variable) in set.vars.iter().enumerate() {
                for suffix in ["m", "v"] {
                    expected_optimizer
                        .insert(format!("optim.{index}.{suffix}"), variable.dims().to_vec());
                }
            }
        }
        "prodigy" => {
            for key in ["optim.d", "optim.d_max", "optim.d_numerator"] {
                let value = metadata
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .ok_or_else(|| {
                        CandleError::Msg("MOSS missing Prodigy coupled scalar".into())
                    })?;
                if !value.is_finite() {
                    return Err(CandleError::Msg("MOSS nonfinite Prodigy scalar".into()));
                }
            }
            if state.update > 0 {
                for (index, variable) in set.vars.iter().enumerate() {
                    for suffix in ["exp_avg", "exp_avg_sq", "s", "p0"] {
                        expected_optimizer
                            .insert(format!("optim.{index}.{suffix}"), variable.dims().to_vec());
                    }
                }
            }
        }
        "rose" => (),
        _ => {
            return Err(CandleError::Msg(
                "MOSS unsupported optimizer state tag".into(),
            ))
        }
    }
    if optimizer.len() != expected_optimizer.len()
        || expected_optimizer.iter().any(|(key, shape)| {
            optimizer
                .get(key)
                .is_none_or(|t| t.dtype() != DType::F32 || t.dims() != shape.as_slice())
        })
    {
        return Err(CandleError::Msg(
            "MOSS incomplete/mismatched optimizer tensor surface".into(),
        ));
    }

    // Nothing is published to a live run before complete identity/tensor validation above.
    set.restore_named(&factors)?;
    opt.restore(&optimizer, &metadata)?;
    let window = GradientWindow {
        gradients: if state.accumulated_tokens > 0 {
            Some(gradients)
        } else {
            None
        },
        tokens: state.accumulated_tokens,
    };
    Ok((state, window))
}

#[cfg(test)]
mod native_profile_metadata_tests {
    use super::*;
    use candle_gen::gen_core::train::{TrainingConfig, TrainingItem};

    fn fixture() -> (
        TrainingRequest,
        DataPlan,
        contract::recovery::RunIdentity,
        RunState,
    ) {
        let req = TrainingRequest {
            items: vec![TrainingItem::captioned(
                "never-read.png".into(),
                "metadata".into(),
            )],
            config: TrainingConfig {
                steps: 4,
                resolution: 64,
                seed: 7,
                gradient_accumulation: 1,
                ..Default::default()
            },
            output_dir: "unused".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        };
        // Scalar geometry/schedule only: no image read, planner/device or tensors.
        let plan = DataPlan::new(&req, &[(64, 64)]).unwrap();
        let identity = contract::recovery::RunIdentity {
            digest: "a".repeat(64),
            source_checkpoint: "parent-unchanged".into(),
            dataset_fingerprint: "dataset-unchanged".into(),
            effective_config: r#"{"network_type":"lora","seed":7}"#.into(),
        };
        let state = RunState::after_step(&req, &plan, &identity, 2, 2, 0, 0.25).unwrap();
        (req, plan, identity, state)
    }

    #[test]
    fn current_bytes_and_export_data_rng_cursor_views_are_unchanged() {
        let (req, plan, export, state) = fixture();
        let original = serde_json::to_vec(&state).unwrap();
        assert_eq!(
            serde_json::to_vec(&state_for_profile(&state, TrainingProfile::Current).unwrap())
                .unwrap(),
            original
        );
        let export_bytes = serde_json::to_vec(&export).unwrap();
        let private = state_for_profile(&state, TrainingProfile::WithoutE3).unwrap();
        assert_eq!(
            saved_profile(&private.effective_identity).unwrap(),
            TrainingProfile::WithoutE3
        );
        assert_eq!(
            private.effective_identity.source_checkpoint,
            export.source_checkpoint
        );
        assert_eq!(
            private.effective_identity.dataset_fingerprint,
            export.dataset_fingerprint
        );
        assert_eq!(private.effective_identity.digest.len(), 64);
        assert!(private
            .effective_identity
            .digest
            .bytes()
            .all(|b| b.is_ascii_hexdigit()));
        assert_ne!(private.effective_identity.digest, export.digest);
        assert_eq!(serde_json::to_vec(&export).unwrap(), export_bytes);
        assert!(
            private.validate(&req, &plan, &export).is_err(),
            "old Current readers must reject nonCurrent trajectory"
        );
        assert_eq!(
            checked_state(&private, &req, &plan, &export, TrainingProfile::WithoutE3).unwrap(),
            state
        );
        assert!(checked_state(&private, &req, &plan, &export, TrainingProfile::WithoutE1).is_err());
        let mut wrong = export.clone();
        wrong.dataset_fingerprint = "changed".into();
        assert!(checked_state(&private, &req, &plan, &wrong, TrainingProfile::WithoutE3).is_err());
    }

    #[test]
    fn finite_native_profile_parser_rejects_experimental_and_malformed_domains() {
        let (_, _, export, _) = fixture();
        assert_eq!(saved_profile(&export).unwrap(), TrainingProfile::Current);
        let mut private = private_identity(&export, TrainingProfile::WithoutE3).unwrap();
        let mut controls: std::collections::BTreeMap<String, serde_json::Value> =
            serde_json::from_str(&private.effective_config).unwrap();
        for value in [
            "moss-native-private-execution-v1;execution_profile=comparison-proof-v1/plus-streaming",
            "moss-native-private-execution-v1;execution_profile=future/legacy",
            "unrelated-domain",
            "moss-native-private-execution-v1;execution_profile=measured-reference-v1/current",
        ] {
            controls.insert(PRIVATE_PROFILE_KEY.into(), value.into());
            private.effective_config = serde_json::to_string(&controls).unwrap();
            assert!(saved_profile(&private).is_err());
        }
    }

    fn publish_mock(root: &Path, state: &RunState) {
        // Sealed scalar fixture bytes are deliberately not tensor payloads; no
        // tensor loader is invoked by profile/header/preflight metadata tests.
        bundle::publish(root, "adapter", state, |dir| {
            std::fs::write(dir.join("tensors.safetensors"), b"mock-only").unwrap();
            std::fs::write(dir.join("optimizer.safetensors"), b"mock-only").unwrap();
            Ok(())
        })
        .unwrap();
    }

    #[test]
    fn native_owner_routing_restores_legacy_current_without_selecting_a_profile() {
        let (mut req, _, _, state) = fixture();
        let dir = tempfile::tempdir().unwrap();
        req.output_dir = dir.path().into();
        req.config.resume = true;
        publish_mock(dir.path(), &state);
        for id in [
            MossImageVariant::Base.provider_id(),
            MossImageVariant::Sft.provider_id(),
        ] {
            let job = job_policy(id, true, &req).unwrap();
            assert_eq!(job.model(), id);
            assert_eq!(job.profile(), TrainingProfile::Current);
        }
        let path = bundle::latest(dir.path(), "adapter").unwrap();
        std::fs::write(path.join("tensors.safetensors"), b"tampered").unwrap();
        assert!(job_policy(MossImageVariant::Base.provider_id(), true, &req).is_err());
    }

    #[test]
    fn pending_native_selector_refuses_unadopted_tagged_state_before_preparation() {
        let (mut req, _, _, state) = fixture();
        let dir = tempfile::tempdir().unwrap();
        req.output_dir = dir.path().into();
        req.config.resume = true;
        publish_mock(
            dir.path(),
            &state_for_profile(&state, TrainingProfile::WithoutE3).unwrap(),
        );
        assert!(job_policy(MossImageVariant::Base.provider_id(), true, &req).is_err());
        assert!(job_policy(MossImageVariant::Base.provider_id(), false, &req).is_err());
    }

    #[test]
    fn selected_base_256_scope_keeps_other_modes_resolutions_and_buckets_current() {
        let (mut req, _, _, _) = fixture();
        req.config.resolution = 256;
        let base = MossImageVariant::Base.provider_id();
        let job = job_policy(base, true, &req).unwrap();
        assert_eq!(job.model(), base);
        assert_eq!(job.profile(), TrainingProfile::WithoutE1);
        policy::with_job_policy(job, || {
            assert!(policy::legacy_e1());
            assert!(policy::cache_enabled() && policy::projection_enabled());
            policy::without_production_profile(|| {
                assert_eq!(policy::active_profile(), TrainingProfile::Current);
                assert!(!policy::legacy_e1());
            });
            assert_eq!(policy::active_profile(), TrainingProfile::WithoutE1);
        });
        for resolution in [0, 64, 255, 257, 512, 1024] {
            req.config.resolution = resolution;
            let outside = job_policy(base, true, &req).unwrap();
            assert_eq!(outside.profile(), TrainingProfile::Current);
            assert!(outside
                .with_native_resume_profile(TrainingProfile::WithoutE1)
                .is_err());
        }
        req.config.resolution = 256;
        assert_eq!(
            job_policy(base, false, &req).unwrap().profile(),
            TrainingProfile::Current
        );
        assert_eq!(
            job_policy(MossImageVariant::Sft.provider_id(), true, &req)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        req.config
            .resolution_buckets
            .push(gen_core::train::ResolutionBucket {
                resolution: 256,
                repeats: 1,
            });
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::Current
        );
        req.config.resolution_buckets.clear();
        req.config
            .aspect_ratio_buckets
            .push(gen_core::train::AspectRatioBucket {
                width: 256,
                height: 256,
                repeats: 1,
            });
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::Current
        );
        req.config.aspect_ratio_buckets.clear();
        req.config.full_finetune = true;
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::Current
        );
        req.config.full_finetune = false;
        req.config.network_type = gen_core::NetworkType::Lokr;
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::Current
        );
    }

    #[test]
    fn selected_256_native_tag_and_historical_current_keep_original_identity() {
        let (mut req, _, original_identity, _) = fixture();
        req.config.resolution = 256;
        let plan = DataPlan::new(&req, &[(256, 256)]).unwrap();
        let original =
            RunState::after_step(&req, &plan, &original_identity, 2, 2, 0, 0.25).unwrap();
        let tagged = state_for_profile(&original, TrainingProfile::WithoutE1).unwrap();
        let selected_dir = tempfile::tempdir().unwrap();
        publish_mock(selected_dir.path(), &tagged);
        req.output_dir = selected_dir.path().into();
        req.config.resume = true;
        let base = MossImageVariant::Base.provider_id();
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::WithoutE1
        );
        assert_eq!(
            checked_state(
                &tagged,
                &req,
                &plan,
                &original_identity,
                TrainingProfile::WithoutE1
            )
            .unwrap(),
            original
        );
        assert!(job_policy(base, false, &req).is_err());
        assert!(job_policy(MossImageVariant::Sft.provider_id(), true, &req).is_err());
        req.config.resolution = 512;
        assert!(job_policy(base, true, &req).is_err());
        req.config.resolution = 256;
        let old_dir = tempfile::tempdir().unwrap();
        publish_mock(old_dir.path(), &original);
        req.output_dir = old_dir.path().into();
        assert_eq!(
            job_policy(base, true, &req).unwrap().profile(),
            TrainingProfile::Current
        );
        assert_eq!(
            checked_state(
                &original,
                &req,
                &plan,
                &original_identity,
                TrainingProfile::Current
            )
            .unwrap(),
            original
        );
        assert_eq!(original.effective_identity, original_identity);
    }
}

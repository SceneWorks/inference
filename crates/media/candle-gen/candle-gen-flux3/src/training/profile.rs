//! Native private resume identity, separate from public adapter provenance.
use candle_gen::gen_core::train::TrainingRequest;
use candle_gen::train::policy::{self, JobTrainingPolicy, TrainingProfile};
use candle_gen::{CandleError as Error, Result};
use gen_core_flux3::training::{
    controller::CANDLE_EXECUTION_PROFILE_KEY, options::Workflow, ResumeIdentity,
};

const DOMAIN: &str = "flux3-native-private-execution-v1";

pub(super) fn job_policy(
    cuda: bool,
    workflow: Workflow,
    req: &TrainingRequest,
) -> Result<JobTrainingPolicy> {
    let job = JobTrainingPolicy::resolve(candle_gen::gen_core::flux3::BASE_MODEL_ID, cuda, req)?;
    Ok(for_native_mode(job, workflow, req))
}

fn for_native_mode(
    job: JobTrainingPolicy,
    workflow: Workflow,
    req: &TrainingRequest,
) -> JobTrainingPolicy {
    // The measured native Base fixture has one reference image and mode=edit.
    // Its result cannot select a profile for the distinct Captioned workflow.
    if workflow != Workflow::PairedEdit
        || req.config.full_finetune
        || req.config.network_type != candle_gen::gen_core::NetworkType::Lora
    {
        job.for_unmeasured_mode()
    } else {
        job
    }
}

pub(super) fn private_identity(
    export: &ResumeIdentity,
    profile: TrainingProfile,
) -> Result<ResumeIdentity> {
    if export
        .configuration
        .contains_key(CANDLE_EXECUTION_PROFILE_KEY)
    {
        return Err(Error::Msg(
            "flux3 expected an unchanged export identity".into(),
        ));
    }
    let mut private = export.clone();
    let binding = policy::bind_state_identity(DOMAIN, profile);
    if binding != DOMAIN {
        private
            .configuration
            .insert(CANDLE_EXECUTION_PROFILE_KEY.into(), binding);
    }
    Ok(private)
}

pub(super) fn saved_profile(binding: Option<&str>) -> Result<TrainingProfile> {
    let Some(binding) = binding else {
        return Ok(TrainingProfile::Current);
    };
    if !binding.starts_with(&format!("{DOMAIN};execution_profile=")) {
        return Err(Error::Msg(
            "flux3 unknown private execution profile domain".into(),
        ));
    }
    let profile = policy::profile_from_state_identity(binding)?;
    if profile == TrainingProfile::Current {
        return Err(Error::Msg("flux3 noncanonical tagged Current state".into()));
    }
    Ok(profile)
}

#[cfg(test)]
mod native_profile_metadata_tests {
    use super::*;
    use gen_core_flux3::training::{checkpoint::TrainingCursor, rng};
    use std::collections::BTreeMap;

    fn identity() -> ResumeIdentity {
        ResumeIdentity {
            base_repo: "metadata-only".into(),
            base_revision: "a".repeat(40),
            artifacts: BTreeMap::from([("source".into(), "b".repeat(64))]),
            dataset: "c".repeat(64),
            preprocessing: "d".repeat(64),
            configuration: [
                "optimizer",
                "learning_rate",
                "rank",
                "alpha",
                "scaling",
                "precision",
                "backend",
                "scheduler",
                "accumulation",
                "rng",
            ]
            .into_iter()
            .map(|k| (k.into(), "metadata".into()))
            .collect(),
            factors: BTreeMap::from([("factor".into(), vec![2, 3])]),
        }
    }
    fn request() -> TrainingRequest {
        TrainingRequest {
            items: vec![],
            config: Default::default(),
            output_dir: "unused".into(),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        }
    }

    #[test]
    fn current_and_tagged_private_views_never_mutate_export_base_dataset_or_rng() {
        let export = identity();
        let bytes = export.to_metadata().unwrap();
        assert_eq!(
            private_identity(&export, TrainingProfile::Current)
                .unwrap()
                .to_metadata()
                .unwrap(),
            bytes
        );
        let private = private_identity(&export, TrainingProfile::WithoutE3).unwrap();
        assert_eq!(
            saved_profile(
                private
                    .configuration
                    .get(CANDLE_EXECUTION_PROFILE_KEY)
                    .map(String::as_str)
            )
            .unwrap(),
            TrainingProfile::WithoutE3
        );
        let mut public_view = private.clone();
        public_view
            .configuration
            .remove(CANDLE_EXECUTION_PROFILE_KEY);
        assert_eq!(public_view.to_metadata().unwrap(), bytes);
        assert_eq!(export.to_metadata().unwrap(), bytes);
        assert!(
            export.ensure_matches(&private).is_err(),
            "old strict readers reject a different private trajectory"
        );
        let cursor = TrainingCursor {
            micro_step: 2,
            update_step: 2,
            accumulation_count: 0,
            sampler_epoch: 0,
            sampler_cursor: 2,
            rng_state: rng::state_record(7, 2),
            last_loss_bits: Some(0.25f32.to_bits()),
        };
        let encoded = cursor.to_metadata().unwrap();
        private_identity(&export, TrainingProfile::Legacy).unwrap();
        assert_eq!(cursor.to_metadata().unwrap(), encoded);
    }

    #[test]
    fn native_mode_and_profile_decode_remain_neutral_and_reject_proof_domains() {
        let req = request();
        let id = super::super::trainer::trainer_descriptor().id;
        assert_eq!(id, "flux3_klein_base");
        for workflow in [Workflow::Captioned, Workflow::PairedEdit] {
            for cuda in [true, false] {
                let job = job_policy(cuda, workflow, &req).unwrap();
                assert_eq!(job.model(), id);
                assert_eq!(job.profile(), TrainingProfile::Current);
                assert!(job
                    .with_native_resume_profile(TrainingProfile::WithoutE3)
                    .is_err());
            }
        }
        assert_eq!(saved_profile(None).unwrap(), TrainingProfile::Current);
        for binding in ["flux3-native-private-execution-v1;execution_profile=comparison-proof-v1/plus-streaming","flux3-native-private-execution-v1;execution_profile=future/minus-e3","wrong-domain","flux3-native-private-execution-v1;execution_profile=measured-reference-v1/current"] {
            assert!(saved_profile(Some(binding)).is_err());
        }
    }

    #[test]
    fn paired_edit_gate_preserves_finite_policy_and_rejects_unmeasured_modes() {
        let mut req = request();
        // Use an already selected opaque finite policy solely to exercise the
        // production mode gate. The actual FLUX selector remains Current.
        let finite = JobTrainingPolicy::resolve("kolors", true, &req).unwrap();
        assert_eq!(finite.profile(), TrainingProfile::WithoutE3);
        let paired = for_native_mode(finite, Workflow::PairedEdit, &req);
        assert_eq!(paired, finite);
        assert!(paired
            .with_native_resume_profile(TrainingProfile::WithoutE3)
            .is_ok());
        assert_eq!(
            paired
                .with_native_resume_profile(TrainingProfile::Current)
                .unwrap()
                .profile(),
            TrainingProfile::Current
        );
        assert!(paired
            .with_native_resume_profile(TrainingProfile::Legacy)
            .is_err());
        let captioned = for_native_mode(finite, Workflow::Captioned, &req);
        assert_eq!(captioned.model(), finite.model());
        assert_eq!(captioned.profile(), TrainingProfile::Current);
        assert!(captioned
            .with_native_resume_profile(TrainingProfile::WithoutE3)
            .is_err());
        req.config.full_finetune = true;
        assert_eq!(
            for_native_mode(finite, Workflow::PairedEdit, &req).profile(),
            TrainingProfile::Current
        );
        req.config.full_finetune = false;
        req.config.network_type = candle_gen::gen_core::NetworkType::Lokr;
        assert_eq!(
            for_native_mode(finite, Workflow::PairedEdit, &req).profile(),
            TrainingProfile::Current
        );
    }

    #[test]
    fn fixed_64_paired_edit_recipe_uses_existing_native_options_contract() {
        use candle_gen::gen_core::train::TrainingItem;
        use gen_core_flux3::training::options::{TrainingOptions, TrainingPrecision, OPTIONS_KEY};
        let mut req = request();
        let mut item = TrainingItem::captioned("metadata-target.png".into(), "a blue tile".into());
        item.reference_image_paths
            .push("metadata-reference.png".into());
        req.items.push(item);
        req.config.resolution = 64;
        req.config.steps = 2;
        req.config.rank = 2;
        req.config.alpha = 2.;
        req.config.learning_rate = 1e-3;
        req.config.optimizer = "adamw".into();
        req.config.loss_type = "mse".into();
        req.config.train_dtype = "bf16".into();
        req.config.seed = 7;
        req.config.batch_size = 1;
        req.config.gradient_accumulation = 1;
        req.config.gradient_checkpointing = false;
        req.config.model_options.insert(
            OPTIONS_KEY.into(),
            serde_json::json!({"mode":"edit", "caption_dropout":0., "random_flip":false}),
        );
        // This parser reads config/path metadata only, never files or tensors.
        let parsed = TrainingOptions::parse(&req).unwrap();
        assert_eq!(parsed.workflow, Workflow::PairedEdit);
        assert_eq!(parsed.precision, TrainingPrecision::Bf16);
        for unsupported in [0, 63, 65, 4097] {
            req.config.resolution = unsupported;
            assert!(TrainingOptions::parse(&req).is_err());
        }
        req.config.resolution = 64;
        req.items[0].reference_image_paths.clear();
        assert!(
            TrainingOptions::parse(&req).is_err(),
            "edit mode requires its reference"
        );
    }
}

//! The trainer testkit verifying itself: a configurable in-crate stub trainer drives each
//! conformance check, and one deliberately-broken variant per check proves the check fires
//! (sc-4895). The stub is pure-host (no tensor library), so these run on the Linux gen-core lane.

use std::path::PathBuf;

use super::*;
use gen_core::registry::TrainerRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{
    Error, Modality, NetworkType, Trainer, TrainerDescriptor, TrainingItem, TrainingOutput,
    TrainingProgress, TrainingRequest,
};

/// The registered stub id (round-trips through the explicit fixture registry below).
const STUB_ID: &str = "testkit_trainer_stub";
/// A stub id deliberately NOT registered — exercises the registry-check failure path.
const UNREG_ID: &str = "testkit_trainer_unregistered_stub";

/// Which contract guarantees the stub upholds. `good()` upholds all; each broken-stub test flips
/// exactly one to false and asserts the matching check fails.
#[derive(Clone, Copy)]
struct Behavior {
    /// `validate()` enforces the dataset/network-type floor (vs. rubber-stamping every request).
    honest_validate: bool,
    /// Emits a `TrainingProgress::Training` per optimizer step.
    emit_progress: bool,
    /// Checks `CancelFlag` in the caching + training loops and bails.
    honor_cancel: bool,
    /// On a cancelled-before-any-step run, returns the typed `Error::Canceled` (vs. a stringified
    /// `Error::Msg`).
    typed_cancel: bool,
    /// `Some(text)` replaces the shared edit floor's refusal with `Error::Msg(text)` — the
    /// flattened-variant / wrong-message class the edit honesty check must catch (sc-24161).
    edit_refusal_override: Option<&'static str>,
    /// Routes `validate` (and so `train`) through the shared technique floor
    /// (`validate_training_techniques`, epic 2123) vs. silently ignoring technique knobs.
    honor_techniques: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            honest_validate: true,
            emit_progress: true,
            honor_cancel: true,
            typed_cancel: true,
            edit_refusal_override: None,
            honor_techniques: true,
        }
    }
}

struct StubTrainer {
    desc: TrainerDescriptor,
    behavior: Behavior,
}

fn stub_desc(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "testkit",
        backend: "stub",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
        supports_control: false,
        // Adapter-only: no full base fine-tune path (sc-14056). The shared
        // `validate_full_finetune_request` floor makes a `full_finetune` request a typed reject.
        supports_full_finetune: false,
        max_reference_images: 0,
        techniques: gen_core::TrainingTechniques::NONE,
    }
}

impl StubTrainer {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }

    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn Trainer> {
        Box::new(Self::new(id, behavior))
    }

    /// The error a cancelled-before-any-step run surfaces — typed `Canceled` for the good stub, a
    /// stringified `Msg` for the broken one (the exact pre-sc-4895 family behavior).
    fn cancel_err(&self) -> Error {
        if self.behavior.typed_cancel {
            Error::Canceled
        } else {
            Error::Msg("stub trainer: training cancelled".to_owned())
        }
    }
}

impl Trainer for StubTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.desc
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if !self.behavior.honest_validate {
            return Ok(());
        }
        // Route through the shared capability floors (F-006), like a real family trainer.
        gen_core::train::validate_control_request(&self.desc, req)?;
        gen_core::train::validate_full_finetune_request(&self.desc, req)?;
        if self.behavior.honor_techniques {
            gen_core::train::validate_training_techniques(&self.desc, req)?;
        }
        gen_core::train::validate_edit_request(&self.desc, req).map_err(|e| {
            match self.behavior.edit_refusal_override {
                Some(text) => Error::Msg(text.to_owned()),
                None => e,
            }
        })?;
        if req.items.is_empty() {
            return Err(Error::Msg("stub trainer: dataset is empty".to_owned()));
        }
        match req.config.network_type {
            NetworkType::Lokr if !self.desc.supports_lokr => {
                Err(Error::Msg("stub trainer: LoKr not supported".to_owned()))
            }
            NetworkType::Lora if !self.desc.supports_lora => {
                Err(Error::Msg("stub trainer: LoRA not supported".to_owned()))
            }
            _ => Ok(()),
        }
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        if self.behavior.honest_validate {
            self.validate(req)?;
        }
        on_progress(TrainingProgress::Preparing);
        on_progress(TrainingProgress::LoadingModel);

        // --- cache (per-item, cancellable) ---
        let total = req.items.len() as u32;
        let mut cached = 0u32;
        for i in 0..req.items.len() {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            cached += 1;
        }
        if cached == 0 {
            // Disambiguate (sc-4895): cancelled-during-caching → typed Canceled; otherwise a real
            // "no usable dataset items" error.
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                return Err(self.cancel_err());
            }
            return Err(Error::Msg(
                "stub trainer: no usable dataset items".to_owned(),
            ));
        }

        // --- train (per-step, cancellable) ---
        let steps = req.config.steps;
        let mut steps_run = 0u32;
        for step in 1..=steps {
            if self.behavior.honor_cancel && req.cancel.is_cancelled() {
                break;
            }
            steps_run = step;
            if self.behavior.emit_progress {
                on_progress(TrainingProgress::Training {
                    step,
                    total: steps,
                    loss: 1.0 / step as f32,
                });
            }
        }
        if steps_run == 0 {
            // Cancelled before any step → typed Canceled, no adapter written (F-040).
            return Err(self.cancel_err());
        }

        on_progress(TrainingProgress::Saving);
        Ok(TrainingOutput {
            adapter_path: req.output_dir.join(&req.file_name),
            steps: steps_run,
            final_loss: 0.0,
        })
    }
}

fn stub_descriptor() -> TrainerDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    Ok(StubTrainer::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: TrainerRegistration = TrainerRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_trainer(STUB_REGISTRATION)
        .build()
        .expect("stub trainer registry should build")
}

fn item(name: &str) -> TrainingItem {
    TrainingItem {
        image_path: PathBuf::from(format!("/nonexistent/{name}.png")),
        caption: format!("a {name}"),
        control_image_path: None,
        model_options: Default::default(),
        reference_image_paths: Vec::new(),
        subject_mask_path: None,
    }
}

fn profile(tmp: &tempfile::TempDir) -> TrainerProfile {
    // Two dummy items (the stub never reads them); 2 steps via `cheap`.
    TrainerProfile::cheap(
        vec![item("red"), item("blue")],
        tmp.path().join("gen_core_testkit_trainer_stub"),
    )
}

fn make_good() -> Box<dyn Trainer> {
    StubTrainer::boxed(STUB_ID, Behavior::good())
}

#[test]
fn good_stub_passes_full_conformance() {
    let tmp = tempfile::tempdir().unwrap();
    trainer_conformance(make_good, &profile(&tmp));
}

#[test]
fn good_stub_passes_every_check_individually() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = StubTrainer::new(STUB_ID, Behavior::good());
    check_trainer_validate(&g, &profile(&tmp)).unwrap();
    check_trainer_progress(&mut g, &profile(&tmp)).unwrap();
    check_trainer_cancellation(&make_good, &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make_good, &profile(&tmp)).unwrap();
    check_trainer_registry(&registry(), &g).unwrap();
}

#[test]
fn dishonest_validate_fails_validate_check() {
    let tmp = tempfile::tempdir().unwrap();
    let g = StubTrainer::new(
        STUB_ID,
        Behavior {
            honest_validate: false,
            ..Behavior::good()
        },
    );
    assert!(check_trainer_validate(&g, &profile(&tmp)).is_err());
}

#[test]
fn missing_progress_fails_progress_check() {
    let tmp = tempfile::tempdir().unwrap();
    let mut g = StubTrainer::new(
        STUB_ID,
        Behavior {
            emit_progress: false,
            ..Behavior::good()
        },
    );
    let err = check_trainer_progress(&mut g, &profile(&tmp)).unwrap_err();
    assert!(err.contains("Training"), "got: {err}");
}

#[test]
fn ignoring_cancel_fails_cancellation_check() {
    let tmp = tempfile::tempdir().unwrap();
    let err = check_trainer_cancellation(
        &|| {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(&tmp),
    )
    .unwrap_err();
    assert!(err.contains("returned Ok"), "got: {err}");
}

#[test]
fn stringified_cancel_fails_cancellation_check() {
    let tmp = tempfile::tempdir().unwrap();
    // The exact pre-sc-4895 family behavior: stops early but returns Error::Msg, not Canceled.
    let err = check_trainer_cancellation(
        &|| {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    typed_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(&tmp),
    )
    .unwrap_err();
    assert!(err.contains("typed Err(Error::Canceled)"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let g = StubTrainer::new(UNREG_ID, Behavior::good());
    assert!(check_trainer_registry(&registry(), &g).is_err());
}

#[test]
#[should_panic(expected = "conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    let tmp = tempfile::tempdir().unwrap();
    trainer_conformance(
        || {
            StubTrainer::boxed(
                STUB_ID,
                Behavior {
                    honor_cancel: false,
                    ..Behavior::good()
                },
            )
        },
        &profile(&tmp),
    );
}

/// A stub advertising `max_reference_images = cap` with the given behavior.
fn edit_stub(cap: u32, behavior: Behavior) -> StubTrainer {
    let mut stub = StubTrainer::new(STUB_ID, behavior);
    stub.desc.max_reference_images = cap;
    stub
}

/// sc-24161: the edit honesty check passes an honest trainer at either kind of cap.
#[test]
fn honest_edit_refusals_pass_the_validate_check() {
    let tmp = tempfile::tempdir().unwrap();
    for cap in [0, 3] {
        check_trainer_validate(&edit_stub(cap, Behavior::good()), &profile(&tmp))
            .unwrap_or_else(|e| panic!("cap {cap}: {e}"));
    }
}

/// sc-24161: a non-edit trainer that refuses an edit dataset with a flattened `Msg` (the LTX-2.5
/// preflight's old shape) fails the check — the refusal must stay a typed `Unsupported`.
#[test]
fn a_flattened_edit_refusal_fails_the_validate_check() {
    let tmp = tempfile::tempdir().unwrap();
    let stub = edit_stub(
        0,
        Behavior {
            edit_refusal_override: Some("edit training is not supported"),
            ..Behavior::good()
        },
    );
    let err = check_trainer_validate(&stub, &profile(&tmp)).unwrap_err();
    assert!(err.contains("typed Error::Unsupported"), "got: {err}");
}

/// sc-24161: an edit-capable trainer whose over-cap refusal does not name the cap fails the check.
#[test]
fn an_over_cap_refusal_that_does_not_name_the_cap_fails_the_validate_check() {
    let tmp = tempfile::tempdir().unwrap();
    let stub = edit_stub(
        3,
        Behavior {
            edit_refusal_override: Some("bad dataset"),
            ..Behavior::good()
        },
    );
    let err = check_trainer_validate(&stub, &profile(&tmp)).unwrap_err();
    assert!(err.contains("at most 3"), "got: {err}");
}

fn ignores_techniques() -> Behavior {
    Behavior {
        honor_techniques: false,
        ..Behavior::good()
    }
}

/// sc-24826: a trainer that does not declare weight noise but silently accepts it fails the
/// validate check AND the train-entry refusal check.
#[test]
fn silently_ignored_weight_noise_fails_both_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let err = check_trainer_validate(
        &StubTrainer::new(STUB_ID, ignores_techniques()),
        &profile(&tmp),
    )
    .unwrap_err();
    assert!(
        err.contains("techniques.weight_noise == false"),
        "got: {err}"
    );
    let err = check_trainer_technique_refusal(
        &|| StubTrainer::boxed(STUB_ID, ignores_techniques()),
        &profile(&tmp),
    )
    .unwrap_err();
    assert!(err.contains("silently ignored"), "got: {err}");
}

/// sc-24826: a trainer that declares weight noise passes both checks (accepts the knob, still
/// refuses it with a full fine-tune).
#[test]
fn declared_weight_noise_passes_the_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
        stub.desc.techniques.weight_noise = true;
        Box::new(stub)
    };
    check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap();
    trainer_conformance(make, &profile(&tmp));
}

/// sc-24826: a declared-weight-noise trainer that rejects the knob anyway fails the validate check.
#[test]
fn declared_but_rejected_weight_noise_fails_the_validate_check() {
    let tmp = tempfile::tempdir().unwrap();
    // Declares weight noise but routes through a descriptor copy that does not — the floor refuses.
    struct Liar(StubTrainer, TrainerDescriptor);
    impl Trainer for Liar {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.1
        }
        fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
            self.0.validate(req)
        }
        fn train(
            &mut self,
            req: &TrainingRequest,
            on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> gen_core::Result<TrainingOutput> {
            self.0.train(req, on_progress)
        }
    }
    let inner = StubTrainer::new(STUB_ID, Behavior::good());
    let mut claimed = stub_desc(STUB_ID);
    claimed.techniques.weight_noise = true;
    let err = check_trainer_validate(&Liar(inner, claimed), &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("techniques.weight_noise == true"),
        "got: {err}"
    );
}

/// sc-24827: a trainer that declares weight noise but silently accepts an undeclared gradient-noise
/// request fails the validate check AND the train-entry refusal check (the gradient probe runs
/// independently of the weight-noise one).
#[test]
fn silently_ignored_gradient_noise_fails_both_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, ignores_techniques());
        stub.desc.techniques.weight_noise = true;
        Box::new(stub)
    };
    let err = check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("techniques.gradient_noise == false"),
        "got: {err}"
    );
    let err = check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("gradient_noise_eta > 0") && err.contains("silently ignored"),
        "got: {err}"
    );
}

/// sc-24827: a trainer declaring the full adapter-noise pair passes every technique check and the
/// whole conformance suite.
#[test]
fn declared_adapter_noise_passes_the_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
        stub.desc.techniques = gen_core::TrainingTechniques::ADAPTER_NOISE;
        Box::new(stub)
    };
    check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap();
    trainer_conformance(make, &profile(&tmp));
}

/// sc-2127: a trainer that does not declare resolution buckets but silently accepts them (it
/// rubber-stamps every technique knob; both noise techniques are declared so those probes pass)
/// fails the
/// validate check AND the train-entry refusal check, naming the bucket flag.
#[test]
fn silently_ignored_resolution_buckets_fail_both_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, ignores_techniques());
        // Every probed technique but buckets is declared, so only the bucket probe can fail.
        stub.desc.techniques = gen_core::TrainingTechniques {
            depth_anchoring: true,
            subject_mask_loss: true,
            ..gen_core::TrainingTechniques::ADAPTER_NOISE
        };
        Box::new(stub)
    };
    let err = check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("techniques.resolution_buckets == false"),
        "got: {err}"
    );
    let err = check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("resolution_buckets set") && err.contains("silently ignored"),
        "got: {err}"
    );
}

/// sc-2127: a trainer that declares resolution buckets passes the technique checks and the
/// bucketed progress run.
#[test]
fn declared_resolution_buckets_pass_the_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
        stub.desc.techniques.resolution_buckets = true;
        Box::new(stub)
    };
    check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap();
    check_trainer_bucketed_progress(&make, &profile(&tmp)).unwrap();
    trainer_conformance(make, &profile(&tmp));
}

/// sc-2127: a bucket-capable trainer whose bucketed run does not stream training progress is caught
/// by the bucketed progress check (the positive path is a real run, not just a validate stamp).
#[test]
fn declared_resolution_buckets_that_break_training_fail_the_bucketed_progress_check() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(
            STUB_ID,
            Behavior {
                emit_progress: false,
                ..Behavior::good()
            },
        );
        stub.desc.techniques.resolution_buckets = true;
        Box::new(stub)
    };
    let err = check_trainer_bucketed_progress(&make, &profile(&tmp)).unwrap_err();
    assert!(err.contains("Training"), "got: {err}");
}

/// sc-2125: a trainer that declares both noise techniques but silently accepts an undeclared
/// depth-anchoring request fails the validate check and the train-entry refusal check on the depth
/// probe. Mutation: drop the depth probe from `TECHNIQUE_PROBES` ⇒ red.
#[test]
fn silently_ignored_depth_anchoring_fails_both_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, ignores_techniques());
        stub.desc.techniques = gen_core::TrainingTechniques::ADAPTER_NOISE;
        Box::new(stub)
    };
    let err = check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("techniques.depth_anchoring == false"),
        "got: {err}"
    );
    let err = check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("depth_anchoring") && err.contains("silently ignored"),
        "got: {err}"
    );
}

/// sc-2125: a trainer that declares depth anchoring (and routes through the floor) passes the
/// technique checks — depth anchoring is not adapter-only, so no full-fine-tune refusal is demanded.
#[test]
fn declared_depth_anchoring_passes_the_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
        stub.desc.techniques.depth_anchoring = true;
        Box::new(stub)
    };
    check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap();
}

/// sc-24828: a trainer that declares every other probed technique but silently accepts an
/// undeclared subject-masked-loss request fails the validate check and the train-entry refusal
/// check on the masked-loss probe. Mutation: drop the probe from `TECHNIQUE_PROBES` ⇒ red.
#[test]
fn silently_ignored_subject_mask_loss_fails_both_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, ignores_techniques());
        stub.desc.techniques = gen_core::TrainingTechniques {
            depth_anchoring: true,
            ..gen_core::TrainingTechniques::ADAPTER_NOISE
        };
        Box::new(stub)
    };
    let err = check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("techniques.subject_mask_loss == false"),
        "got: {err}"
    );
    let err = check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap_err();
    assert!(
        err.contains("subject_mask_loss") && err.contains("silently ignored"),
        "got: {err}"
    );
}

/// sc-24828: a trainer that declares subject-masked loss passes the technique checks — accepted on
/// a fully masked request, refused (through the shared floor) when an item has no mask, and not
/// required to refuse a full fine-tune (a loss term, not adapter-only).
#[test]
fn declared_subject_mask_loss_passes_the_technique_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let make = || -> Box<dyn Trainer> {
        let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
        stub.desc.techniques.subject_mask_loss = true;
        Box::new(stub)
    };
    check_trainer_validate(make().as_ref(), &profile(&tmp)).unwrap();
    check_trainer_technique_refusal(&make, &profile(&tmp)).unwrap();
}

/// sc-24828: a trainer that declares subject-masked loss but trains an item without its mask
/// (its `validate` papers over a missing mask) fails the validate check. Mutation: drop
/// `check_subject_mask_missing_refused` from `check_trainer_validate` ⇒ red.
#[test]
fn declared_subject_mask_loss_that_ignores_missing_masks_fails_the_validate_check() {
    struct MaskForger(StubTrainer);
    impl Trainer for MaskForger {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.0.desc
        }
        fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
            let mut forged = req.clone();
            if forged.config.subject_mask_loss.is_some() {
                for item in &mut forged.items {
                    item.subject_mask_path
                        .get_or_insert_with(|| item.image_path.clone());
                }
            }
            self.0.validate(&forged)
        }
        fn train(
            &mut self,
            req: &TrainingRequest,
            on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> gen_core::Result<TrainingOutput> {
            self.0.train(req, on_progress)
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    let mut stub = StubTrainer::new(STUB_ID, Behavior::good());
    stub.desc.techniques.subject_mask_loss = true;
    let err = check_trainer_validate(&MaskForger(stub), &profile(&tmp)).unwrap_err();
    assert!(err.contains("lacking its mask"), "got: {err}");
}

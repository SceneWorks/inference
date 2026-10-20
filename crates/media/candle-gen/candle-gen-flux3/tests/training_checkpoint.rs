//! Actual saved-state continuation/reload on SYNTHETIC tiny weights, not downloaded BFL weights.
use crate::common::{fixture, fixtures, host_f32, ids_of, tiny_config};
use candle_gen::candle_core::{backprop::GradStore, DType, Device};
use candle_gen::gen_core::flux3::{
    training::{
        checkpoint::{ResumePolicy, TrainingCursor},
        lora_targets, ResumeIdentity,
    },
    BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION,
};
use candle_gen::gen_core::{AdapterKind, AdapterSpec, CancelFlag};
use candle_gen::train::optim::{accumulate_grads, TrainOptimizer};
use candle_gen_flux3::{
    training::{
        checkpoint::{export_adapter, restore, save, Snapshot},
        TrainingGraph,
    },
    Flux3Dit, SampleTokens,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn graph() -> TrainingGraph {
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let dit = Flux3Dit::load(
        &fixtures().join("tiny-base-snapshot/model.safetensors"),
        &config.dit,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let golden = fixture("flux3_training_golden.safetensors");
    let initial = lora_targets(&config.dit, &[])
        .unwrap()
        .into_iter()
        .map(|projection| {
            let down = golden
                .require(&format!("initial/{}.lora_A.weight", projection.native))
                .clone();
            (projection.native, down)
        })
        .collect();
    TrainingGraph::new(dit, 3, 1.5, &[], initial).unwrap()
}
fn optimizer(graph: &TrainingGraph) -> TrainOptimizer {
    TrainOptimizer::from_config("adamw", graph.vars(), 0.0003, 0.0001).unwrap()
}
fn identity(graph: &TrainingGraph) -> ResumeIdentity {
    let mut factors = BTreeMap::new();
    for factor in &graph.factors {
        factors.insert(
            format!("{}.lora_A.weight", factor.target.native),
            factor.down.dims().to_vec(),
        );
        factors.insert(
            format!("{}.lora_B.weight", factor.target.native),
            factor.up.dims().to_vec(),
        );
    }
    // Repo/revision are the S6 FORMAT-COMPATIBILITY target. The actual artifact
    // identity below is the committed tiny synthetic file, NOT the BFL LFS hash.
    let artifact = std::fs::read(fixtures().join("tiny-base-snapshot/model.safetensors")).unwrap();
    ResumeIdentity {
        base_repo: BASE_WEIGHTS_REPO.into(),
        base_revision: BASE_WEIGHTS_REVISION.into(),
        artifacts: BTreeMap::from([(
            "synthetic-tiny-base/model.safetensors".into(),
            format!("{:x}", Sha256::digest(artifact)),
        )]),
        dataset: format!(
            "{:x}",
            Sha256::digest(b"synthetic-ordered-cached-inputs-0-1")
        ),
        preprocessing: format!(
            "{:x}",
            Sha256::digest(b"synthetic-cached-latents-caption-reference-fixture-v1")
        ),
        configuration: BTreeMap::from([
            ("optimizer".into(), "adamw".into()),
            ("learning_rate".into(), "0.0003".into()),
            ("rank".into(), "3".into()),
            ("alpha".into(), "1.5".into()),
            ("scaling".into(), "alpha/rank".into()),
            ("precision".into(), "f32".into()),
            ("backend".into(), "candle-cpu".into()),
            ("scheduler".into(), "constant".into()),
            ("accumulation".into(), "1".into()),
            (
                "accumulation_scaling".into(),
                "loss/configured_N_before_backward".into(),
            ),
            ("rng".into(), "synthetic-oracle-cursor-v1".into()),
            (
                "fixture".into(),
                "SYNTHETIC tiny weights, NOT downloaded BFL weights".into(),
            ),
        ]),
        factors,
    }
}
fn grads(graph: &TrainingGraph, step: u64) -> GradStore {
    grads_scaled(graph, step, 1)
}
fn grads_scaled(graph: &TrainingGraph, step: u64, accumulation: u32) -> GradStore {
    let golden = fixture("flux3_training_golden.safetensors");
    let sigma = host_f32(golden.require(&format!("step{step}/sigma")))[0];
    let image = ((golden.require("clean") * (1.0 - sigma) as f64).unwrap()
        + (golden.require("noise") * sigma as f64).unwrap())
    .unwrap();
    let target = golden
        .require("noise")
        .sub(golden.require("clean"))
        .unwrap();
    let text_ids = ids_of(golden.require("text_ids"));
    let image_ids = ids_of(golden.require("image_ids"));
    let reference_ids = ids_of(golden.require("reference_ids"));
    graph
        .loss(
            &[SampleTokens {
                text: golden.require("text"),
                text_ids: &text_ids,
                image: &image,
                image_ids: &image_ids,
                image_cond: Some((golden.require("reference"), &reference_ids)),
                timestep: sigma,
                target_timestep: None,
            }],
            &target,
            1.0 / accumulation as f64,
        )
        .unwrap()
        .backward()
        .unwrap()
}
fn named(graph: &TrainingGraph) -> BTreeMap<String, candle_gen::candle_core::Var> {
    graph
        .factors
        .iter()
        .flat_map(|f| {
            [
                (format!("{}.lora_A.weight", f.target.native), f.down.clone()),
                (format!("{}.lora_B.weight", f.target.native), f.up.clone()),
            ]
        })
        .collect()
}
fn values(graph: &TrainingGraph) -> BTreeMap<String, Vec<u32>> {
    named(graph)
        .into_iter()
        .map(|(name, var)| {
            (
                name,
                host_f32(var.as_tensor())
                    .into_iter()
                    .map(f32::to_bits)
                    .collect(),
            )
        })
        .collect()
}
fn cursor(accumulation: u32) -> TrainingCursor {
    TrainingCursor {
        micro_step: 1,
        update_step: u64::from(accumulation == 1),
        accumulation_count: u32::from(accumulation > 1),
        sampler_epoch: 0,
        sampler_cursor: 1,
        rng_state: vec![0x53, 1, 0, 0],
        last_loss_bits: None,
    }
}

#[test]
fn real_saved_checkpoint_resumes_optimizer_then_adapter_reloads_with_effect_and_zero_identity() {
    let continuous = graph();
    let mut opt = optimizer(&continuous);
    opt.step(&grads(&continuous, 0)).unwrap();
    let identity = identity(&continuous);
    let cursor = cursor(1);
    let policy = ResumePolicy {
        total_micro_steps: 2,
        accumulation: 1,
        sampler_len: 2,
    };
    let dir = tempfile::tempdir().unwrap();
    let cancel = CancelFlag::new();
    let pending = BTreeMap::new();
    let path = save(
        &continuous,
        &opt,
        Snapshot {
            identity: &identity,
            cursor: &cursor,
            policy,
            pending: &pending,
        },
        dir.path(),
        "arbitrary-name-not-step-999.resume.safetensors",
        &cancel,
    )
    .unwrap();
    opt.step(&grads(&continuous, 1)).unwrap();
    let resumed = graph();
    let mut resumed_opt = optimizer(&resumed);
    let state = restore(
        &resumed,
        &mut resumed_opt,
        &identity,
        policy,
        &path,
        &cancel,
    )
    .unwrap();
    assert_eq!(state.cursor, cursor);
    assert!(state.pending.is_empty());
    resumed_opt
        .step(&grads(&resumed, state.cursor.sampler_cursor))
        .unwrap();
    assert_eq!(
        values(&resumed),
        values(&continuous),
        "actual saved optimizer+trainable state must continue bit-exactly on the same CPU backend"
    );
    let adapter = export_adapter(
        &resumed,
        &identity,
        dir.path(),
        "trained.safetensors",
        &cancel,
    )
    .unwrap();
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let spec = AdapterSpec::new(adapter, 1.0, AdapterKind::Lora);
    assert_eq!(
        candle_gen::gen_core::flux3::adapters::inspect(&spec, &config.dit)
            .unwrap()
            .len(),
        54
    );
    let golden = fixture("flux3_training_golden.safetensors");
    let text_ids = ids_of(golden.require("text_ids"));
    let image_ids = ids_of(golden.require("image_ids"));
    let reference_ids = ids_of(golden.require("reference_ids"));
    let sample = [SampleTokens {
        text: golden.require("text"),
        text_ids: &text_ids,
        image: golden.require("noise"),
        image_ids: &image_ids,
        image_cond: Some((golden.require("reference"), &reference_ids)),
        timestep: 0.6,
        target_timestep: None,
    }];
    let load = || {
        Flux3Dit::load(
            &fixtures().join("tiny-base-snapshot/model.safetensors"),
            &config.dit,
            DType::F32,
            &Device::Cpu,
        )
        .unwrap()
    };
    let mut dit = load();
    let bare = dit.forward(&sample).unwrap();
    candle_gen_flux3::adapters::install(&mut dit, std::slice::from_ref(&spec)).unwrap();
    let on = dit.forward(&sample).unwrap();
    assert!(
        host_f32(&on.sub(&bare).unwrap())
            .iter()
            .any(|v| v.abs() > 1e-7),
        "trained adapter must affect actual generation velocity"
    );
    let mut zero = load();
    let mut zero_spec = spec;
    zero_spec.scale = 0.0;
    candle_gen_flux3::adapters::install(&mut zero, &[zero_spec]).unwrap();
    assert_eq!(
        host_f32(&zero.forward(&sample).unwrap())
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        host_f32(&bare)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
}

#[test]
fn interrupted_gradient_accumulation_is_saved_and_resumed_without_reset() {
    let continuous = graph();
    let mut opt = optimizer(&continuous);
    let first = grads_scaled(&continuous, 0, 2);
    let pending = named(&continuous)
        .into_iter()
        .map(|(name, var)| (name, first.get(var.as_tensor()).unwrap().clone()))
        .collect::<BTreeMap<_, _>>();
    let mut identity = identity(&continuous);
    identity
        .configuration
        .insert("accumulation".into(), "2".into());
    let cursor = cursor(2);
    let policy = ResumePolicy {
        total_micro_steps: 2,
        accumulation: 2,
        sampler_len: 2,
    };
    let root = tempfile::tempdir().unwrap();
    let cancel = CancelFlag::new();
    let path = save(
        &continuous,
        &opt,
        Snapshot {
            identity: &identity,
            cursor: &cursor,
            policy,
            pending: &pending,
        },
        root.path(),
        "partial.resume.safetensors",
        &cancel,
    )
    .unwrap();
    let mut acc = Some(first);
    accumulate_grads(
        &mut acc,
        grads_scaled(&continuous, 1, 2),
        &continuous.vars(),
    )
    .unwrap();
    let acc = acc.unwrap();
    opt.step(&acc).unwrap();
    let resumed = graph();
    let mut resumed_opt = optimizer(&resumed);
    let state = restore(
        &resumed,
        &mut resumed_opt,
        &identity,
        policy,
        &path,
        &cancel,
    )
    .unwrap();
    assert_eq!(state.cursor.accumulation_count, 1);
    let mut resumed_acc = GradStore::default();
    for (name, var) in named(&resumed) {
        resumed_acc.insert(var.as_tensor(), state.pending[&name].clone());
    }
    let mut resumed_acc = Some(resumed_acc);
    accumulate_grads(
        &mut resumed_acc,
        grads_scaled(&resumed, state.cursor.sampler_cursor, 2),
        &resumed.vars(),
    )
    .unwrap();
    let resumed_acc = resumed_acc.unwrap();
    resumed_opt.step(&resumed_acc).unwrap();
    assert_eq!(values(&resumed), values(&continuous));
}

#[test]
fn mismatched_identity_and_corrupted_state_are_refused_before_mutating_factors() {
    let trained = graph();
    let mut opt = optimizer(&trained);
    opt.step(&grads(&trained, 0)).unwrap();
    let identity = identity(&trained);
    let cursor = cursor(1);
    let policy = ResumePolicy {
        total_micro_steps: 2,
        accumulation: 1,
        sampler_len: 2,
    };
    let root = tempfile::tempdir().unwrap();
    let flag = CancelFlag::new();
    let pending = BTreeMap::new();
    let path = save(
        &trained,
        &opt,
        Snapshot {
            identity: &identity,
            cursor: &cursor,
            policy,
            pending: &pending,
        },
        root.path(),
        "state.resume.safetensors",
        &flag,
    )
    .unwrap();
    let fresh = graph();
    let mut fresh_opt = optimizer(&fresh);
    let initial = values(&fresh);
    let mut wrong = identity.clone();
    wrong.dataset = "e".repeat(64);
    let error = match restore(&fresh, &mut fresh_opt, &wrong, policy, &path, &flag) {
        Ok(_) => panic!("changed data was accepted"),
        Err(e) => e,
    };
    assert!(matches!(error, candle_gen::CandleError::Unsupported(_)));
    assert!(error.to_string().contains("dataset"));
    assert_eq!(values(&fresh), initial);
    let mut wrong = identity.clone();
    wrong
        .configuration
        .insert("learning_rate".into(), "0.001".into());
    let error = match restore(&fresh, &mut fresh_opt, &wrong, policy, &path, &flag) {
        Ok(_) => panic!("changed optimizer configuration was accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("configuration"));
    assert_eq!(values(&fresh), initial);
    let changed_policy = ResumePolicy {
        total_micro_steps: 3,
        ..policy
    };
    let error = match restore(
        &fresh,
        &mut fresh_opt,
        &identity,
        changed_policy,
        &path,
        &flag,
    ) {
        Ok(_) => panic!("changed effective run policy was accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("policy"));
    assert_eq!(values(&fresh), initial);
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    let corrupted = root.path().join("corrupted.resume.safetensors");
    std::fs::write(&corrupted, bytes).unwrap();
    let error = match restore(&fresh, &mut fresh_opt, &identity, policy, &corrupted, &flag) {
        Ok(_) => panic!("corrupted payload was accepted"),
        Err(e) => e,
    };
    assert!(error.to_string().contains("digest"));
    assert_eq!(values(&fresh), initial);
    let (_, state) = fresh_opt.snapshot().unwrap();
    assert_eq!(state["optim.step"], "0");
}

/// PR #1288 review: the exported global alpha is the CONFIGURED value, never re-derived as
/// `scale * rank` (29/7*7 is 29.000000000000004 in f64, which a re-derivation would publish).
#[test]
fn export_records_the_configured_alpha_verbatim() {
    let rank = 7usize;
    let alpha = 29.0f64;
    assert_ne!(
        (alpha / rank as f64) * rank as f64,
        alpha,
        "the case must be one where re-deriving alpha from the scale rounds"
    );
    let mut config = tiny_config();
    config.dit.flow_map = false;
    let dit = Flux3Dit::load(
        &fixtures().join("tiny-base-snapshot/model.safetensors"),
        &config.dit,
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let initial = lora_targets(&config.dit, &[])
        .unwrap()
        .into_iter()
        .map(|projection| {
            let down = candle_gen::candle_core::Tensor::full(
                0.01f32,
                (rank, projection.input),
                &Device::Cpu,
            )
            .unwrap();
            (projection.native, down)
        })
        .collect();
    let graph = TrainingGraph::new(dit, rank, alpha, &[], initial).unwrap();
    let root = tempfile::tempdir().unwrap();
    let path = export_adapter(
        &graph,
        &identity(&graph),
        root.path(),
        "alpha.safetensors",
        &CancelFlag::new(),
    )
    .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let metadata = header.metadata().as_ref().unwrap();
    assert_eq!(metadata["lora_alpha"], "29");
    assert_eq!(metadata["rank"], "7");
    let peft: serde_json::Value = serde_json::from_str(&metadata["lora_adapter_metadata"]).unwrap();
    assert_eq!(
        peft["lora_alpha"].as_f64().unwrap().to_bits(),
        alpha.to_bits()
    );
    assert_eq!(peft["r"], serde_json::json!(7));
    let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
    let resolved = candle_gen::gen_core::flux3::adapters::inspect(&spec, &config.dit).unwrap();
    assert_eq!(resolved.len(), 54);
    assert!(resolved
        .iter()
        .all(|module| module.rank == rank
            && module.scale.to_bits() == (alpha / rank as f64).to_bits()));
}

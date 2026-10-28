//! Real saved-state continuation on SYNTHETIC tiny weights, on supported MLX hosts only.
use crate::common::{cpu_process, fixture, fixtures, host_f32, on_cpu, tiny_base_config};
use mlx_gen::gen_core::flux3::{
    training::{
        checkpoint::{ResumePolicy, TrainingCursor},
        lora_targets, ResumeIdentity,
    },
    BASE_WEIGHTS_REPO, BASE_WEIGHTS_REVISION,
};
use mlx_gen::{
    gen_core::{AdapterKind, AdapterSpec, CancelFlag},
    train::lora::LoraParams,
    weights::Weights,
};
use mlx_gen_flux3::{
    training::{
        checkpoint::{export_adapter, restore, save, Snapshot},
        optim::SourceAdamW,
        TrainingGraph,
    },
    Flux3Dit, SampleTokens,
};
use mlx_rs::{Array, Dtype};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn ids(array: &Array) -> Vec<[f32; 4]> {
    host_f32(array)
        .chunks_exact(4)
        .map(|v| [v[0], v[1], v[2], v[3]])
        .collect()
}
fn graph() -> (TrainingGraph, LoraParams) {
    let config = tiny_base_config();
    let weights =
        Weights::from_file(fixtures().join("tiny-base-snapshot/model.safetensors")).unwrap();
    let dit = Flux3Dit::from_weights(&weights, &config.dit, Dtype::Float32).unwrap();
    let golden = fixture("flux3_training_golden.safetensors");
    let mut params = LoraParams::new();
    for projection in lora_targets(&config.dit, &[]).unwrap() {
        for suffix in ["lora_A.weight", "lora_B.weight"] {
            let name = format!("{}.{}", projection.native, suffix);
            params.insert(
                name.clone().into(),
                golden.require(&format!("initial/{name}")).unwrap().clone(),
            );
        }
    }
    (TrainingGraph::new(dit, 3, 1.5, &[]).unwrap(), params)
}
fn identity(params: &LoraParams) -> ResumeIdentity {
    // Repo/revision name FORMAT COMPATIBILITY; the artifact hash identifies the
    // actual tiny SYNTHETIC fixture, not a downloaded BFL model or an auth receipt.
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
            ("backend".into(), "mlx-cpu".into()),
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
        factors: params
            .iter()
            .map(|(key, a)| {
                (
                    key.to_string(),
                    a.shape().iter().map(|d| *d as usize).collect(),
                )
            })
            .collect(),
    }
}
fn grads(graph: &mut TrainingGraph, params: &LoraParams, step: u64) -> LoraParams {
    grads_scaled(graph, params, step, 1)
}
fn grads_scaled(
    graph: &mut TrainingGraph,
    params: &LoraParams,
    step: u64,
    accumulation: u32,
) -> LoraParams {
    let golden = fixture("flux3_training_golden.safetensors");
    let get = |key: &str| golden.require(key).unwrap();
    let sigma = host_f32(get(&format!("step{step}/sigma")))[0];
    let image = get("clean")
        .multiply(Array::from_f32(1.0 - sigma))
        .unwrap()
        .add(get("noise").multiply(Array::from_f32(sigma)).unwrap())
        .unwrap();
    let target = get("noise").subtract(get("clean")).unwrap();
    let text_ids = ids(get("text_ids"));
    let image_ids = ids(get("image_ids"));
    let reference_ids = ids(get("reference_ids"));
    graph
        .loss_and_grads(
            params,
            &[SampleTokens {
                text: get("text"),
                text_ids: &text_ids,
                image: &image,
                image_ids: &image_ids,
                image_cond: Some((get("reference"), &reference_ids)),
                timestep: sigma,
                target_timestep: None,
            }],
            &target,
            1.0 / accumulation as f32,
        )
        .unwrap()
        .1
}
fn values(params: &LoraParams) -> BTreeMap<String, Vec<u32>> {
    params
        .iter()
        .map(|(key, a)| {
            (
                key.to_string(),
                host_f32(a).into_iter().map(f32::to_bits).collect(),
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
fn saved_checkpoint_resumes_corrected_optimizer_and_adapter_reloads_with_effect() {
    cpu_process("training_checkpoint::saved_checkpoint_resumes_corrected_optimizer_and_adapter_reloads_with_effect", || on_cpu(|| {
        let (mut continuous,mut params) = graph(); let mut optimizer = SourceAdamW::new(0.0003,0.0001).unwrap();
        let g = grads(&mut continuous,&params,0); optimizer.step(&mut params,&g).unwrap();
        let identity = identity(&params); let cursor = cursor(1); let policy = ResumePolicy { total_micro_steps:2, accumulation:1, sampler_len:2 };
        let root = tempfile::tempdir().unwrap(); let flag = CancelFlag::new(); let pending = LoraParams::new();
        let path = save(&continuous, Snapshot { params:&params, optimizer:&optimizer, identity:&identity, cursor:&cursor, policy, pending:&pending }, root.path(), "arbitrary-not-step999.resume.safetensors", &flag).unwrap();
        let g = grads(&mut continuous,&params,1); optimizer.step(&mut params,&g).unwrap();
        let (mut resumed,mut restored) = graph(); let mut resumed_opt = SourceAdamW::new(0.0003,0.0001).unwrap();
        let state = restore(&resumed,&mut restored,&mut resumed_opt,&identity,policy,&path,&flag).unwrap();
        assert_eq!(state.cursor,cursor); assert!(state.pending.is_empty());
        let g = grads(&mut resumed,&restored,state.cursor.sampler_cursor); resumed_opt.step(&mut restored,&g).unwrap();
        assert_eq!(values(&restored),values(&params));
        let path = export_adapter(&resumed,&restored,&identity,root.path(),"trained.safetensors",&flag).unwrap();
        let config = tiny_base_config();
        let spec = AdapterSpec::new(path,1.0,AdapterKind::Lora);
        assert_eq!(mlx_gen::gen_core::flux3::adapters::inspect(&spec,&config.dit).unwrap().len(),54);
        let golden=fixture("flux3_training_golden.safetensors"); let get=|key:&str| golden.require(key).unwrap();
        let text_ids=ids(get("text_ids")); let image_ids=ids(get("image_ids")); let reference_ids=ids(get("reference_ids"));
        let sample=[SampleTokens {text:get("text"),text_ids:&text_ids,image:get("noise"),image_ids:&image_ids,image_cond:Some((get("reference"),&reference_ids)),timestep:0.6,target_timestep:None}];
        let load=|| {let w=Weights::from_file(fixtures().join("tiny-base-snapshot/model.safetensors")).unwrap(); Flux3Dit::from_weights(&w,&config.dit,Dtype::Float32).unwrap()};
        let mut dit=load(); let bare=dit.forward(&sample).unwrap();
        mlx_gen_flux3::adapters::install(&mut dit,std::slice::from_ref(&spec)).unwrap();
        let on=dit.forward(&sample).unwrap(); assert!(host_f32(&on.subtract(&bare).unwrap()).iter().any(|v| v.abs()>1e-7));
        let mut zero=load(); let mut zero_spec=spec; zero_spec.scale=0.0;
        mlx_gen_flux3::adapters::install(&mut zero,&[zero_spec]).unwrap();
        assert_eq!(host_f32(&zero.forward(&sample).unwrap()).into_iter().map(f32::to_bits).collect::<Vec<_>>(),host_f32(&bare).into_iter().map(f32::to_bits).collect::<Vec<_>>());
    }));
}

#[test]
fn partial_accumulation_roundtrips_and_identity_mismatch_refuses_without_reset() {
    cpu_process("training_checkpoint::partial_accumulation_roundtrips_and_identity_mismatch_refuses_without_reset", || on_cpu(|| {
        let (mut continuous,mut params)=graph(); let mut optimizer=SourceAdamW::new(0.0003,0.0001).unwrap();
        let first=grads_scaled(&mut continuous,&params,0,2); let mut identity=identity(&params); identity.configuration.insert("accumulation".into(),"2".into());
        let cursor=cursor(2); let policy=ResumePolicy {total_micro_steps:2,accumulation:2,sampler_len:2};
        let root=tempfile::tempdir().unwrap(); let flag=CancelFlag::new();
        let path=save(&continuous,Snapshot {params:&params,optimizer:&optimizer,identity:&identity,cursor:&cursor,policy,pending:&first},root.path(),"partial.resume.safetensors",&flag).unwrap();
        let second=grads_scaled(&mut continuous,&params,1,2);
        let average:LoraParams=first.iter().map(|(key,g)| (key.clone(),g.add(&second[key]).unwrap())).collect(); optimizer.step(&mut params,&average).unwrap();
        let (mut resumed,mut restored)=graph(); let mut resumed_opt=SourceAdamW::new(0.0003,0.0001).unwrap();
        let initial=values(&restored); let mut wrong=identity.clone(); wrong.dataset="e".repeat(64);
        assert!(restore(&resumed,&mut restored,&mut resumed_opt,&wrong,policy,&path,&flag).is_err()); assert_eq!(values(&restored),initial); assert!(resumed_opt.moments.is_empty());
        let mut wrong=identity.clone(); wrong.configuration.insert("learning_rate".into(),"0.001".into());
        assert!(restore(&resumed,&mut restored,&mut resumed_opt,&wrong,policy,&path,&flag).is_err()); assert_eq!(values(&restored),initial);
        let changed_policy=ResumePolicy {total_micro_steps:3,..policy};
        assert!(restore(&resumed,&mut restored,&mut resumed_opt,&identity,changed_policy,&path,&flag).is_err()); assert_eq!(values(&restored),initial); assert!(resumed_opt.moments.is_empty());
        let state=restore(&resumed,&mut restored,&mut resumed_opt,&identity,policy,&path,&flag).unwrap(); assert_eq!(state.cursor.accumulation_count,1);
        let second=grads_scaled(&mut resumed,&restored,state.cursor.sampler_cursor,2);
        let average:LoraParams=state.pending.iter().map(|(key,g)| (key.clone(),g.add(&second[key]).unwrap())).collect(); resumed_opt.step(&mut restored,&average).unwrap();
        assert_eq!(values(&restored),values(&params));
    }));
}

#[test]
fn checkpoint_codec_preserves_logical_strided_factor_moment_and_pending_bytes() {
    cpu_process("training_checkpoint::checkpoint_codec_preserves_logical_strided_factor_moment_and_pending_bytes", || {
        on_cpu(|| {
            use mlx_gen_flux3::training::optim::AdamMoment;
            let (source, mut params) = graph();
            let key = std::rc::Rc::<str>::from(format!("{}.lora_A.weight", source.targets[0].native));
            let rows = source.rank; let columns = source.targets[0].input;
            let strided = |offset: f32| {
                let data = (0..rows*columns).map(|i| offset+i as f32*0.001).collect::<Vec<_>>();
                Array::from_slice(&data,&[columns as i32,rows as i32]).transpose_axes(&[1,0]).unwrap()
            };
            params.insert(key.clone(), strided(0.0));
            let mut optimizer = SourceAdamW::new(0.0003,0.0001).unwrap();
            optimizer.moments.insert(key.clone(), AdamMoment {first:strided(0.25),second:strided(0.5),step:1});
            let pending = LoraParams::from([(key.clone(),strided(0.75))]);
            let mut identity = identity(&params); identity.configuration.insert("accumulation".into(),"2".into());
            let cursor = TrainingCursor {micro_step:3,update_step:1,accumulation_count:1,sampler_epoch:0,sampler_cursor:3,rng_state:vec![0x53,3,0,0],last_loss_bits:None};
            let policy = ResumePolicy {total_micro_steps:4,accumulation:2,sampler_len:4};
            let root = tempfile::tempdir().unwrap(); let flag = CancelFlag::new();
            let path = save(&source,Snapshot {params:&params,optimizer:&optimizer,identity:&identity,cursor:&cursor,policy,pending:&pending},root.path(),"strided.resume.safetensors",&flag).unwrap();
            let bytes = std::fs::read(&path).unwrap();
            let (_,header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
            let manifest:serde_json::Value = serde_json::from_str(&header.metadata().as_ref().unwrap()["tensor_manifest"]).unwrap();
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            for (prefix,offset) in [("factor",0.0f32),("moment_first",0.25),("moment_second",0.5),("pending",0.75)] {
                let name = format!("{prefix}/{key}"); let tensor = tensors.tensor(&name).unwrap();
                assert_eq!(tensor.shape(),[rows,columns]);
                // Independent row/column indexing, NOT the codec's flatten/readback helper.
                let expected = (0..rows).flat_map(|row| (0..columns).map(move |col| offset+(col*rows+row) as f32*0.001)).collect::<Vec<_>>();
                let expected_bytes = expected.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
                assert_eq!(tensor.data(),expected_bytes);
                assert_eq!(manifest[&name]["sha256"].as_str(),Some(format!("{:x}",Sha256::digest(&expected_bytes)).as_str()));
            }
            let (fresh,mut restored) = graph(); let mut fresh_optimizer = SourceAdamW::new(0.0003,0.0001).unwrap();
            let state = restore(&fresh,&mut restored,&mut fresh_optimizer,&identity,policy,&path,&flag).unwrap();
            assert_eq!(values(&restored),values(&params));
            assert_eq!(host_f32(&state.pending[&key]),host_f32(&pending[&key]));
            assert_eq!(host_f32(&fresh_optimizer.moments[&key].first),host_f32(&optimizer.moments[&key].first));
        })
    });
}

/// PR #1288 review, MLX mirror: the exported global alpha is the CONFIGURED value, never
/// re-derived as `scale * rank` (29/7*7 is 29.000000000000004 in f64).
#[test]
fn export_records_the_configured_alpha_verbatim() {
    cpu_process(
        "training_checkpoint::export_records_the_configured_alpha_verbatim",
        || {
            on_cpu(|| {
                let rank = 7usize;
                let alpha = 29.0f64;
                assert_ne!((alpha / rank as f64) * rank as f64, alpha);
                let config = tiny_base_config();
                let weights =
                    Weights::from_file(fixtures().join("tiny-base-snapshot/model.safetensors"))
                        .unwrap();
                let dit = Flux3Dit::from_weights(&weights, &config.dit, Dtype::Float32).unwrap();
                let graph = TrainingGraph::new(dit, rank, alpha, &[]).unwrap();
                let mut params = LoraParams::new();
                for target in &graph.targets {
                    params.insert(
                        format!("{}.lora_A.weight", target.native).into(),
                        Array::full::<f32>(
                            &[rank as i32, target.input as i32],
                            Array::from_f32(0.01),
                        )
                        .unwrap(),
                    );
                    params.insert(
                        format!("{}.lora_B.weight", target.native).into(),
                        Array::zeros::<f32>(&[target.output as i32, rank as i32]).unwrap(),
                    );
                }
                let root = tempfile::tempdir().unwrap();
                let path = export_adapter(
                    &graph,
                    &params,
                    &identity(&params),
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
                let peft: serde_json::Value =
                    serde_json::from_str(&metadata["lora_adapter_metadata"]).unwrap();
                assert_eq!(
                    peft["lora_alpha"].as_f64().unwrap().to_bits(),
                    alpha.to_bits()
                );
                assert_eq!(peft["r"], serde_json::json!(7));
                let spec = AdapterSpec::new(path, 1.0, AdapterKind::Lora);
                let resolved =
                    mlx_gen::gen_core::flux3::adapters::inspect(&spec, &config.dit).unwrap();
                assert_eq!(resolved.len(), 54);
                assert!(resolved.iter().all(|module| module.rank == rank
                    && module.scale.to_bits() == (alpha / rank as f64).to_bits()));
            })
        },
    );
}

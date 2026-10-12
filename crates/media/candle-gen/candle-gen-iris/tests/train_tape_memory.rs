//! sc-25690 — what a training step keeps on the autograd tape, on the miniature backbone and the
//! miniature critic (Candle CPU): the folded nodes ([`candle_gen_iris::tape`]) retain a fraction of
//! the plain composable chains' bytes while every gradient stays the plain graph's bit for bit, and
//! an activation-checkpointed step casts the f32 masters once.

use std::collections::{HashMap, HashSet};

use candle_gen::candle_core::{CpuStorage, DType, Storage, Tensor, Var};
use candle_gen::gen_core::iris::restoration_train::ConvNextConfig;
use candle_gen::gen_core::iris::train::ac::AcPolicy;
use candle_gen::gen_core::iris::train::{
    FlowObjective, Prediction, TimestepSampler, TrainSchedule,
};
use candle_gen_iris::nn::loader_cast_bytes;
use candle_gen_iris::tape::with_unfolded_tape;
use candle_gen_iris::train::critic::{head_from_tensors, ConvNextTrunk, Critic};
use candle_gen_iris::train::model::{
    flow_loss, loss_and_grads_ext, StepBatch, TrainExtras, TrainModel, Trainable,
};
use candle_gen_iris::train::optim::{snapshot, Params, Tensors};

use crate::common::{cpu, fixture, host_f32, host_i32, tiny_backbone, tiny_config, Fixture};

fn f32t(t: &Tensor) -> Tensor {
    t.to_dtype(DType::F32).unwrap()
}

fn objective() -> FlowObjective {
    FlowObjective {
        num_train_timesteps: 1000,
        shift: 4.0,
        base_shift: 4.0,
        shift_law: "none".into(),
        shift_base_tokens: 256,
        sampler: TimestepSampler::LogitNormal {
            mean: 0.0,
            std: 1.0,
        },
        prediction: Prediction::Velocity,
        x_pred_sigma_min: 0.05,
    }
}

/// Micro-batch 0 of the training oracle.
fn mb0() -> StepBatch {
    let g = fixture("iris_train_golden.safetensors");
    let get = |k: &str| g.require(&format!("mb0.{k}")).clone();
    let t_idx = host_i32(&get("t_idx"));
    let mask = host_i32(&get("y_mask"));
    let sched = TrainSchedule::new(1000, 4.0);
    let b = t_idx.len();
    let sig: Vec<f32> = t_idx.iter().map(|&t| sched.sigmas[t as usize]).collect();
    let tm: Vec<f32> = t_idx
        .iter()
        .map(|&t| sched.model_times[t as usize])
        .collect();
    StepBatch {
        x0: f32t(&get("x0")),
        noise: f32t(&get("noise")),
        sigma: Tensor::from_vec(sig, (b, 1, 1, 1), &cpu()).unwrap(),
        t: Tensor::from_vec(tm, b, &cpu()).unwrap(),
        states: f32t(&get("y")),
        masks: mask.chunks(mask.len() / b).map(|c| c.to_vec()).collect(),
    }
}

fn base_params() -> Params {
    candle_gen::candle_core::safetensors::load(tiny_backbone().join("model.safetensors"), &cpu())
        .unwrap()
        .into_iter()
        .map(|(k, v)| (k, Var::from_tensor(&f32t(&v)).unwrap()))
        .collect()
}

fn full_model(compute: DType) -> TrainModel {
    TrainModel {
        cfg: tiny_config().model,
        compute,
        device: cpu(),
        trainable: Trainable::Full,
    }
}

fn step(model: &TrainModel, params: &Params, ac: AcPolicy) -> (f32, Tensors) {
    let extras = TrainExtras { repa: None, ac };
    let (loss, _, grads) =
        loss_and_grads_ext(model, params, &mb0(), &objective(), &extras, None, 1.0).unwrap();
    (loss, grads)
}

fn bits(t: &Tensor) -> Vec<u32> {
    host_f32(t).into_iter().map(f32::to_bits).collect()
}

/// Bytes of the buffers `root`'s tape keeps alive beyond the trained tensors themselves: every
/// distinct storage a tracked node holds (views share their source's), measured on the CPU backend.
fn tape_bytes(root: &Tensor) -> usize {
    fn key_and_bytes(t: &Tensor) -> (usize, usize) {
        let (storage, _) = t.storage_and_layout();
        let bytes = match &*storage {
            Storage::Cpu(CpuStorage::F32(v)) => v.len() * 4,
            Storage::Cpu(CpuStorage::BF16(v)) => v.len() * 2,
            Storage::Cpu(CpuStorage::F16(v)) => v.len() * 2,
            Storage::Cpu(CpuStorage::F64(v)) => v.len() * 8,
            other => panic!("unexpected tape storage {:?}", other.dtype()),
        };
        (&*storage as *const Storage as usize, bytes)
    }
    let nodes = root.sorted_nodes();
    let trained: HashSet<usize> = nodes
        .iter()
        .filter(|n| n.is_variable())
        .map(|n| key_and_bytes(n).0)
        .collect();
    let buffers: HashMap<usize, usize> = nodes
        .iter()
        .map(|n| key_and_bytes(n))
        .filter(|(key, _)| !trained.contains(key))
        .collect();
    buffers.values().sum()
}

fn mini_cfg() -> ConvNextConfig {
    ConvNextConfig {
        depths: vec![1, 1, 2, 1],
        dims: vec![8, 16, 24, 32],
        norm_eps: 1e-5,
        embed_dim: 16,
    }
}

fn mini_critic(g: &Fixture) -> Critic {
    let map: HashMap<String, Tensor> = g
        .tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("critic/w/")
                .map(|s| (s.to_string(), v.clone()))
        })
        .collect();
    Critic {
        trunk: ConvNextTrunk::from_map(&map, &mini_cfg(), DType::F32, &cpu()).unwrap(),
        head: mini_cfg().head(12),
        compute: DType::F32,
    }
}

/// The generator's adversarial term on a tracked image, and the critic's own loss on its trained
/// head: `(loss, the tape root, the gradients)`.
fn critic_terms(g: &Fixture) -> Vec<(Tensor, Vec<Vec<u32>>)> {
    let critic = mini_critic(g);
    let (head, sn) = head_from_tensors(&critic.head, |k| {
        g.tensors.get(&format!("critic/head0/{k}")).cloned()
    })
    .unwrap();
    let x = Var::from_tensor(g.require("critic/x")).unwrap();
    // generator step: gradient to the image through the frozen trunk and the head
    let gen = critic
        .generator_term(&snapshot(&head), &sn, x.as_tensor(), 0.8)
        .unwrap()
        .mean_all()
        .unwrap();
    let store = gen.backward().unwrap();
    let gen_grads = vec![bits(store.get(x.as_tensor()).unwrap())];
    // critic step: gradient to the head (the image is a constant)
    let logits = critic
        .logits(&snapshot(&head), &sn, g.require("critic/x"))
        .unwrap();
    let d = logits
        .iter()
        .map(|l| l.mean_all().unwrap())
        .reduce(|a, b| a.add(&b).unwrap())
        .unwrap();
    let store = d.backward().unwrap();
    let d_grads = head
        .values()
        .map(|v| bits(store.get(v.as_tensor()).unwrap()))
        .collect();
    vec![(gen, gen_grads), (d, d_grads)]
}

/// Folding the composable RMSNorm / attention chains into one tape node each changes no bit of the
/// loss or of any gradient — under the f32 policy and the bf16 one, with and without activation
/// checkpointing (whose recompute replays the folded nodes).
#[test]
fn folded_generation_gradients_are_the_plain_graphs_bit_for_bit() {
    let params = base_params();
    for compute in [DType::F32, DType::BF16] {
        let model = full_model(compute);
        for ac in [AcPolicy::None, AcPolicy::Full] {
            let (loss, folded) = step(&model, &params, ac);
            let (plain_loss, plain) = with_unfolded_tape(|| step(&model, &params, ac));
            assert_eq!(
                loss.to_bits(),
                plain_loss.to_bits(),
                "{compute:?} {}: loss",
                ac.name()
            );
            assert_eq!(folded.len(), plain.len());
            let mut nonzero = 0;
            for (k, g) in &plain {
                let want = bits(g);
                nonzero += usize::from(want.iter().any(|b| b << 1 != 0));
                assert_eq!(bits(&folded[k]), want, "{compute:?} {}: {k}", ac.name());
            }
            // the comparison is of real gradients, not of two maps of zeros
            assert!(nonzero * 10 > plain.len() * 9, "{nonzero}/{}", plain.len());
        }
    }
}

/// The folded depthwise sums (the trunk's 7×7, the head's BlurPool) leave the generator term's
/// gradient to the image and the critic loss's gradients to the head bit-identical.
#[test]
fn folded_critic_gradients_are_the_plain_graphs_bit_for_bit() {
    let g = fixture("iris_restoration_train_golden.safetensors");
    let folded = critic_terms(&g);
    let plain = with_unfolded_tape(|| critic_terms(&g));
    for ((f_loss, f_grads), (p_loss, p_grads)) in folded.iter().zip(&plain) {
        assert_eq!(bits(f_loss), bits(p_loss));
        assert_eq!(f_grads.len(), p_grads.len());
        for (f, p) in f_grads.iter().zip(p_grads) {
            assert!(
                p.iter().any(|b| b << 1 != 0),
                "a zero gradient proves nothing"
            );
            assert_eq!(f, p);
        }
    }
}

/// The tape of a generation step and of the generator's adversarial term keep a fraction of what
/// the plain composable chains keep. Measured on these miniatures: 0.71 and 0.14 (the real shapes
/// fold more — the attention scores grow with the square of the token count, these have 4 to 20
/// tokens).
#[test]
fn the_folded_tape_keeps_a_fraction_of_the_plain_chains() {
    let params = base_params();
    let model = full_model(DType::F32);
    let generation = || {
        let dit = model.dit(&snapshot(&params)).unwrap();
        tape_bytes(&flow_loss(&dit, &mb0(), &objective()).unwrap())
    };
    let (folded, plain) = (generation(), with_unfolded_tape(generation));
    eprintln!(
        "generation tape: folded {folded} B, plain {plain} B ({:.2})",
        folded as f64 / plain as f64
    );
    assert!(folded * 100 <= plain * 75, "{folded} vs {plain}");

    let g = fixture("iris_restoration_train_golden.safetensors");
    let critic = || tape_bytes(&critic_terms(&g)[0].0);
    let (folded, plain) = (critic(), with_unfolded_tape(critic));
    eprintln!(
        "critic generator-term tape: folded {folded} B, plain {plain} B ({:.2})",
        folded as f64 / plain as f64
    );
    assert!(folded * 100 <= plain * 20, "{folded} vs {plain}");
}

/// An activation-checkpointed full step builds two backbones (on the tape, and off it for the
/// regions' forward) from one cast of the f32 masters: the bf16 policy casts exactly the bytes a
/// plain step casts.
#[test]
fn activation_checkpointing_casts_the_masters_once() {
    let params = base_params();
    let model = full_model(DType::BF16);
    let cast = |ac: AcPolicy| {
        let before = loader_cast_bytes();
        step(&model, &params, ac);
        loader_cast_bytes() - before
    };
    let (none, full) = (cast(AcPolicy::None), cast(AcPolicy::Full));
    assert!(none > 0, "the bf16 policy casts the f32 masters");
    assert_eq!(full, none, "a checkpointed step must not cast them again");
}

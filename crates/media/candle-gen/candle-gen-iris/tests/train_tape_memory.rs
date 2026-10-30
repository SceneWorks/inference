//! sc-25690 — what a training step keeps on the autograd tape, on the miniature backbone and the
//! miniature critic (Candle CPU): the folded nodes ([`candle_gen_iris::tape`]) retain a fraction of
//! the plain composable chains' bytes while every gradient stays the plain graph's bit for bit, and
//! an activation-checkpointed step casts the f32 masters once.

use std::collections::{HashMap, HashSet};

use candle_gen::candle_core::{CpuStorage, DType, Device, Storage, Tensor, Var, D};
use candle_gen::gen_core::iris::restoration_train::ConvNextConfig;
use candle_gen::gen_core::iris::train::ac::AcPolicy;
use candle_gen::gen_core::iris::train::{
    FlowObjective, Prediction, TimestepSampler, TrainSchedule,
};
use candle_gen::gen_core::CancelFlag;
use candle_gen_iris::nn::loader_cast_bytes;
use candle_gen_iris::tape::{fold1, fold3, with_unfolded_tape};
use candle_gen_iris::train::critic::{head_from_tensors, ConvNextTrunk, Critic};
use candle_gen_iris::train::model::{
    flow_loss, loss_and_grads_ext, StepBatch, TrainExtras, TrainModel, Trainable,
};
use candle_gen_iris::train::optim::{snapshot, Params, Tensors};

use crate::common::{
    cpu, device, fixture, host_f32, host_i32, tiny_backbone, tiny_config, Fixture,
};

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
        geometry: Default::default(),
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

/// The critic head-step tape with the BlurPool sum folded, as a percentage of the plain one (measured
/// 34 on the miniature).
const BLUR_FOLD_PCT: usize = 40;

fn mini_cfg() -> ConvNextConfig {
    ConvNextConfig {
        depths: vec![1, 1, 2, 1],
        dims: vec![8, 16, 24, 32],
        norm_eps: 1e-5,
        embed_dim: 16,
    }
}

fn mini_critic(g: &Fixture, compute: DType, device: &Device) -> Critic {
    let map: HashMap<String, Tensor> = g
        .tensors
        .iter()
        .filter_map(|(k, v)| {
            k.strip_prefix("critic/w/")
                .map(|s| (s.to_string(), v.clone()))
        })
        .collect();
    Critic {
        trunk: ConvNextTrunk::from_map(&map, &mini_cfg(), compute, device).unwrap(),
        head: mini_cfg().head(12),
        compute,
    }
}

/// The generator's adversarial term on a tracked image, and the critic's own loss on its trained
/// head: `(the tape root, the gradients)` each.
fn critic_terms_on(g: &Fixture, compute: DType, device: &Device) -> Vec<(Tensor, Vec<Vec<u32>>)> {
    let critic = mini_critic(g, compute, device);
    let on = |t: &Tensor| t.to_device(device).unwrap();
    let (head, sn) = head_from_tensors(&critic.head, |k| {
        g.tensors.get(&format!("critic/head0/{k}")).map(on)
    })
    .unwrap();
    let image = on(g.require("critic/x"));
    let x = Var::from_tensor(&image).unwrap();
    // generator step: gradient to the image through the frozen trunk and the head
    let gen = critic
        .generator_term(&snapshot(&head), &sn, x.as_tensor(), 0.8)
        .unwrap()
        .mean_all()
        .unwrap();
    let store = gen.backward().unwrap();
    let gen_grads = vec![bits(store.get(x.as_tensor()).unwrap())];
    // critic step: gradient to the head (the image is a constant)
    let logits = critic.logits(&snapshot(&head), &sn, &image).unwrap();
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

fn critic_terms(g: &Fixture) -> Vec<(Tensor, Vec<Vec<u32>>)> {
    critic_terms_on(g, DType::F32, &cpu())
}

/// Folding the composable RMSNorm / attention chains into one tape node each changes no bit of the
/// loss or of any gradient — under the f32 policy and the bf16 one, with and without activation
/// checkpointing (whose recompute replays the folded nodes).
#[test]
#[ignore = "model inference; explicit model workflow only"]
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::folded_generation_gradients_are_the_plain_graphs_bit_for_bit");
}

/// The folded depthwise sums (the trunk's 7×7, the head's BlurPool) leave the generator term's
/// gradient to the image and the critic loss's gradients to the head bit-identical.
#[test]
#[ignore = "model inference; explicit model workflow only"]
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::folded_critic_gradients_are_the_plain_graphs_bit_for_bit");
}

/// The tape of a generation step and of the generator's adversarial term keep a fraction of what
/// the plain composable chains keep. Measured on these miniatures: 0.71 and 0.14 (the real shapes
/// fold more — the attention scores grow with the square of the token count, these have 4 to 20
/// tokens).
#[test]
#[ignore = "model inference; explicit model workflow only"]
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

    // The critic's own step: the image is a constant, so the trunk leaves no tape at all and the
    // only fold on it is the head's second BlurPool (the one behind a trained convolution).
    let head_step = || tape_bytes(&critic_terms(&g)[1].0);
    let (folded, plain) = (head_step(), with_unfolded_tape(head_step));
    eprintln!(
        "critic head-step tape (BlurPool): folded {folded} B, plain {plain} B ({:.2})",
        folded as f64 / plain as f64
    );
    assert!(folded * 100 <= plain * BLUR_FOLD_PCT, "{folded} vs {plain}");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::the_folded_tape_keeps_a_fraction_of_the_plain_chains");
}

/// An activation-checkpointed full step builds two backbones (on the tape, and off it for the
/// regions' forward) from one cast of the f32 masters: the bf16 policy casts exactly the bytes a
/// plain step casts.
#[test]
#[ignore = "model inference; explicit model workflow only"]
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

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::activation_checkpointing_casts_the_masters_once");
}

/// Deterministic, non-symmetric values in `[-1, 1)`.
fn ramp(shape: &[usize], seed: u32) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n as u32)
        .map(|i| (((i * 7919 + seed * 104_729) % 1000) as f32) / 500.0 - 1.0)
        .collect();
    Tensor::from_vec(data, shape, &cpu()).unwrap()
}

/// The folded SDPA kernel in its CHUNKED form — a score budget of two query rows, so three kernel
/// launches whose outputs are concatenated — with grouped-query k / v (each kv head expanded to two
/// query heads) and a per-row `[B, 1, Sq, Sk]` mask: the gradients to the q, k and v leaves are the
/// plain chain's bit for bit.
#[test]
fn folded_chunked_attention_gradients_are_the_plain_graphs_bit_for_bit() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let (b, h, kv, sq, sk, d) = (2usize, 4usize, 2usize, 6usize, 5usize, 8usize);
    let (q0, k0, v0) = (
        Var::from_tensor(&ramp(&[b, h, sq, d], 1)).unwrap(),
        Var::from_tensor(&ramp(&[b, kv, sk, d], 2)).unwrap(),
        Var::from_tensor(&ramp(&[b, kv, sk, d], 3)).unwrap(),
    );
    // additive mask: some keys closed per batch row and query row
    let mask = ramp(&[b, 1, sq, sk], 4)
        .ge(0.6)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .affine(-1e4, 0.0)
        .unwrap();
    let weight = ramp(&[b, h, sq, d], 5);
    let budget = 2 * b * h * sk; // two query rows per launch
    let launches = Arc::new(AtomicUsize::new(0));
    let run = || {
        let expand = |t: &Tensor| {
            t.unsqueeze(2)
                .unwrap()
                .expand((b, kv, h / kv, sk, d))
                .unwrap()
                .reshape((b, h, sk, d))
                .unwrap()
        };
        // non-leaf operands, each read by the attention only
        let q = (q0.as_tensor() * 1.5).unwrap();
        let (k, v) = (expand(k0.as_tensor()), expand(v0.as_tensor()));
        let (mask, launches) = (mask.clone(), launches.clone());
        let out = fold3("test-sdpa", &q, &k, &v, move |q, k, v| {
            let softmax = |s: &Tensor| {
                launches.fetch_add(1, Ordering::Relaxed);
                candle_gen::candle_nn::ops::softmax(s, D::Minus1)
            };
            candle_gen::sdpa_budgeted_bhsd(q, k, v, 0.35, Some(&mask), softmax, budget)
        })
        .unwrap();
        let loss = out.mul(&weight).unwrap().sum_all().unwrap();
        let store = loss.backward().unwrap();
        let grads: Vec<Vec<u32>> = [&q0, &k0, &v0]
            .iter()
            .map(|v| bits(store.get(v.as_tensor()).unwrap()))
            .collect();
        (bits(&loss), grads)
    };
    launches.store(0, Ordering::Relaxed);
    let plain = with_unfolded_tape(run);
    assert_eq!(
        launches.load(Ordering::Relaxed),
        3,
        "six query rows at two per launch"
    );
    launches.store(0, Ordering::Relaxed);
    let folded = run();
    // the value off the tape, then the replay on it
    assert_eq!(launches.load(Ordering::Relaxed), 6);
    assert_eq!(folded.0, plain.0, "loss");
    for (name, (f, p)) in ["q", "k", "v"].iter().zip(folded.1.iter().zip(&plain.1)) {
        assert!(
            p.iter().any(|b| b << 1 != 0),
            "{name}: a zero gradient proves nothing"
        );
        assert_eq!(f, p, "{name}");
    }
}

/// The module's own `attention` under a request-scoped score budget (the bounded-attention rung:
/// batch groups, each chunked over its query rows) with tracked operands — a combination no
/// product path reaches today (only the render sets a budget), kept correct: the fold wraps the
/// bounded kernel whole, so the gradients are still the plain chain's bit for bit.
#[test]
fn a_tracked_attention_under_a_score_budget_folds_bit_for_bit() {
    use candle_gen_iris::nn::{attention, with_attention_budget};
    let (b, n, h, kv, d) = (3usize, 6usize, 4usize, 2usize, 8usize);
    let (q0, k0, v0) = (
        Var::from_tensor(&ramp(&[b, n, h, d], 6)).unwrap(),
        Var::from_tensor(&ramp(&[b, n, kv, d], 7)).unwrap(),
        Var::from_tensor(&ramp(&[b, n, kv, d], 8)).unwrap(),
    );
    let weight = ramp(&[b, n, h * d], 9);
    let run = |budget: Option<usize>| {
        with_attention_budget(budget, &CancelFlag::new(), || {
            let out = attention(
                q0.as_tensor(),
                k0.as_tensor(),
                v0.as_tensor(),
                DType::F32,
                None,
            )
            .unwrap();
            let loss = out.mul(&weight).unwrap().sum_all().unwrap();
            let store = loss.backward().unwrap();
            let grads: Vec<Vec<u32>> = [&q0, &k0, &v0]
                .iter()
                .map(|v| bits(store.get(v.as_tensor()).unwrap()))
                .collect();
            (bits(&loss), grads)
        })
    };
    // 24 score elements = one query row of one batch row (4 heads · 6 keys): every launch is a
    // single row, so both the batch split and the query-row chunking engage.
    use candle_gen::attention::chunk_probe;
    for budget in [None, Some(4 * 6), Some(2 * 4 * 6 * 6)] {
        chunk_probe::reset();
        let folded = run(budget);
        // the budget reaches the folded kernel: 24 elements = one query row per launch, 288 =
        // four query rows of the whole batch
        let chunks = chunk_probe::max_chunk_count();
        let want = match budget {
            None => 1,
            Some(24) => 6,
            Some(_) => 2,
        };
        assert_eq!(chunks, want, "{budget:?}");
        let plain = with_unfolded_tape(|| run(budget));
        assert_eq!(folded, plain, "budget {budget:?}");
        assert!(plain.1.iter().all(|g| g.iter().any(|b| b << 1 != 0)));
    }
    // and the bounded kernel is not the unbounded one in disguise: same math, another GEMM order
    let (unbounded, bounded) = (run(None), run(Some(4 * 6)));
    let close = |a: &[u32], b: &[u32]| {
        a.iter()
            .zip(b)
            .all(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs() <= 1e-4)
    };
    assert!(close(&unbounded.0, &bounded.0));
    for (u, v) in unbounded.1.iter().zip(&bounded.1) {
        assert!(close(u, v));
    }
}

/// A tracked operand the folded chain never reads gets a zero gradient: candle's backward expects
/// one for every tracked node it visits, and here nothing else reads that node.
#[test]
fn a_tracked_operand_the_chain_never_reads_gets_a_zero_gradient() {
    let x = Var::from_tensor(&ramp(&[3, 4], 10)).unwrap();
    let y = Var::from_tensor(&ramp(&[3, 4], 11)).unwrap();
    let run = || {
        let a = (x.as_tensor() * 3.0).unwrap();
        let unread = (y.as_tensor() * 2.0).unwrap();
        let out = fold3("test-unread", &a, &a, &unread, |p, q, _| p.mul(q)).unwrap();
        let store = out.sum_all().unwrap().backward().unwrap();
        (
            bits(store.get(x.as_tensor()).unwrap()),
            store.get(y.as_tensor()).map(bits),
        )
    };
    let (gx, gy) = run();
    let (plain_x, plain_y) = with_unfolded_tape(run);
    assert_eq!(gx, plain_x);
    // the plain chain never reaches `y` at all; the folded node names it, with a zero gradient
    assert_eq!(plain_y, None);
    assert_eq!(gy, Some(vec![0u32; 12]));
}

/// A chain that returns a view into a larger buffer (a prefix `narrow` is contiguous at offset 0)
/// still yields a node that owns exactly its elements, with the plain chain's value and gradient.
#[test]
fn a_folded_view_is_compacted_to_its_own_elements() {
    let x = Var::from_tensor(&ramp(&[4, 3], 12)).unwrap();
    let run = || {
        let a = (x.as_tensor() * 3.0).unwrap();
        let out = fold1("test-prefix", &a, |a| a.sqr()?.narrow(0, 0, 2)).unwrap();
        let store = out.sum_all().unwrap().backward().unwrap();
        let (storage, layout) = out.storage_and_layout();
        let held = match &*storage {
            Storage::Cpu(CpuStorage::F32(v)) => v.len(),
            other => panic!("unexpected storage {:?}", other.dtype()),
        };
        (
            bits(&out),
            bits(store.get(x.as_tensor()).unwrap()),
            held,
            layout.shape().elem_count(),
        )
    };
    let (value, grad, held, elems) = run();
    let (plain_value, plain_grad, plain_held, _) = with_unfolded_tape(run);
    assert_eq!(value, plain_value);
    assert_eq!(grad, plain_grad);
    assert_eq!(
        plain_held, 12,
        "the plain chain's result is a view of the 4×3 square"
    );
    assert_eq!(
        (held, elems),
        (6, 6),
        "the folded node owns its 2×3 prefix only"
    );
}

/// The folds on the BUILD's device at the bf16 policy (`cuda:0` under `--features cuda`, where the
/// critic's convolutions and the attention GEMMs run as native bf16 kernels; on a CPU build the
/// critic falls back to f32, which has no half convolution). Run by the CUDA memory-probe lane.
/// Bit-identity needs a deterministic device, so the plain chain is evaluated twice first: when it
/// agrees with itself the folded tape must agree with it bit for bit.
#[test]
#[ignore = "device check: run on the CUDA lane (cargo test --features cuda -- --ignored)"]
fn folds_are_bit_identical_on_the_build_device() {
    let dev = device();
    let g = fixture("iris_restoration_train_golden.safetensors");
    let critic_compute = if dev.is_cpu() {
        DType::F32
    } else {
        DType::BF16
    };
    let params: Params = base_params()
        .into_iter()
        .map(|(k, v)| {
            let t = v.as_tensor().to_device(&dev).unwrap();
            (k, Var::from_tensor(&t).unwrap())
        })
        .collect();
    let model = TrainModel {
        geometry: Default::default(),
        cfg: tiny_config().model,
        compute: DType::BF16,
        device: dev.clone(),
        trainable: Trainable::Full,
    };
    let batch = {
        let b = mb0();
        let on = |t: &Tensor| t.to_device(&dev).unwrap();
        StepBatch {
            x0: on(&b.x0),
            noise: on(&b.noise),
            sigma: on(&b.sigma),
            t: on(&b.t),
            states: on(&b.states),
            masks: b.masks,
        }
    };
    let all = || {
        let mut out: Vec<Vec<u32>> = Vec::new();
        for ac in [AcPolicy::None, AcPolicy::Full] {
            let extras = TrainExtras { repa: None, ac };
            let (loss, _, grads) =
                loss_and_grads_ext(&model, &params, &batch, &objective(), &extras, None, 1.0)
                    .unwrap();
            out.push(vec![loss.to_bits()]);
            out.extend(grads.values().map(bits));
        }
        for (loss, grads) in critic_terms_on(&g, critic_compute, &dev) {
            out.push(bits(&loss));
            out.extend(grads);
        }
        out
    };
    let (plain, again) = with_unfolded_tape(|| (all(), all()));
    let folded = all();
    let differing =
        |a: &[Vec<u32>], b: &[Vec<u32>]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let noise = differing(&plain, &again);
    let moved = differing(&folded, &plain);
    eprintln!(
        "[[IRIS_FOLD_DEVICE]] device={:?} critic={critic_compute:?} tensors={} plain_vs_plain_differing={noise} folded_vs_plain_differing={moved}",
        dev.location(),
        plain.len()
    );
    assert_eq!(folded.len(), plain.len());
    assert_eq!(
        noise, 0,
        "the device does not reproduce the plain chain bit for bit"
    );
    assert_eq!(moved, 0, "folding moved {moved} tensors on this device");

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::folds_are_bit_identical_on_the_build_device");
}

/// An identity node that holds a token for as long as it is on a tape and, when a gradient passes
/// through it, records how many holders `watch` still has.
struct Witness {
    _token: Option<std::sync::Arc<()>>,
    watch: Option<std::sync::Weak<()>>,
    seen: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl candle_gen::candle_core::CustomOp1 for Witness {
    fn name(&self) -> &'static str {
        "test-witness"
    }
    fn cpu_fwd(
        &self,
        storage: &CpuStorage,
        layout: &candle_gen::candle_core::Layout,
    ) -> candle_gen::candle_core::Result<(CpuStorage, candle_gen::candle_core::Shape)> {
        assert!(layout.is_contiguous() && layout.start_offset() == 0);
        Ok((storage.clone(), layout.shape().clone()))
    }
    fn bwd(
        &self,
        _: &Tensor,
        _: &Tensor,
        grad: &Tensor,
    ) -> candle_gen::candle_core::Result<Option<Tensor>> {
        if let Some(watch) = &self.watch {
            self.seen
                .store(watch.strong_count(), std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Some(grad.clone()))
    }
}

/// A checkpointed backward releases everything downstream of the last regions (the head, the loss,
/// a task's loss networks) once its plain backward has run — before any region is replayed, not
/// after the last one. A witness on the patch embedding sees the gradient arrive at the very end
/// (the propagation out of the first region); by then the witness downstream of the regions must
/// be off the tape.
#[test]
#[ignore = "model inference; explicit model workflow only"]
fn a_checkpointed_backward_drops_the_loss_graph_before_replaying_the_regions() {
    use candle_gen_iris::dit::TextBatch;
    use candle_gen_iris::train::model::grads_under_ac;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    const PROBED: &str = "s_embedder.proj.weight";
    let params = base_params();
    let model = full_model(DType::F32);
    let token = Arc::new(());
    let seen = Arc::new(AtomicUsize::new(usize::MAX));
    let mut tensors = snapshot(&params);
    let upstream = tensors[PROBED]
        .apply_op1(Witness {
            _token: None,
            watch: Some(Arc::downgrade(&token)),
            seen: seen.clone(),
        })
        .unwrap();
    tensors.insert(PROBED.into(), upstream);
    let (tracked, free) = model.dit_pair(&tensors, true).unwrap();
    let batch = mb0();
    let downstream = Witness {
        _token: Some(token.clone()),
        watch: None,
        seen: seen.clone(),
    };
    let grads = grads_under_ac(&tracked, free.as_ref(), AcPolicy::Full, &params, |hooks| {
        let out = tracked.forward_hooked(
            &batch.x0,
            &batch.t,
            &TextBatch {
                states: &batch.states,
                mask: &batch.masks,
            },
            hooks,
        )?;
        Ok(out.sqr()?.apply_op1(downstream)?.mean_all()?)
    })
    .unwrap();
    assert!(
        bits(&grads[PROBED]).iter().any(|b| b << 1 != 0),
        "the gradient reached the patch embedding through its witness"
    );
    assert_eq!(
        seen.load(Ordering::Relaxed),
        1,
        "holders of the downstream token when the last gradient arrived: the test alone"
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED train_tape_memory::a_checkpointed_backward_drops_the_loss_graph_before_replaying_the_regions");
}

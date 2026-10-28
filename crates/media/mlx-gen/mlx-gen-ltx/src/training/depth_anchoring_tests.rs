#![cfg(test)]

use super::*;
use mlx_gen::train::perceptual::{AuxLoss, AuxLossSchedule};
use mlx_gen::train::taehv::{synthetic_taehv_weights, TaehvDecoder};
use mlx_gen_depth::anchor::{synthetic_weights, tiny_config, DepthAnchorLoss};
use mlx_gen_depth::DepthAnythingV2;

const LE: usize = 2;

fn schedule() -> AuxLossSchedule {
    AuxLossSchedule {
        weight: 0.1,
        t_min: 0.0,
        t_max: 1.0,
        every_n: 2,
    }
}

fn path() -> PerceptualPath {
    let tae = TaehvConfig {
        latent_channels: 4,
        channels: [8, 6, 4, 4],
        ..TaehvConfig::taeltx2_3()
    };
    let dec = TaehvDecoder::from_weights(&synthetic_taehv_weights(&tae, 11).unwrap(), tae).unwrap();
    let da2 = tiny_config();
    let depth = DepthAnchorLoss::new(
        DepthAnythingV2::from_weights(&synthetic_weights(&da2, 12).unwrap(), da2).unwrap(),
    );
    PerceptualPath::new(
        Some(Box::new(dec)),
        vec![AuxLoss {
            schedule: schedule(),
            loss: Box::new(depth),
        }],
    )
    .unwrap()
}

/// The one-item, one-bucket schedule `train_impl` builds for a single cached entry.
fn one_item() -> BucketSchedule {
    BucketSchedule::new(1, &TrainingConfig::default().training_buckets(), 7)
}

struct Fixture {
    dit: LtxDiT,
    targets: Vec<LtxLoraTarget>,
    params: LoraParams,
    blocks: Vec<Vec<BlockLoraRef>>,
    clean: Array,
    ctx: Array,
    positions: Array,
    noise: Array,
}

fn fixture() -> Fixture {
    let cfg = crate::transformer::rung4_block_window_tests::tiny_cfg();
    let map = crate::transformer::rung4_block_window_tests::tiny_weight_map(&cfg);
    let mut dit =
        LtxDiT::from_weights(&Weights::from_map(map), &cfg, Precision::quant_f32(8, 64)).unwrap();
    let suffixes: Vec<String> = DEFAULT_TARGET_SUFFIXES
        .iter()
        .map(|s| s.to_string())
        .collect();
    let (targets, params) = build_targets(&mut dit, cfg.num_layers, &suffixes, 2, 7).unwrap();
    let blocks = group_block_targets(&targets, cfg.num_layers as usize);
    let key = |k| Some(random::key(k).unwrap());
    let c = cfg.in_channels;
    let s = (LE * LE) as i32;
    let clean = random::normal::<f32>(&[1, s, c], None, None, key(21).as_ref()).unwrap();
    let noise = random::normal::<f32>(&[1, s, c], None, None, key(22).as_ref()).unwrap();
    let ctx = random::normal::<f32>(
        &[1, 3, cfg.cross_attention_dim],
        None,
        None,
        key(23).as_ref(),
    )
    .unwrap();
    let positions = create_position_grid(1, 1, LE, LE);
    eval([&clean, &noise, &ctx, &positions]).unwrap();
    Fixture {
        dit,
        targets,
        params,
        blocks,
        clean,
        ctx,
        positions,
        noise,
    }
}

fn step(
    f: &mut Fixture,
    sigma: f32,
    ckpt: bool,
    aux: Option<AuxStep<'_>>,
) -> (StepLosses, LoraParams) {
    f.dit.set_sdpa_checkpoint(!ckpt);
    let (l, g) = compute_step_loss_grads(
        &mut f.dit,
        &f.params,
        &f.targets,
        4.0,
        2.0,
        &f.clean,
        &f.ctx,
        &f.positions,
        sigma,
        &f.noise,
        false,
        None,
        ckpt.then_some(f.blocks.as_slice()),
        Dtype::Float32,
        aux,
    )
    .unwrap();
    eval(g.values()).unwrap();
    (l, g)
}

fn cache_of(f: &Fixture, n: usize) -> Vec<(Array, Array, Option<Array>)> {
    (0..n)
        .map(|_| (f.clean.clone(), f.ctx.clone(), None))
        .collect()
}

fn bits(a: &Array) -> Vec<u32> {
    a.as_slice::<f32>().iter().map(|x| x.to_bits()).collect()
}

fn aux<'a>(p: &'a PerceptualPath, plan: &'a StepPlan) -> Option<AuxStep<'a>> {
    Some(AuxStep {
        path: p,
        plan,
        image: 0,
        latent_edge: LE,
    })
}

/// AC (a)+(b), dense and block-checkpointed: a depth step (key 2) computes no diffusion term,
/// its total IS the weighted depth term, and the zero-init LoRA-B factors get a nonzero finite
/// gradient; a diffusion step (key 1) carries no depth term. Mutation: force
/// `diffusion_on = true` ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_step_trains_the_lora_through_depth_only_on_both_paths() {
    let mut f = fixture();
    let d = aux_driver(
        path(),
        &cache_of(&f, 1),
        &[LE],
        &one_item(),
        1,
        &Default::default(),
    )
    .unwrap();
    let p = d.into_path();
    for ckpt in [false, true] {
        let plan = p.plan(1, 0, 0.5).unwrap();
        let (diff, _) = step(&mut f, plan.noise_level, ckpt, aux(&p, &plan));
        assert_eq!(diff.aux, None, "ckpt={ckpt}");
        assert_eq!(Some(diff.total), diff.diffusion);
        let plan = p.plan(2, 0, 0.5).unwrap();
        assert!(!plan.diffusion);
        let (depth, g) = step(&mut f, plan.noise_level, ckpt, aux(&p, &plan));
        assert_eq!(depth.diffusion, None, "ckpt={ckpt}");
        let a = depth.aux.expect("depth term");
        assert!(a > 0.0 && a.is_finite(), "ckpt={ckpt}: {a}");
        assert_eq!(depth.total, a);
        let gb: f32 = g
            .iter()
            .filter(|(k, _)| k.ends_with(".lora_b"))
            .map(|(_, v)| v.abs().unwrap().sum(None).unwrap().item::<f32>())
            .sum();
        assert!(gb > 0.0 && gb.is_finite(), "ckpt={ckpt}: LoRA-B grad {gb}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::depth_anchoring_tests::depth_step_trains_the_lora_through_depth_only_on_both_paths");
}

/// AC (c): depth off ⇒ bit-identical to the pre-epic-2123 step (its closure reproduced), and a
/// diffusion-only step of an enabled path takes the same graph. Mutation: scale the diffusion
/// reduction (×1.0001) ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn depth_off_is_bit_identical_to_the_legacy_step() {
    assert!(load_perceptual_path(&TrainingConfig::default())
        .unwrap()
        .is_none());
    assert_eq!(
        perceptual_footprint_gb(&TrainingConfig::default(), 1024, 4),
        0.0
    );
    let mut f = fixture();
    let (off, g_off) = step(&mut f, 0.5, false, None);
    assert_eq!(off.aux, None);
    let sigma = 0.5f32;
    let x_t = add(
        multiply(&f.clean, Array::from_slice(&[1.0 - sigma], &[1])).unwrap(),
        multiply(&f.noise, Array::from_slice(&[sigma], &[1])).unwrap(),
    )
    .unwrap();
    let target = subtract(&f.noise, &f.clean).unwrap();
    let timestep = Array::from_slice(&[sigma], &[1, 1]);
    let (ctx, pos) = (f.ctx.clone(), f.positions.clone());
    let dit = &mut f.dit;
    let targets = &f.targets;
    dit.set_sdpa_checkpoint(true);
    let legacy = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, 4.0, 2.0, None)?;
        let v = dit
            .forward(&x_t, &timestep, &ctx, None, &pos, None)
            .map_err(|e| Exception::custom(e.to_string()))?;
        Ok(vec![reduce_loss(&subtract(&v, &target)?, None, false)?])
    };
    let (val, g_legacy) = keyed_value_and_grad(legacy)(f.params.clone(), 0).unwrap();
    eval(g_legacy.values()).unwrap();
    assert_eq!(off.total, val[0].item::<f32>());
    for (k, v) in &g_legacy {
        assert_eq!(bits(v), bits(&g_off[k]), "{k}");
    }
    let d = aux_driver(
        path(),
        &cache_of(&f, 1),
        &[LE],
        &one_item(),
        1,
        &Default::default(),
    )
    .unwrap();
    let p = d.into_path();
    let plan = p.plan(1, 0, 0.5).unwrap();
    let (on, g_on) = step(&mut f, 0.5, false, aux(&p, &plan));
    assert_eq!(on, off);
    for (k, v) in &g_off {
        assert_eq!(bits(v), bits(&g_on[k]), "{k}");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::depth_anchoring_tests::depth_off_is_bit_identical_to_the_legacy_step");
}

/// The patchified `(1, S, C)` cache latent unpatchifies to the decoder's NCHW at its bucket's
/// latent edge and round-trips `flatten_latent`; with two buckets the optimizer windows
/// interleave (never two depth steps in a row, sc-2124) and every image gets both kinds across
/// its buckets (reference = (item, bucket) entry at THAT bucket's edge). Mutations: key the
/// alternation per epoch (`AuxAlternation::new(1, epoch_len)` in `AuxDriver::prepare_keyed`) ⇒
/// a depth run ⇒ red; unpatchify every entry at the first bucket's edge ⇒ the 4×4 entries fail
/// to reshape ⇒ red.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn buckets_interleave_with_per_entry_references_at_their_own_edge() {
    let z = random::normal::<f32>(&[1, 4, 1, 3, 3], None, None, Some(&random::key(5).unwrap()))
        .unwrap();
    let back = unflatten_latent(&flatten_latent(&z).unwrap(), 3).unwrap();
    assert_eq!(back.shape(), &[1, 4, 3, 3]);
    assert_eq!(
        bits(&back.flatten(None, None).unwrap()),
        bits(&z.flatten(None, None).unwrap())
    );
    assert!(unflatten_latent(&flatten_latent(&z).unwrap(), 2).is_err());

    let f = fixture();
    let items = 2usize;
    let buckets = [
        gen_core::train::ResolutionBucket {
            resolution: 64,
            repeats: 1,
        },
        gen_core::train::ResolutionBucket {
            resolution: 128,
            repeats: 1,
        },
    ];
    let edges = [2usize, 4];
    let cache: Vec<(Array, Array, Option<Array>)> = (0..items)
        .flat_map(|i| {
            edges.iter().map(move |&le| {
                let s = (le * le) as i32;
                let x = random::normal::<f32>(
                    &[1, s, 4],
                    None,
                    None,
                    Some(&random::key(40 + i as u64 * 10 + le as u64).unwrap()),
                )
                .unwrap();
                (x, Array::zeros::<f32>(&[1]).unwrap(), None)
            })
        })
        .collect();
    drop(f);
    let schedule = BucketSchedule::new(items, &buckets, 7);
    let mut d = aux_driver(path(), &cache, &edges, &schedule, 1, &Default::default()).unwrap();
    assert_eq!(d.path().reference_computations(), cache.len());
    let steps = 2 * schedule.epoch_len() as u32;
    let mut kinds = Vec::new();
    for s in 1..=steps {
        let plan = d.sample(s, &schedule).plan(0.5).unwrap().unwrap().plan;
        kinds.push((schedule.sample((s - 1) as usize).0, !plan.diffusion));
    }
    assert!((0..steps as usize).any(|k| schedule.cache_index(k) != schedule.sample(k).0));
    assert!(
        kinds.windows(2).all(|w| !(w[0].1 && w[1].1)),
        "two depth steps in a row: {kinds:?}"
    );
    for image in 0..items {
        let mine: Vec<bool> = kinds
            .iter()
            .filter(|(i, _)| *i == image)
            .map(|(_, d)| *d)
            .collect();
        assert!(
            mine.contains(&true) && mine.contains(&false),
            "image {image} ({kinds:?})"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::depth_anchoring_tests::buckets_interleave_with_per_entry_references_at_their_own_edge");
}

/// AC (d), E7: depth grows the estimate by TAELTX2.3 + DA2 (more for Large) and the guard
/// counts it on the dense and the checkpointed path (synthetic budgets). Mutations: drop
/// `+ extra_gb` ⇒ red; skip the checkpointed projection ⇒ red.
#[test]
fn memory_estimate_includes_the_aux_models_on_both_paths() {
    let mut on = TrainingConfig::default();
    on.depth_anchoring.schedule = schedule();
    let small = perceptual_footprint_gb(&on, 1024, 4);
    on.depth_anchoring.model_size = gen_core::train::DepthModelSize::Large;
    let large = perceptual_footprint_gb(&on, 1024, 4);
    assert!(
        small > 0.0 && large - small > 1.0,
        "small {small} large {large}"
    );
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    let between = |base: f64| ((base + large / 2.0) / 0.85 * GIB) as usize;
    let dense = projected_dense_peak_gb(16.0 * 16.0);
    assert!(preflight_memory_guard_with_budget(&on, 16, 0.0, false, between(dense)).is_ok());
    assert!(preflight_memory_guard_with_budget(&on, 16, large, false, between(dense)).is_err());
    let ck = checkpointed_baseline_gb();
    assert!(ck < dense);
    assert!(preflight_memory_guard_with_budget(&on, 16, 0.0, true, between(ck)).is_ok());
    let err = preflight_memory_guard_with_budget(&on, 16, large, true, between(ck))
        .expect_err("checkpointed depth job over budget")
        .to_string();
    assert!(err.contains("[depth]"), "{err}");
}

/// AC (e): LTX-2.3 declares depth anchoring and a missing TAELTX2.3 checkpoint is a named error
/// (LTX-2.5's declaration and its six video-less-workflow refusals live in
/// `ltx25_depth_anchoring_tests`). Mutations: undeclare depth on 2.3 ⇒ red; drop the decoder
/// name from the load error ⇒ red.
#[test]
fn ltx23_declares_depth_and_missing_decoder_is_named() {
    assert!(trainer_descriptor().techniques.depth_anchoring);
    let tmp = tempfile::tempdir().unwrap();
    let mut c = TrainingConfig::default();
    c.depth_anchoring.schedule = schedule();
    c.perceptual_decoder_dir = Some(tmp.path().join("no-taehv"));
    c.depth_anchoring.model_dir = Some(tmp.path().join("no-da2"));
    let err = load_perceptual_path(&c).err().unwrap().to_string();
    assert!(err.contains("TAELTX2.3"), "{err}");
}

//! The restoration recipe's training-only networks and loss terms on Candle vs HYPIR's own code —
//! the twin of `mlx-gen-iris/tests/restoration_train_parity.rs` on the same committed fixture
//! (`tools/dump_iris_restoration_train.py`), FP32 on the Candle CPU device (true f32, like the
//! oracle): LPIPS-VGG on formula weights (distances, input gradient); the critic on a miniature timm
//! ConvNeXt (feature levels, generator term + input gradient, the critic step with its two power
//! iterations, loss and head gradients, `clip_grad_norm_` + AdamW); the generator loss of
//! `BaseTrainer.optimize_generator` term by term with input gradients. Tolerances: 1e-4 of each
//! tensor's peak for maps and gradients, 1e-5 for reduced losses and the optimizer step.

use std::collections::HashMap;

use candle_gen::candle_core::{Device, Tensor, Var};
use candle_gen_iris::train::critic::{
    head_from_tensors, multilevel_loss, power_iteration, ConvNextTrunk, Critic, SpectralState,
};
use candle_gen_iris::train::lpips::Lpips;
use candle_gen_iris::train::optim::{clip_grads, snapshot, IrisOptimizer, Params, Tensors};
use gen_core_iris::restoration_train::{ConvNextConfig, CriticHeadConfig, VGG16_WIDTHS};
use gen_core_iris::train::{MuonAdjustLr, OptimizerKind, OptimizerPlan, ParamRoute};

use crate::common::{fixture, host_f32, Fixture};

const FWD: f32 = 1e-4;
const SCALAR: f32 = 1e-5;

fn golden() -> Fixture {
    fixture("iris_restoration_train_golden.safetensors")
}

fn close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length");
    let mut max_abs = 0f32;
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        assert!(d.is_finite(), "{name}: non-finite ({a} vs {b})");
        max_abs = max_abs.max(d);
    }
    let peak = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    eprintln!("{name}: max|Δ|={max_abs:.3e} peak={peak:.3e}");
    assert!(
        max_abs <= tol * peak,
        "{name}: max|Δ|={max_abs:.3e} > {tol:.0e}·{peak:.3e}"
    );
}

fn mini_cfg() -> ConvNextConfig {
    ConvNextConfig {
        depths: vec![1, 1, 2, 1],
        dims: vec![8, 16, 24, 32],
        norm_eps: 1e-5,
        embed_dim: 16,
    }
}

fn mini_head() -> CriticHeadConfig {
    mini_cfg().head(12)
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
        trunk: ConvNextTrunk::from_map(
            &map,
            &mini_cfg(),
            candle_gen::candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap(),
        head: mini_head(),
        compute: candle_gen::candle_core::DType::F32,
    }
}

fn head_state(g: &Fixture, prefix: &str) -> (Params, SpectralState) {
    head_from_tensors(&mini_head(), |k| {
        g.tensors.get(&format!("{prefix}/{k}")).cloned()
    })
    .unwrap()
}

fn grad_of(x: &Tensor, f: impl Fn(&Tensor) -> Tensor) -> (f32, Vec<f32>) {
    let v = Var::from_tensor(x).unwrap();
    let loss = f(v.as_tensor());
    let store = loss.backward().unwrap();
    (
        loss.to_vec0::<f32>().unwrap(),
        host_f32(store.get(v.as_tensor()).unwrap()),
    )
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn lpips_vgg_matches_the_lpips_package() {
    let g = golden();
    let net = Lpips::formula(VGG16_WIDTHS, &Device::Cpu).unwrap();
    let (in0, in1) = (g.require("lpips/in0"), g.require("lpips/in1"));
    let d = net.distance(in0, in1).unwrap();
    close(
        "lpips dist",
        &host_f32(&d),
        &host_f32(g.require("lpips/dist")),
        FWD,
    );
    let (_, grad) = grad_of(in0, |x| net.distance(x, in1).unwrap().mean_all().unwrap());
    close(
        "lpips d/din0",
        &grad,
        &host_f32(g.require("lpips/grad_in0")),
        FWD,
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_parity::lpips_vgg_matches_the_lpips_package");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn critic_trunk_features_match_hypirs_visual_forward() {
    let g = golden();
    let critic = mini_critic(&g);
    let (maps, emb) = critic.trunk.features(g.require("critic/x")).unwrap();
    for (i, m) in maps.iter().enumerate() {
        close(
            &format!("feat{i}"),
            &host_f32(m),
            &host_f32(g.require(&format!("critic/feat{i}"))),
            FWD,
        );
    }
    close(
        "feat3 (pooled)",
        &host_f32(&emb),
        &host_f32(g.require("critic/feat3")),
        FWD,
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_parity::critic_trunk_features_match_hypirs_visual_forward");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_generator_term_and_its_gradient_match() {
    let g = golden();
    let critic = mini_critic(&g);
    let (params, sn) = head_state(&g, "critic/head0");
    let head = snapshot(&params);
    let x = g.require("critic/x");
    let logits = critic.logits(&head, &sn, x).unwrap();
    for (i, l) in logits.iter().enumerate() {
        close(
            &format!("g_logits{i}"),
            &host_f32(l),
            &host_f32(g.require(&format!("critic/g_logits{i}"))),
            FWD,
        );
    }
    let loss = multilevel_loss(&logits, 0.8).unwrap();
    close(
        "g_loss",
        &host_f32(&loss),
        &host_f32(g.require("critic/g_loss")),
        SCALAR,
    );
    let (_, grad) = grad_of(x, |x| {
        critic
            .generator_term(&head, &sn, x, 0.8)
            .unwrap()
            .mean_all()
            .unwrap()
    });
    close(
        "d g_loss / dx",
        &grad,
        &host_f32(g.require("critic/g_grad_x")),
        FWD,
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_parity::the_generator_term_and_its_gradient_match");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_critic_step_matches_power_iterations_losses_grads_and_adamw() {
    let g = golden();
    let critic = mini_critic(&g);
    let head_cfg = mini_head();
    let (params, sn0) = head_state(&g, "critic/head0");
    let cur = snapshot(&params);
    let sn1 = power_iteration(&head_cfg, &cur, &sn0).unwrap();
    let sn2 = power_iteration(&head_cfg, &cur, &sn1).unwrap();
    for (tag, sn) in [("after_real", &sn1), ("after_fake", &sn2)] {
        for layer in head_cfg.layers() {
            let p = &layer.prefix;
            let (u, v) = &sn[p];
            close(
                &format!("{tag} {p} u"),
                &host_f32(u),
                &host_f32(g.require(&format!("critic/{tag}/{p}.weight_u"))),
                SCALAR,
            );
            close(
                &format!("{tag} {p} v"),
                &host_f32(v),
                &host_f32(g.require(&format!("critic/{tag}/{p}.weight_v"))),
                SCALAR,
            );
        }
    }
    let (gt, x) = (g.require("critic/gt"), g.require("critic/x"));
    let real = multilevel_loss(&critic.logits(&cur, &sn1, gt).unwrap(), 0.8).unwrap();
    let fake = multilevel_loss(&critic.logits(&cur, &sn2, x).unwrap(), 0.0).unwrap();
    close(
        "loss_real",
        &host_f32(&real),
        &host_f32(g.require("critic/loss_real")),
        SCALAR,
    );
    close(
        "loss_fake",
        &host_f32(&fake),
        &host_f32(g.require("critic/loss_fake")),
        SCALAR,
    );
    let loss = real
        .mean_all()
        .unwrap()
        .add(&fake.mean_all().unwrap())
        .unwrap();
    close(
        "loss_d",
        &host_f32(&loss),
        &host_f32(g.require("critic/loss_d")),
        SCALAR,
    );
    let store = loss.backward().unwrap();
    for (k, v) in &params {
        close(
            &format!("grad {k}"),
            &host_f32(store.get(v.as_tensor()).unwrap()),
            &host_f32(g.require(&format!("critic/grad/{k}"))),
            FWD,
        );
    }
    // The optimizer on the reference's own gradients.
    let reference: Tensors = params
        .keys()
        .map(|k| (k.clone(), g.require(&format!("critic/grad/{k}")).clone()))
        .collect();
    let (norm, clipped) = clip_grads(reference, 1.0).unwrap();
    close(
        "grad norm",
        &[norm as f32],
        &host_f32(g.require("critic/grad_norm")),
        SCALAR,
    );
    let plan = OptimizerPlan {
        kind: OptimizerKind::AdamW,
        base_lr: 1e-3,
        lr: 1e-3,
        betas: (0.9, 0.999),
        weight_decay: 0.01,
        muon_momentum: 0.95,
        muon_nesterov: true,
        muon_adjust_lr: MuonAdjustLr::None,
        auto_lr: "none".into(),
        base_batch_size: 1,
    };
    let mut opt = IrisOptimizer::new(&plan, &params, |_, _| ParamRoute::AdamW).unwrap();
    opt.step(&params, &clipped, 1e-3).unwrap();
    for (k, p) in &params {
        close(
            &format!("stepped {k}"),
            &host_f32(p.as_tensor()),
            &host_f32(g.require(&format!("critic/stepped/{k}"))),
            SCALAR,
        );
    }

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_parity::the_critic_step_matches_power_iterations_losses_grads_and_adamw");
}

#[test]
#[ignore = "model inference; explicit model workflow only"]
fn the_generator_loss_terms_and_gradients_match_optimize_generator() {
    let g = golden();
    let critic = mini_critic(&g);
    let (params, sn) = head_state(&g, "gloss/head");
    let head = snapshot(&params);
    let net = Lpips::formula(VGG16_WIDTHS, &Device::Cpu).unwrap();
    let x = g.require("gloss/x");
    let gt = g.require("critic/gt");
    let l2 = |v: &Tensor| v.sub(gt).unwrap().sqr().unwrap().mean_all().unwrap();
    let lp = |v: &Tensor| (net.distance(v, gt).unwrap().mean_all().unwrap() * 5.0).unwrap();
    let gan = |v: &Tensor| {
        (critic
            .generator_term(&head, &sn, v, 0.8)
            .unwrap()
            .mean_all()
            .unwrap()
            * 0.5)
            .unwrap()
    };
    for (name, f) in [
        ("l2", &l2 as &dyn Fn(&Tensor) -> Tensor),
        ("lpips", &lp),
        ("gan", &gan),
    ] {
        let (val, grad) = grad_of(x, f);
        close(
            name,
            &[val],
            &host_f32(g.require(&format!("gloss/{name}"))),
            SCALAR,
        );
        // The 64-px LPIPS gradient runs through four pooled VGG stages whose formula-weight
        // activations reach ~3e4 with pre-ReLU values within 1e-1 of zero: a summation-order
        // difference (~1e-6 relative) flips a few ReLU gates there (measured: ≤ 0.9 % of peak; the
        // first four taps' gradients and the 40-px gradient above agree to ~1e-5).
        let tol = if name == "lpips" { 1.5e-2 } else { FWD };
        close(
            &format!("grad {name}"),
            &grad,
            &host_f32(g.require(&format!("gloss/grad_{name}"))),
            tol,
        );
    }
    let (val, grad) = grad_of(x, |v| l2(v).add(&lp(v)).unwrap().add(&gan(v)).unwrap());
    close("total", &[val], &host_f32(g.require("gloss/total")), SCALAR);
    close(
        "grad total",
        &grad,
        &host_f32(g.require("gloss/grad_total")),
        1.5e-2,
    );

    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED restoration_train_parity::the_generator_loss_terms_and_gradients_match_optimize_generator");
}

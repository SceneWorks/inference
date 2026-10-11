//! The restoration recipe's training-only networks and loss terms on MLX vs HYPIR's own code
//! (`tools/dump_iris_restoration_train.py` → `iris_restoration_train_golden.safetensors`), FP32 on
//! the default (Metal) stream — the production path:
//!
//! * LPIPS-VGG (formula weights, the real VGG16 / LPIPS-v0.1 architecture): per-sample distances and
//!   `d mean(LPIPS) / d in0`;
//! * the critic on a miniature timm ConvNeXt trunk: the four feature levels HYPIR's
//!   `_visual_forward` returns, the generator term (eval mode) and its input gradient, the critic
//!   step (train mode — two spectral power iterations, `u`/`v` after each, `loss_D`, every head
//!   gradient), and one `clip_grad_norm_(1.0)` + `AdamW` step of the head;
//! * the generator loss of `BaseTrainer.optimize_generator` — `mse + 5·LPIPS + 0.5·D(x, for_G)`:
//!   each term, the total, and each term's input gradient.
//!
//! Tolerances are relative to each tensor's peak and cover Metal's reduced-precision f32 GEMM/conv
//! accumulation (TF32-class, measured ≤ 2.7e-3 of peak on the critic logits, ≤ 1.5e-2 on the
//! smallest gradient term, whose own peak is ~40× below the total's): [`FWD`] for maps, logits and
//! gradients, [`SCALAR`] for the reduced losses and the optimizer step.
//!
//! (On the MLX CPU stream these gradients used to change from run to run. The cause was MLX's CPU
//! direct convolution, not its GEMM: with the ConvNeXt 7x7 depthwise conv (padding 3) on the
//! trunk's 2x2 last stage, its border loop wrote whole rows past the output buffer, corrupting
//! whatever the eval had allocated next. Fixed by the mlx-rs fork's `cpu-conv-border-clamp.patch`,
//! upstream ml-explore/mlx af9a5924; pinned by the fork's `mlx-tests/tests/cpu_conv_padding_overhang.rs`.)

use std::collections::HashMap;
use std::rc::Rc;

use mlx_gen::gen_core::iris::restoration_train::{ConvNextConfig, CriticHeadConfig, VGG16_WIDTHS};
use mlx_gen::gen_core::iris::train::{OptimizerKind, OptimizerPlan, ParamRoute};
use mlx_gen::weights::Weights;
use mlx_gen_iris::train::critic::{
    head_from_arrays, multilevel_loss, power_iteration, ConvNextTrunk, Critic,
};
use mlx_gen_iris::train::lpips::Lpips;
use mlx_gen_iris::train::optim::{clip_grads, IrisOptimizer, Params};
use mlx_rs::transforms::keyed_value_and_grad;
use mlx_rs::{Array, Dtype};

use crate::common::{fixture, host_f32};

/// Forwards, logits and gradients (relative to peak).
const FWD: f32 = 2e-2;
/// Losses, power-iteration vectors, the optimizer step (relative to peak).
const SCALAR: f32 = 2e-3;

fn golden() -> Weights {
    fixture("iris_restoration_train_golden.safetensors")
}

fn get(w: &Weights, k: &str) -> Array {
    w.get(k)
        .unwrap_or_else(|| panic!("fixture has no {k}"))
        .clone()
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

fn mini_critic(g: &Weights) -> Critic {
    let map: HashMap<String, Array> = g
        .keys()
        .filter_map(|k| {
            k.strip_prefix("critic/w/")
                .map(|s| (s.to_string(), get(g, k)))
        })
        .collect();
    Critic {
        trunk: ConvNextTrunk::from_map(&map, &mini_cfg(), Dtype::Float32).unwrap(),
        head: mini_head(),
        compute: Dtype::Float32,
    }
}

fn head_state(g: &Weights, prefix: &str) -> (Params, mlx_gen_iris::train::critic::SpectralState) {
    head_from_arrays(&mini_head(), |k| g.get(&format!("{prefix}/{k}")).cloned()).unwrap()
}

fn grad_of(x: &Array, f: impl Fn(&Array) -> mlx_rs::error::Result<Array>) -> (f32, Vec<f32>) {
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        Ok(vec![f(&p["x"])?])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let mut p = HashMap::new();
    p.insert(Rc::from("x"), x.clone());
    let (v, g) = vg(p, 0).unwrap();
    (v[0].item::<f32>(), host_f32(&g["x"]))
}

fn exc(e: mlx_gen::Error) -> mlx_rs::error::Exception {
    mlx_rs::error::Exception::custom(e.to_string())
}

#[test]
fn lpips_vgg_matches_the_lpips_package() {
    {
        let g = golden();
        let net = Lpips::formula(VGG16_WIDTHS).unwrap();
        let (in0, in1) = (get(&g, "lpips/in0"), get(&g, "lpips/in1"));
        let d = net.distance(&in0, &in1).unwrap();
        close(
            "lpips dist",
            &host_f32(&d),
            &host_f32(&get(&g, "lpips/dist")),
            FWD,
        );
        let (_, grad) = grad_of(&in0, |x| net.distance(x, &in1).map_err(exc)?.mean(None));
        close(
            "lpips d/din0",
            &grad,
            &host_f32(&get(&g, "lpips/grad_in0")),
            FWD,
        );
    }
}

#[test]
fn critic_trunk_features_match_hypirs_visual_forward() {
    {
        let g = golden();
        let critic = mini_critic(&g);
        let (maps, emb) = critic.trunk.features(&get(&g, "critic/x")).unwrap();
        for (i, m) in maps.iter().enumerate() {
            let nchw = m.transpose_axes(&[0, 3, 1, 2]).unwrap();
            close(
                &format!("feat{i}"),
                &host_f32(&nchw),
                &host_f32(&get(&g, &format!("critic/feat{i}"))),
                FWD,
            );
        }
        close(
            "feat3 (pooled)",
            &host_f32(&emb),
            &host_f32(&get(&g, "critic/feat3")),
            FWD,
        );
    }
}

#[test]
fn the_generator_term_and_its_gradient_match() {
    {
        let g = golden();
        let critic = mini_critic(&g);
        let (params, sn) = head_state(&g, "critic/head0");
        let x = get(&g, "critic/x");
        let logits = critic.logits(&params, &sn, &x).unwrap();
        for (i, l) in logits.iter().enumerate() {
            close(
                &format!("g_logits{i}"),
                &host_f32(l),
                &host_f32(&get(&g, &format!("critic/g_logits{i}"))),
                FWD,
            );
        }
        let loss = multilevel_loss(&logits, 0.8).unwrap();
        close(
            "g_loss",
            &host_f32(&loss),
            &host_f32(&get(&g, "critic/g_loss")),
            SCALAR,
        );
        let (_, grad) = grad_of(&x, |x| {
            critic
                .generator_term(&params, &sn, x, 0.8)
                .map_err(exc)?
                .mean(None)
        });
        close(
            "d g_loss / dx",
            &grad,
            &host_f32(&get(&g, "critic/g_grad_x")),
            FWD,
        );
    }
}

#[test]
fn the_critic_step_matches_power_iterations_losses_grads_and_adamw() {
    {
        let g = golden();
        let critic = mini_critic(&g);
        let head = mini_head();
        let (params, sn0) = head_state(&g, "critic/head0");
        let sn1 = power_iteration(&head, &params, &sn0).unwrap();
        let sn2 = power_iteration(&head, &params, &sn1).unwrap();
        for (tag, sn) in [("after_real", &sn1), ("after_fake", &sn2)] {
            for layer in head.layers() {
                let (u, v) = &sn[&layer.prefix];
                let p = &layer.prefix;
                close(
                    &format!("{tag} {p} u"),
                    &host_f32(u),
                    &host_f32(&get(&g, &format!("critic/{tag}/{p}.weight_u"))),
                    1e-5,
                );
                close(
                    &format!("{tag} {p} v"),
                    &host_f32(v),
                    &host_f32(&get(&g, &format!("critic/{tag}/{p}.weight_v"))),
                    1e-5,
                );
            }
        }
        let (gt, x) = (get(&g, "critic/gt"), get(&g, "critic/x"));
        let real = multilevel_loss(&critic.logits(&params, &sn1, &gt).unwrap(), 0.8).unwrap();
        let fake = multilevel_loss(&critic.logits(&params, &sn2, &x).unwrap(), 0.0).unwrap();
        close(
            "loss_real",
            &host_f32(&real),
            &host_f32(&get(&g, "critic/loss_real")),
            SCALAR,
        );
        close(
            "loss_fake",
            &host_f32(&fake),
            &host_f32(&get(&g, "critic/loss_fake")),
            SCALAR,
        );
        let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
            let r =
                multilevel_loss(&critic.logits(&p, &sn1, &gt).map_err(exc)?, 0.8).map_err(exc)?;
            let f =
                multilevel_loss(&critic.logits(&p, &sn2, &x).map_err(exc)?, 0.0).map_err(exc)?;
            Ok(vec![r.mean(None)?.add(&f.mean(None)?)?])
        };
        let mut vg = keyed_value_and_grad(loss_fn);
        let (v, grads) = vg(params.clone(), 0).unwrap();
        close(
            "loss_d",
            &host_f32(&v[0]),
            &host_f32(&get(&g, "critic/loss_d")),
            SCALAR,
        );
        // Every head gradient against the largest head-gradient magnitude: a one-logit level's
        // bias gradient is a single reduced number (Metal accumulation error ~3% of its own size).
        let peak = grads
            .keys()
            .flat_map(|k| host_f32(&get(&g, &format!("critic/grad/{k}"))))
            .fold(0f32, |m, v| m.max(v.abs()));
        for (k, gr) in &grads {
            let got = host_f32(gr);
            let want = host_f32(&get(&g, &format!("critic/grad/{k}")));
            let d = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("grad {k}: max|Δ|={d:.3e} (head peak {peak:.3e})");
            assert!(
                d <= FWD * peak,
                "grad {k}: max|Δ|={d:.3e} > {FWD}·{peak:.3e}"
            );
        }
        // The optimizer on the reference's own gradients (the gradients are checked above).
        let reference: Params = grads
            .keys()
            .map(|k| (k.clone(), get(&g, &format!("critic/grad/{k}"))))
            .collect();
        let (norm, clipped) = clip_grads(reference, 1.0).unwrap();
        close(
            "grad norm",
            &[norm as f32],
            &host_f32(&get(&g, "critic/grad_norm")),
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
            muon_adjust_lr: mlx_gen::gen_core::iris::train::MuonAdjustLr::None,
            auto_lr: "none".into(),
            base_batch_size: 1,
        };
        let mut opt = IrisOptimizer::new(&plan, &params, |_, _| ParamRoute::AdamW).unwrap();
        let mut stepped = params.clone();
        opt.step(&mut stepped, &clipped, 1e-3).unwrap();
        for (k, p) in &stepped {
            close(
                &format!("stepped {k}"),
                &host_f32(p),
                &host_f32(&get(&g, &format!("critic/stepped/{k}"))),
                SCALAR,
            );
        }
    }
}

#[test]
fn the_generator_loss_terms_and_gradients_match_optimize_generator() {
    {
        let g = golden();
        let critic = mini_critic(&g);
        let (params, sn) = head_state(&g, "gloss/head");
        let net = Lpips::formula(VGG16_WIDTHS).unwrap();
        let x = get(&g, "gloss/x");
        let gt = get(&g, "critic/gt");
        let l2 =
            |v: &Array| -> mlx_rs::error::Result<Array> { v.subtract(&gt)?.square()?.mean(None) };
        let lp = |v: &Array| -> mlx_rs::error::Result<Array> {
            net.distance(v, &gt)
                .map_err(exc)?
                .mean(None)?
                .multiply(Array::from_f32(5.0))
        };
        let gan = |v: &Array| -> mlx_rs::error::Result<Array> {
            critic
                .generator_term(&params, &sn, v, 0.8)
                .map_err(exc)?
                .mean(None)?
                .multiply(Array::from_f32(0.5))
        };
        for (name, f) in [
            ("l2", &l2 as &dyn Fn(&Array) -> mlx_rs::error::Result<Array>),
            ("lpips", &lp),
            ("gan", &gan),
        ] {
            let (val, grad) = grad_of(&x, f);
            close(
                name,
                &[val],
                &host_f32(&get(&g, &format!("gloss/{name}"))),
                SCALAR,
            );
            close(
                &format!("grad {name}"),
                &grad,
                &host_f32(&get(&g, &format!("gloss/grad_{name}"))),
                FWD,
            );
        }
        let (val, grad) = grad_of(&x, |v| l2(v)?.add(&lp(v)?)?.add(&gan(v)?));
        close("total", &[val], &host_f32(&get(&g, "gloss/total")), SCALAR);
        close(
            "grad total",
            &grad,
            &host_f32(&get(&g, "gloss/grad_total")),
            FWD,
        );
    }
}

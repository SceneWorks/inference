//! SW-7098 gradient correctness on real weights (candle/CUDA, q4 tier). `#[ignore]`d: run only by
//! the `H3-Image training (real weights)` workflow — never by ordinary CI, which runs no model.
//!
//! The streamed, checkpointed backward (one block copied into the device window at a time, one
//! segment VJP per block) must give the gradients of the resident whole-graph `loss.backward()`
//! over the same DiT, and both must agree with a central finite difference of the loss — along the
//! gradient and per factor along a random direction — whose step stands far above the loss's own
//! noise floor (a repeat at the same factors, floored at f32's relative precision). The gate also
//! logs, for comparison with MLX, every factor's streamed-vs-resident norm ratio and cosine, the
//! cotangent at four block boundaries streamed vs resident, and two blocks' VJPs under the bounded
//! training attention against the unbounded composable one. Float32 compute, so the finite difference measures the gradient rather than
//! bf16 rounding.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image::sampler_sigmas;
use candle_gen::gen_core::train::TrainingConfig;
use candle_gen::gen_core::CancelFlag;

use gen_core_train::h3_image::training_step;

use super::step::{diagnostics, loss_and_grads, noised_rows, StepInputs};
use super::{install_adapters, LoraSet};
use crate::dit::model::image_modulation;
use crate::dit::MiniMaxH3Dit;
use crate::image_pipeline::{reference_noise, ImageLayout};

/// Streamed vs resident: the same math in a different evaluation order.
const LOSS_REL_TOL: f32 = 1e-4;
const GRAD_REL_L2_TOL: f32 = 1e-3;
/// The finite difference against the analytic directional derivative.
const FD_REL_TOL: f32 = 5e-2;

fn normal(shape: &[usize], seed: u64, scale: f64, device: &Device) -> Tensor {
    (reference_noise(seed, 7, 0, shape, device).unwrap() * scale).unwrap()
}

fn sum(t: &Tensor) -> f32 {
    t.to_dtype(DType::F32)
        .unwrap()
        .sum_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

fn rel_l2(a: &Tensor, b: &Tensor) -> f32 {
    let d = sum(&(a - b).unwrap().sqr().unwrap());
    let n = sum(&b.sqr().unwrap());
    (d / n.max(1e-30)).sqrt()
}

fn norm(t: &Tensor) -> f32 {
    sum(&t.sqr().unwrap()).sqrt()
}

fn cosine(a: &Tensor, b: &Tensor) -> f32 {
    sum(&(a * b).unwrap()) / (norm(a) * norm(b)).max(1e-30)
}

fn values(set: &LoraSet) -> HashMap<String, Tensor> {
    set.named_vars()
        .into_iter()
        .map(|(k, v)| (k, v.as_tensor().copy().unwrap()))
        .collect()
}

fn assign(set: &LoraSet, values: &HashMap<String, Tensor>) {
    for (k, v) in set.named_vars() {
        v.set(&values[&k]).unwrap();
    }
}

#[test]
#[ignore = "needs the converted q4 DiT tier (MINIMAX_H3_IMAGE_TIERS_ROOT) and CUDA; loads the q4 image DiT streamed and resident"]
fn streamed_backward_matches_resident_and_finite_differences_q4_cuda() {
    let root = std::env::var("MINIMAX_H3_IMAGE_TIERS_ROOT").unwrap_or_default();
    assert!(!root.is_empty(), "MINIMAX_H3_IMAGE_TIERS_ROOT must be set");
    let dir = std::path::Path::new(&root)
        .join("q4")
        .join(crate::image_model::IMAGE_DIT_PARTITION);
    let device = candle_gen::default_device().unwrap();
    assert!(device.is_cuda(), "this gate runs on CUDA");
    let cancel = CancelFlag::new();
    let timesteps = gen_core_h3::h3_schedule::SigmaSchedule::h3_image_distilled()
        .unwrap()
        .timesteps()
        .to_vec();

    // Two blocks' attention query and FFN output projections, rank 4, the up factors moved off
    // zero so every factor has a gradient.
    let paths: Vec<String> = [0usize, 1]
        .iter()
        .flat_map(|i| {
            [
                format!("transformer_blocks.{i}.attn.to_q"),
                format!("transformer_blocks.{i}.ff.net.2"),
            ]
        })
        .collect();
    let cfg = TrainingConfig {
        rank: 4,
        alpha: 4.0,
        seed: 7098,
        ..TrainingConfig::default()
    };
    println!(
        "[sw-7098 candle] gate weights: the q4 tier at {} (packed q4 DiT, float32 compute)",
        dir.display()
    );
    let mut streamed = MiniMaxH3Dit::load_from_dir(&dir, &Device::Cpu, DType::F32).unwrap();
    let set_s = install_adapters(&mut streamed, &paths, &cfg, &device).unwrap();
    for (i, (key, var)) in set_s.named_vars().into_iter().enumerate() {
        if key.ends_with("lora_B.weight") {
            var.set(&normal(var.dims(), 100 + i as u64, 0.02, &device))
                .unwrap();
        }
    }
    let params = values(&set_s);
    streamed.prepare_streamed(&device, 1).unwrap();
    let modulation_s = image_modulation(&mut streamed, &timesteps, false, &cancel).unwrap();

    let patch = streamed.config().patch_size;
    let layout = ImageLayout::text_to_image(16, 32, 32, patch).unwrap();
    let rows = layout.num_packed_image_rows();
    let context = normal(&[1, 16, 5120], 1, 1.0, &device);
    let step = training_step(sampler_sigmas()[2]);
    let x0 = normal(&[1, rows, 96], 2, 1.0, &device);
    let noise = normal(&[1, rows, 96], 3, 1.0, &device);
    let (packed, target) = noised_rows(&x0, &noise, step).unwrap();
    let text_s = streamed.embed_context(&context).unwrap();
    fn inputs<'a>(
        text_rows: &'a Tensor,
        layout: &'a ImageLayout,
        packed: &'a Tensor,
        target: &'a Tensor,
        step: gen_core_train::h3_image::TrainingStep,
    ) -> StepInputs<'a> {
        StepInputs {
            text_rows,
            layout,
            packed,
            target,
            step,
        }
    }

    let (loss_s, grads_s) = loss_and_grads(
        &streamed,
        &modulation_s,
        &set_s.vars,
        &inputs(&text_s, &layout, &packed, &target, step),
        false,
    )
    .unwrap();
    assert!(loss_s.is_finite());
    let grads_s: HashMap<String, Tensor> = set_s
        .named_vars()
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                grads_s
                    .get(v.as_tensor())
                    .expect("a gradient")
                    .copy()
                    .unwrap(),
            )
        })
        .collect();

    let loss_of = |set: &LoraSet, values: &HashMap<String, Tensor>| -> f32 {
        assign(set, values);
        loss_and_grads(
            &streamed,
            &modulation_s,
            &set.vars,
            &inputs(&text_s, &layout, &packed, &target, step),
            false,
        )
        .unwrap()
        .0
    };
    // The loss's own resolution: a repeat at the same factors, floored at f32's relative
    // precision over the mean the loss reduces. A finite difference whose predicted change does
    // not stand far above it measures noise, not the gradient.
    let repeat = loss_of(&set_s, &params);
    let floor = (repeat - loss_s).abs().max(loss_s.abs() * 1e-6);
    let shifted = |dir: &HashMap<String, Tensor>, by: f32| -> HashMap<String, Tensor> {
        params
            .iter()
            .map(|(k, v)| {
                let d = dir
                    .get(k)
                    .map(|d| (d * f64::from(by)).unwrap())
                    .unwrap_or_else(|| v.zeros_like().unwrap());
                (k.clone(), (v + d).unwrap())
            })
            .collect()
    };
    let dot = |a: &HashMap<String, Tensor>, b: &HashMap<String, Tensor>| -> f32 {
        a.iter()
            .filter_map(|(k, v)| b.get(k).map(|w| sum(&(v * w).unwrap())))
            .sum()
    };
    // Central difference along `dir`, its step sized so the predicted one-sided change is 1e3 x the
    // floor, capped at `cap` (a step that moves the factors by `cap · |dir|`). A change under 100 x
    // the floor is reported but not asserted: it measures noise.
    let central = |dir: &HashMap<String, Tensor>, cap: f32| -> (f32, f32, f32, bool) {
        let analytic = dot(&grads_s, dir);
        let eps = (1e3 * floor / analytic.abs().max(1e-30)).min(cap);
        let fd = (loss_of(&set_s, &shifted(dir, eps)) - loss_of(&set_s, &shifted(dir, -eps)))
            / (2.0 * eps);
        let measurable = analytic.abs() * eps >= 1e2 * floor;
        (analytic, fd, eps, measurable)
    };
    // 1. Along the analytic gradient: the predicted change is |g|^2 * eps, the step capped at 10 %
    //    of all the factors' norm.
    let (analytic, fd, eps, measurable) = central(
        &grads_s,
        0.1 * dot(&params, &params).sqrt().max(1e-6) / dot(&grads_s, &grads_s).sqrt().max(1e-30),
    );
    let fd_rel = (fd - analytic).abs() / analytic.abs().max(1e-12);
    println!("[sw-7098 candle] directional derivative along g: analytic {analytic:.6e}, finite difference {fd:.6e} at step {eps:.3e} (loss {loss_s:.6e}, noise floor {floor:.3e}, measurable {measurable}), rel {fd_rel:.3e}");
    // 2. Per factor, along a random direction inside that factor only, scaled like the factor: a
    //    factor whose analytic gradient missed a path shows here even when the total along g agrees.
    //    The direction has the factor's own norm, so a step of at most 1 moves the factor by at most
    //    its own size (the 10 % of ALL factors' norm used above left every factor under the floor on
    //    run 43355678592).
    let mut per_factor = Vec::new();
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    for (i, key) in keys.iter().enumerate() {
        let p = &params[*key];
        let r = normal(p.dims(), 500 + i as u64, 1.0, &device);
        let scale =
            (sum(&p.sqr().unwrap()).sqrt() / sum(&r.sqr().unwrap()).sqrt().max(1e-30)).max(1e-3);
        let dir: HashMap<String, Tensor> =
            [((*key).clone(), (r * f64::from(scale)).unwrap())].into();
        let (a, f, e, ok) = central(&dir, 1.0);
        let rel = (f - a).abs() / a.abs().max(1e-12);
        println!("[sw-7098 candle] per-factor FD {key}: analytic {a:.6e}, finite difference {f:.6e}, ratio {:.4e}, step {e:.3e}, measurable {ok}, rel {rel:.3e}", f / a);
        per_factor.push((key.to_string(), rel, ok));
    }
    assign(&set_s, &params);

    // 3. Per boundary, the streamed cotangent arriving at block inputs (compared with the resident
    //    run's below), and per block the bounded attention's VJP against the unbounded composable
    //    one's on the same input and output cotangent.
    let last = streamed.num_layers();
    let boundaries = [0usize, 1, 2, last - 1];
    let cot_s: Vec<Tensor> = boundaries
        .iter()
        .map(|&b| {
            let h = diagnostics::block_input(
                &streamed,
                &modulation_s,
                &inputs(&text_s, &layout, &packed, &target, step),
                b,
            )
            .unwrap();
            diagnostics::boundary_cotangent(
                &streamed,
                &modulation_s,
                &set_s.vars,
                &inputs(&text_s, &layout, &packed, &target, step),
                &h,
                b,
                false,
            )
            .unwrap()
            .copy()
            .unwrap()
        })
        .collect();
    for layer in [0usize, 1] {
        let si = inputs(&text_s, &layout, &packed, &target, step);
        let x = diagnostics::block_input(&streamed, &modulation_s, &si, layer).unwrap();
        let g = normal(x.dims(), 900 + layer as u64, 1.0, &device);
        let (gx_o, grads_o) =
            diagnostics::block_vjp(&streamed, &modulation_s, &si, layer, &x, &g, false).unwrap();
        let (gx_c, grads_c) =
            diagnostics::block_vjp(&streamed, &modulation_s, &si, layer, &x, &g, true).unwrap();
        println!("[sw-7098 candle] block {layer} input VJP, bounded vs composable attention: norm ratio {:.6e}, cosine {:.6e}", norm(&gx_o) / norm(&gx_c), cosine(&gx_o, &gx_c));
        for (key, var) in set_s.named_vars() {
            if let (Some(a), Some(b)) = (grads_o.get(var.as_tensor()), grads_c.get(var.as_tensor()))
            {
                println!("[sw-7098 candle] block {layer} {key} VJP, bounded vs composable attention: norm ratio {:.6e}, cosine {:.6e}", norm(a) / norm(b), cosine(a, b));
            }
        }
    }
    // Evidence for the MLX comparison: the base model's attention sharpness with the LoRA delta
    // zeroed (every up factor at zero), at the same fixture.
    let zeroed: HashMap<String, Tensor> = params
        .iter()
        .map(|(k, v)| {
            let v = if k.ends_with("lora_B.weight") {
                v.zeros_like().unwrap()
            } else {
                v.clone()
            };
            (k.clone(), v)
        })
        .collect();
    assign(&set_s, &zeroed);
    for layer in [0usize, 25, last - 1] {
        let si = inputs(&text_s, &layout, &packed, &target, step);
        let x = diagnostics::block_input(&streamed, &modulation_s, &si, layer).unwrap();
        let (max_abs, mean_max_prob) =
            diagnostics::attention_score_stats(&streamed, &modulation_s, &si, layer, &x).unwrap();
        println!("[sw-7098 candle] block {layer} attention, LoRA delta zeroed: max |q·k·scale| {max_abs:.6e}, mean max-probability {mean_max_prob:.6e}");
    }
    assign(&set_s, &params);
    drop((streamed, modulation_s, set_s, text_s));
    crate::dit::release_device_memory(&device).unwrap();

    let mut resident = MiniMaxH3Dit::load_from_dir(&dir, &device, DType::F32).unwrap();
    let set_r = install_adapters(&mut resident, &paths, &cfg, &device).unwrap();
    assign(&set_r, &params);
    let modulation_r = image_modulation(&mut resident, &timesteps, false, &cancel).unwrap();
    let text_r = resident.embed_context(&context).unwrap();
    let (loss_r, grads_r) = loss_and_grads(
        &resident,
        &modulation_r,
        &set_r.vars,
        &inputs(&text_r, &layout, &packed, &target, step),
        false,
    )
    .unwrap();
    let loss_rel = (loss_s - loss_r).abs() / loss_r.abs().max(1e-12);
    let mut worst = 0f32;
    for (k, v) in set_r.named_vars() {
        let (a, b) = (
            &grads_s[&k],
            grads_r.get(v.as_tensor()).expect("a gradient"),
        );
        let r = rel_l2(a, b);
        worst = worst.max(r);
        println!("[sw-7098 candle] streamed vs resident {k}: norm ratio {:.6e}, cosine {:.6e}, rel L2 {r:.3e}", norm(a) / norm(b), cosine(a, b));
    }
    for (b, cs) in boundaries.iter().zip(&cot_s) {
        let ri = inputs(&text_r, &layout, &packed, &target, step);
        let h = diagnostics::block_input(&resident, &modulation_r, &ri, *b).unwrap();
        let cr = diagnostics::boundary_cotangent(
            &resident,
            &modulation_r,
            &set_r.vars,
            &ri,
            &h,
            *b,
            false,
        )
        .unwrap();
        println!("[sw-7098 candle] boundary {b} cotangent, streamed vs resident: norm ratio {:.6e}, cosine {:.6e}, |streamed| {:.6e}", norm(cs) / norm(&cr), cosine(cs, &cr), norm(cs));
    }
    println!("[sw-7098 candle] streamed vs resident: loss {loss_s:.6e} / {loss_r:.6e} (rel {loss_rel:.3e}), worst grad rel L2 {worst:.3e}");
    assert!(loss_rel <= LOSS_REL_TOL, "loss rel {loss_rel}");
    assert!(worst <= GRAD_REL_L2_TOL, "grad rel L2 {worst}");
    assert!(
        measurable,
        "the finite difference along g stays within the loss's noise floor"
    );
    assert!(fd_rel <= FD_REL_TOL, "finite difference rel {fd_rel}");
    let measured: Vec<_> = per_factor.iter().filter(|(_, _, ok)| *ok).collect();
    assert!(
        !measured.is_empty(),
        "no factor's finite difference rose above the noise floor"
    );
    for (key, rel, _) in measured {
        assert!(*rel <= FD_REL_TOL, "{key}: finite difference rel {rel}");
    }
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4_cuda");
}

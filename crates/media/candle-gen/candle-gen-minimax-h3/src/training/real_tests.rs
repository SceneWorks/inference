//! SW-7098 gradient correctness on real weights (candle/CUDA, q4 tier). `#[ignore]`d: run only by
//! the `H3-Image training (real weights)` workflow — never by ordinary CI, which runs no model.
//!
//! The streamed, checkpointed backward (one block copied into the device window at a time, one
//! segment VJP per block) must give the gradients of the resident whole-graph `loss.backward()`
//! over the same DiT, and both must agree with a central finite difference of the loss along a
//! random direction. Float32 compute, so the finite difference measures the gradient rather than
//! bf16 rounding.

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image::sampler_sigmas;
use candle_gen::gen_core::train::TrainingConfig;
use candle_gen::gen_core::CancelFlag;

use gen_core_train::h3_image::training_step;

use super::step::{loss_and_grads, noised_rows, StepInputs};
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

    // Finite difference along a random direction, through the streamed path.
    let direction: HashMap<String, Tensor> = params
        .iter()
        .enumerate()
        .map(|(i, (k, v))| (k.clone(), normal(v.dims(), 500 + i as u64, 1.0, &device)))
        .collect();
    let dot = |a: &HashMap<String, Tensor>, b: &HashMap<String, Tensor>| -> f32 {
        a.iter().map(|(k, v)| sum(&(v * &b[k]).unwrap())).sum()
    };
    let norm = dot(&direction, &direction).sqrt();
    let eps = 1e-2 / norm;
    let loss_at = |sign: f32| -> f32 {
        let shifted: HashMap<String, Tensor> = params
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    (v + (&direction[k] * f64::from(sign * eps)).unwrap()).unwrap(),
                )
            })
            .collect();
        assign(&set_s, &shifted);
        loss_and_grads(
            &streamed,
            &modulation_s,
            &set_s.vars,
            &inputs(&text_s, &layout, &packed, &target, step),
            false,
        )
        .unwrap()
        .0
    };
    let (plus, minus) = (loss_at(1.0), loss_at(-1.0));
    let fd = (plus - minus) / (2.0 * eps);
    let analytic = dot(&grads_s, &direction);
    let fd_rel = (fd - analytic).abs() / analytic.abs().max(1e-12);
    println!("[sw-7098 candle] directional derivative: analytic {analytic:.6e}, finite difference {fd:.6e}, rel {fd_rel:.3e}");
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
    let worst = set_r
        .named_vars()
        .into_iter()
        .map(|(k, v)| {
            rel_l2(
                &grads_s[&k],
                grads_r.get(v.as_tensor()).expect("a gradient"),
            )
        })
        .fold(0f32, f32::max);
    println!("[sw-7098 candle] streamed vs resident: loss {loss_s:.6e} / {loss_r:.6e} (rel {loss_rel:.3e}), worst grad rel L2 {worst:.3e}");
    assert!(loss_rel <= LOSS_REL_TOL, "loss rel {loss_rel}");
    assert!(worst <= GRAD_REL_L2_TOL, "grad rel L2 {worst}");
    assert!(fd_rel <= FD_REL_TOL, "finite difference rel {fd_rel}");
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4_cuda");
}

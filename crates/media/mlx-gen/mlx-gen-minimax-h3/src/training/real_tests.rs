//! SW-7097 gradient correctness on real weights (MLX, q4 tier). `#[ignore]`d: run only by the
//! `minimax-h3-image-train` real-weights profile — never by ordinary CI, which runs no model.
//!
//! The streamed, checkpointed backward (one block rebuilt at a time, one `vjp` per block) must give
//! the gradients of the resident whole-graph `keyed_value_and_grad` over the same DiT, and both must
//! agree with a central finite difference of the loss along a random direction. Float32 compute,
//! so the finite difference measures the gradient rather than bf16 rounding.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::gen_core::h3_image::sampler_sigmas;
use mlx_gen::train::lora::LoraParams;
use mlx_gen::train::trainer_loop::build_adapter;
use mlx_gen::{CancelFlag, TrainingConfig};

use gen_core_train::h3_image::training_step;

use super::block_trainables;
use super::step::{
    attention, loss_and_grads, noised_rows, parts, run_block, stack_input, streamed_loss_and_grads,
    tail_loss, LossSpec, StepInputs, LOKR_DTYPE,
};
use crate::h3_image::dit::MiniMaxH3ImageDit;
use crate::h3_image::layout::ImageLayout;
use crate::h3_image::IMAGE_DIT_COMPONENT;

/// Streamed vs resident: the same math in a different evaluation order.
const LOSS_REL_TOL: f32 = 1e-4;
const GRAD_REL_L2_TOL: f32 = 1e-3;
/// The finite difference against the analytic directional derivative.
const FD_REL_TOL: f32 = 5e-2;
/// The finite-difference step is sized so the predicted loss change is this many times the
/// measured noise floor (the streamed-vs-resident loss disagreement, at least f32 rounding).
const FD_SIGNAL_OVER_NOISE: f32 = 100.0;
/// The DiT's fused (compiled) SwiGLU product against its eager body under a VJP: the same float32
/// arithmetic, so the gradients must agree to rounding.
const SWIGLU_VJP_REL_TOL: f32 = 1e-6;

/// Localizes run 43340360350's streamed-vs-resident gradient disagreement (rel L2 0.28 with the
/// losses agreeing to 2.4e-5): the block's only compiled op is the SwiGLU product, so its VJP is
/// checked against the eager body first, on seeded inputs, before any weight loads.
fn swiglu_vjp_rel() -> f32 {
    use crate::dit::layers::{swiglu_product, swiglu_product_impl};
    let gate = normal(&[1, 8, 64], 11, 2.0);
    let value = normal(&[1, 8, 64], 12, 2.0);
    let cot = normal(&[1, 8, 64], 13, 1.0);
    let run = |compiled: bool| {
        let (_, g) = mlx_rs::transforms::fallible_vjp(
            |xs: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
                Ok(vec![if compiled {
                    swiglu_product(&xs[0], &xs[1])
                        .map_err(|e| mlx_rs::error::Exception::custom(e.to_string()))?
                } else {
                    swiglu_product_impl((&xs[0], &xs[1]))?
                }])
            },
            &[gate.clone(), value.clone()],
            std::slice::from_ref(&cot),
        )
        .unwrap();
        g
    };
    let (fused, eager) = (run(true), run(false));
    rel_l2(&fused[0], &eager[0]).max(rel_l2(&fused[1], &eager[1]))
}

fn normal(shape: &[i32], seed: u64, scale: f32) -> Array {
    random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap()))
        .unwrap()
        .multiply(Array::from_f32(scale))
        .unwrap()
}

fn rel_l2(a: &Array, b: &Array) -> f32 {
    let d = a.subtract(b).unwrap().square().unwrap().sum(None).unwrap();
    let n = b.square().unwrap().sum(None).unwrap();
    (d.item::<f32>() / n.item::<f32>().max(1e-30)).sqrt()
}

fn dot(a: &LoraParams, b: &LoraParams) -> f32 {
    a.iter()
        .map(|(k, v)| v.multiply(&b[k]).unwrap().sum(None).unwrap().item::<f32>())
        .sum()
}

fn shifted(p: &LoraParams, d: &LoraParams, eps: f32) -> LoraParams {
    p.iter()
        .map(|(k, v)| {
            (
                k.clone(),
                v.add(d[k].multiply(Array::from_f32(eps)).unwrap()).unwrap(),
            )
        })
        .collect()
}

fn norm(a: &Array) -> f32 {
    a.square().unwrap().sum(None).unwrap().item::<f32>().sqrt()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let ab = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
    ab / (norm(a) * norm(b)).max(1e-30)
}

/// The resident whole-graph backward with a zero probe added at every block boundary: the probe
/// gradients are `dL/dh_l` for `l = 0..=50`, the reference for the streamed backward's cotangents.
fn resident_boundary_cotangents(
    dit: &mut MiniMaxH3ImageDit,
    modulation: &crate::h3_image::dit::Modulation,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    inputs: &StepInputs<'_>,
) -> Vec<(usize, Array)> {
    let parts = parts(dit, inputs).unwrap();
    let layers = dit.num_layers();
    let hidden = dit.config().base.hidden_size;
    let seq = inputs.layout.seq_len();
    let dtype = parts.text_rows.dtype();
    let bounded = attention(spec, seq as u64);
    let mut probed = params.clone();
    let probe_key = |l: usize| -> std::rc::Rc<str> { std::rc::Rc::from(format!("probe.{l:02}")) };
    for l in 0..=layers {
        probed.insert(
            probe_key(l),
            Array::zeros::<f32>(&[1, seq, hidden])
                .unwrap()
                .as_dtype(dtype)
                .unwrap(),
        );
    }
    let err = |e: mlx_gen::Error| mlx_rs::error::Exception::custom(e.to_string());
    let loss_fn = |p: LoraParams, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        spec.adapter.install_as(
            &mut *dit,
            &p,
            spec.alpha,
            spec.rank,
            spec.lora_dtype,
            LOKR_DTYPE,
        )?;
        let dit: &MiniMaxH3ImageDit = dit;
        let mut x = stack_input(dit, &parts, inputs.packed)
            .map_err(err)?
            .add(&p[&probe_key(0)])?;
        for (layer, block) in dit.blocks().iter().enumerate() {
            x = run_block(
                dit,
                block,
                layer,
                &x,
                modulation,
                &parts,
                inputs.step.index,
                bounded,
            )
            .map_err(err)?
            .add(&p[&probe_key(layer + 1)])?;
        }
        Ok(vec![tail_loss(
            dit, &x, modulation, &parts, inputs, spec.mae,
        )
        .map_err(err)?])
    };
    let mut vg = mlx_rs::transforms::keyed_value_and_grad(loss_fn);
    let (_, grads) = vg(probed, 0).unwrap();
    let mut out: Vec<(usize, Array)> = (0..=layers)
        .map(|l| (l, grads[&probe_key(l)].clone()))
        .collect();
    out.reverse();
    out
}

#[test]
#[ignore = "needs the converted q4 DiT tier (MINIMAX_H3_IMAGE_TIERS_ROOT); loads the q4 image DiT streamed and resident"]
fn streamed_backward_matches_resident_and_finite_differences_q4() {
    let root = std::env::var("MINIMAX_H3_IMAGE_TIERS_ROOT").unwrap_or_default();
    assert!(!root.is_empty(), "MINIMAX_H3_IMAGE_TIERS_ROOT must be set");
    let dir = std::path::Path::new(&root)
        .join("q4")
        .join(IMAGE_DIT_COMPONENT);
    let swiglu = swiglu_vjp_rel();
    println!("[sw-7097 mlx] compiled vs eager SwiGLU VJP: rel L2 {swiglu:.3e}");
    assert!(
        swiglu <= SWIGLU_VJP_REL_TOL,
        "compiled SwiGLU VJP rel L2 {swiglu}"
    );
    let cancel = CancelFlag::new();
    let mut streamed = MiniMaxH3ImageDit::load_dir_deferred(&dir, Dtype::Float32).unwrap();
    let modulation_s = streamed
        .training_modulation(false, Some((1, &cancel)))
        .unwrap();

    // The attention query and FFN output projections of the first two blocks (the longest
    // chained backward) and of blocks 47 and 49 (the shortest), rank 4, the up factors moved off
    // zero so every factor has a gradient.
    let paths: Vec<String> = [0usize, 1, 47, 49]
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
        seed: 7097,
        ..TrainingConfig::default()
    };
    let (adapter, mut params) = build_adapter(&mut streamed, &paths, &cfg).unwrap();
    for (i, (key, value)) in params.iter_mut().enumerate() {
        if key.ends_with("lora_b") {
            *value = normal(value.shape(), 100 + i as u64, 0.02);
        }
    }
    let blocks = block_trainables(&mut streamed, &paths, &params, &cfg).unwrap();

    let patch = streamed.config().base.patch_size;
    let layout = ImageLayout::t2i(16, 32, 32, patch).unwrap();
    let rows = layout.num_packed_image_rows();
    let context = normal(&[1, 16, 5120], 1, 1.0)
        .as_dtype(Dtype::Float32)
        .unwrap();
    let step = training_step(sampler_sigmas()[2]);
    let x0 = normal(&[1, rows, 96], 2, 1.0);
    let noise = normal(&[1, rows, 96], 3, 1.0);
    let (packed, target) = noised_rows(&x0, &noise, step).unwrap();
    let inputs = StepInputs {
        context: &context,
        layout: &layout,
        packed: &packed,
        target: &target,
        step,
    };
    let spec = |streamed: bool| LossSpec {
        adapter: &adapter,
        blocks: &blocks,
        alpha: cfg.alpha,
        rank: cfg.rank as f32,
        mae: false,
        lora_dtype: None,
        streamed,
        chunked_attention: true,
    };

    let mut cot_s = Vec::new();
    let parts_s = parts(&mut streamed, &inputs).unwrap();
    println!(
        "[sw-7097 mlx] activation dtypes: text rows {:?}, stack input {:?}",
        parts_s.text_rows.dtype(),
        stack_input(&streamed, &parts_s, &packed).unwrap().dtype()
    );
    let (loss_s, grads_s) = streamed_loss_and_grads(
        &mut streamed,
        &modulation_s,
        &parts_s,
        &params,
        &spec(true),
        &inputs,
        Some(&mut cot_s),
    )
    .unwrap();
    assert!(loss_s.is_finite());
    assert_eq!(grads_s.len(), params.len());
    let (again, _) =
        loss_and_grads(&mut streamed, &modulation_s, &params, &spec(true), &inputs).unwrap();
    println!("[sw-7097 mlx] streamed loss repeat: {loss_s:.9e} / {again:.9e}");

    // The resident reference, with every block boundary's cotangent.
    let mut resident = MiniMaxH3ImageDit::load_dir(&dir, Dtype::Float32).unwrap();
    let modulation_r = resident.training_modulation(false, None).unwrap();
    let (loss_r, grads_r) =
        loss_and_grads(&mut resident, &modulation_r, &params, &spec(false), &inputs).unwrap();
    let cot_r =
        resident_boundary_cotangents(&mut resident, &modulation_r, &params, &spec(false), &inputs);
    drop(resident);
    mlx_gen::residency::drain_allocator_cache();

    // (a) per factor: the norm ratio and the cosine.
    let mut keys: Vec<_> = grads_s.keys().cloned().collect();
    keys.sort();
    for key in &keys {
        let (gs, gr) = (&grads_s[key], &grads_r[key]);
        println!(
            "[sw-7097 mlx] factor {key}: |g_s|/|g_r| {:.6}, cosine {:.6}, rel L2 {:.3e}",
            norm(gs) / norm(gr).max(1e-30),
            cosine(gs, gr),
            rel_l2(gs, gr)
        );
    }
    // (b) per block boundary, from the stack output down to block 0's input.
    for ((ls, s), (lr, r)) in cot_s.iter().zip(&cot_r) {
        assert_eq!(ls, lr);
        println!(
            "[sw-7097 mlx] boundary h_{ls:02}: cotangent rel L2 {:.3e}, |s|/|r| {:.6}, cosine {:.6}",
            rel_l2(s, r),
            norm(s) / norm(r).max(1e-30),
            cosine(s, r)
        );
    }

    // The finite difference along a random direction, its step sized from the noise floor.
    let direction: LoraParams = params
        .iter()
        .enumerate()
        .map(|(i, (k, v))| (k.clone(), normal(v.shape(), 500 + i as u64, 1.0)))
        .collect();
    let analytic = dot(&grads_s, &direction);
    // The noise floor is the loss's own rounding within one evaluation path (a repeat of the same
    // streamed evaluation, at least f32 epsilon relative). The streamed-vs-resident loss gap is NOT
    // a noise floor: run 43347516939 sized the step from it (1.49e-3 -> step 3.7 per unit of a
    // N(0, 1) direction, ~180x the 0.02-scale factors), where the loss is dominated by its even
    // (curvature) part and the central difference cancels to ~0.
    let noise_floor = (again - loss_s).abs().max(loss_s.abs() * 1e-6);
    let eps = FD_SIGNAL_OVER_NOISE * noise_floor / analytic.abs().max(1e-30);
    let fd_at = |streamed: &mut MiniMaxH3ImageDit, eps: f32| {
        let (plus, _) = loss_and_grads(
            streamed,
            &modulation_s,
            &shifted(&params, &direction, eps),
            &spec(true),
            &inputs,
        )
        .unwrap();
        let (minus, _) = loss_and_grads(
            streamed,
            &modulation_s,
            &shifted(&params, &direction, -eps),
            &spec(true),
            &inputs,
        )
        .unwrap();
        println!(
            "[sw-7097 mlx] fd step {eps:.3e}: L+ {plus:.9e} L0 {loss_s:.9e} L- {minus:.9e}, second difference {:.3e}",
            plus - 2.0 * loss_s + minus
        );
        (plus - minus) / (2.0 * eps)
    };
    let fd = fd_at(&mut streamed, eps);
    let fd_half = fd_at(&mut streamed, eps / 2.0);
    println!("[sw-7097 mlx] finite difference at step {eps:.3e}: {fd:.6e}; at half the step: {fd_half:.6e}");
    let fd_rel = (fd - analytic).abs() / analytic.abs().max(1e-12);
    println!("[sw-7097 mlx] directional derivative: analytic {analytic:.6e}, finite difference {fd:.6e} at step {eps:.3e} (noise floor {noise_floor:.3e}), rel {fd_rel:.3e}");

    let loss_rel = (loss_s - loss_r).abs() / loss_r.abs().max(1e-12);
    let worst = keys
        .iter()
        .map(|k| rel_l2(&grads_s[k], &grads_r[k]))
        .fold(0f32, f32::max);
    println!("[sw-7097 mlx] streamed vs resident: loss {loss_s:.6e} / {loss_r:.6e} (rel {loss_rel:.3e}), worst grad rel L2 {worst:.3e}");
    assert!(loss_rel <= LOSS_REL_TOL, "loss rel {loss_rel}");
    assert!(worst <= GRAD_REL_L2_TOL, "grad rel L2 {worst}");
    assert!(fd_rel <= FD_REL_TOL, "finite difference rel {fd_rel}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4");
}

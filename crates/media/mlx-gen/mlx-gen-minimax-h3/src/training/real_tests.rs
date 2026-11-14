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
    attention, block_vjp, install_concrete, loss_and_grads, materialize, noised_rows, parts,
    run_block, stack_input, streamed_loss, streamed_loss_and_grads, tail_loss, BlockHost,
    FactorGrads, LossSpec, StepInputs, LOKR_DTYPE,
};
use crate::h3_image::dit::MiniMaxH3ImageDit;
use crate::h3_image::layout::ImageLayout;
use crate::h3_image::IMAGE_DIT_COMPONENT;

/// Streamed vs resident: the same math in a different evaluation order.
const LOSS_REL_TOL: f32 = 1e-4;
const GRAD_REL_L2_TOL: f32 = 1e-3;
/// The finite difference against the analytic directional derivative.
const FD_REL_TOL: f32 = 5e-2;
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

/// `g` restricted to the factors whose keys start with `prefix` (zeros elsewhere).
fn restrict(g: &LoraParams, prefix: &str) -> LoraParams {
    g.iter()
        .map(|(k, v)| {
            let kept = if k.starts_with(prefix) {
                v.clone()
            } else {
                Array::zeros::<f32>(v.shape()).unwrap()
            };
            (k.clone(), kept)
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

/// One block's backward two ways on the same block, input and output cotangent: the trainer's
/// [`block_vjp`] and plain `keyed_value_and_grad` autodiff of `Σ block(input, θ) · cotangent`.
#[allow(clippy::too_many_arguments)]
fn block_backward_both(
    dit: &MiniMaxH3ImageDit,
    block: &crate::dit::block::DitBlock,
    layer: usize,
    trainables: &super::step::BlockTrainables,
    params: &LoraParams,
    spec: &LossSpec<'_>,
    modulation: &crate::h3_image::dit::Modulation,
    parts: &crate::h3_image::dit::DenoiserParts,
    step: usize,
    bounded: mlx_gen::attention::BoundedAttention<'_>,
    input: &Array,
    cot: &Array,
) -> ((Array, FactorGrads), (Array, FactorGrads)) {
    let mut a = block.clone();
    let via_vjp = block_vjp(
        dit,
        &mut a,
        layer,
        trainables,
        params,
        spec,
        modulation,
        parts,
        step,
        bounded,
        input.clone(),
        cot,
    )
    .unwrap();
    let mut b = block.clone();
    let input_key: std::rc::Rc<str> = std::rc::Rc::from("input");
    let mut map: LoraParams = trainables
        .keys
        .iter()
        .map(|k| (k.clone(), params[k].clone()))
        .collect();
    map.insert(input_key.clone(), input.clone());
    let err = |e: mlx_gen::Error| mlx_rs::error::Exception::custom(e.to_string());
    let loss_fn = |p: LoraParams, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        if let Some(adapter) = &trainables.adapter {
            adapter.install_as(
                &mut BlockHost {
                    index: layer,
                    block: &mut b,
                },
                &p,
                spec.alpha,
                spec.rank,
                spec.lora_dtype,
                LOKR_DTYPE,
            )?;
        }
        let out = run_block(
            dit,
            &b,
            layer,
            &p["input"],
            modulation,
            parts,
            step,
            bounded,
        )
        .map_err(err)?;
        Ok(vec![out.multiply(cot)?.sum(None)?])
    };
    let mut vg = mlx_rs::transforms::keyed_value_and_grad(loss_fn);
    let (_, grads) = vg(map, 0).unwrap();
    let via_autodiff = (
        grads[&input_key].clone(),
        trainables
            .keys
            .iter()
            .map(|k| (k.clone(), grads[k].clone()))
            .collect(),
    );
    (via_vjp, via_autodiff)
}

fn compare_block(label: &str, x: &(Array, FactorGrads), y: &(Array, FactorGrads)) {
    println!(
        "[sw-7097 mlx] {label}: input cotangent rel L2 {:.3e}, |x|/|y| {:.6}, cosine {:.6}",
        rel_l2(&x.0, &y.0),
        norm(&x.0) / norm(&y.0).max(1e-30),
        cosine(&x.0, &y.0)
    );
    for ((k, gx), (_, gy)) in x.1.iter().zip(&y.1) {
        println!(
            "[sw-7097 mlx] {label}: {k} rel L2 {:.3e}, |x|/|y| {:.6}, cosine {:.6}",
            rel_l2(gx, gy),
            norm(gx) / norm(gy).max(1e-30),
            cosine(gx, gy)
        );
    }
}

/// The finite-difference arbiter's measurability rule: a central difference is measurable when
/// `|L+ − L−|` is at least this many times the loss's own noise floor.
const FD_MEASURABLE_OVER_NOISE: f32 = 100.0;
/// The halving ladder's length bound (it stops earlier at the first unmeasurable rung).
const FD_MAX_RUNGS: usize = 24;

/// One rung of the finite-difference ladder at step `h` along `d`: the central difference over the
/// realized prediction, `ρ(h) = (L+ − L−) / g·(δ+ − δ−)`, `δ± = (θ ± h·d) − θ` evaluated in f32.
/// `ρ → 1` as `h → 0` for a correct `g`; a central difference's error is `c·h² + O(h⁴)`.
struct Rung {
    h: f32,
    ratio: f32,
    measurable: bool,
}

/// The Richardson verdict along one direction: the extrapolated ratio from the smallest adjacent
/// pair of measurable rungs, `R = (4·ρ(h/2) − ρ(h)) / 3`, which cancels the `h²` term, and the
/// previous pair's `R` for the convergence check. `None` when fewer than two measurable rungs exist.
struct Richardson {
    extrapolated: f32,
    pair: (f32, f32),
    previous: Option<f32>,
}

/// Walk the halving ladder from `h0` until a rung is unmeasurable (or [`FD_MAX_RUNGS`]), logging
/// every rung, then extrapolate. `loss_at(p)` evaluates the untraced streamed loss at factors `p`.
#[allow(clippy::too_many_arguments)]
fn richardson(
    label: &str,
    params: &LoraParams,
    grads: &LoraParams,
    dir: &LoraParams,
    l0: f32,
    floor: f32,
    h0: f32,
    loss_at: &mut dyn FnMut(&LoraParams) -> f32,
) -> Option<Richardson> {
    let mut rungs: Vec<Rung> = Vec::new();
    let mut h = h0;
    for _ in 0..FD_MAX_RUNGS {
        let realized = |eps: f32| {
            let moved = shifted(params, dir, eps);
            let delta: LoraParams = moved
                .iter()
                .map(|(k, v)| (k.clone(), v.subtract(&params[k]).unwrap()))
                .collect();
            let predicted = dot(grads, &delta);
            (moved, predicted)
        };
        let (plus, pred_plus) = realized(h);
        let (minus, pred_minus) = realized(-h);
        let (lp, lm) = (loss_at(&plus), loss_at(&minus));
        let (up, down) = (lp - l0, l0 - lm);
        let asymmetry = (up - down).abs() / up.abs().max(down.abs()).max(1e-30);
        let predicted = pred_plus - pred_minus;
        let ratio = (lp - lm) / predicted;
        let measurable = (lp - lm).abs() >= FD_MEASURABLE_OVER_NOISE * floor;
        println!("[sw-7097 mlx] FD {label}: step {h:.3e}, L+ - L- {:.6e}, realized prediction {predicted:.6e}, ratio {ratio:.6}, signal/noise {:.1}, asymmetry {asymmetry:.3e}, measurable {measurable}", lp - lm, (lp - lm).abs() / floor);
        rungs.push(Rung {
            h,
            ratio,
            measurable,
        });
        if !measurable {
            break;
        }
        h /= 2.0;
    }
    let measured: Vec<&Rung> = rungs.iter().filter(|r| r.measurable).collect();
    let estimates: Vec<(f32, (f32, f32))> = measured
        .windows(2)
        .map(|w| ((4.0 * w[1].ratio - w[0].ratio) / 3.0, (w[0].h, w[1].h)))
        .collect();
    for (r, (a, b)) in &estimates {
        println!(
            "[sw-7097 mlx] FD {label}: Richardson over steps ({a:.3e}, {b:.3e}): ratio {r:.6}"
        );
    }
    let (extrapolated, pair) = *estimates.last()?;
    let previous = estimates.len().checked_sub(2).map(|i| estimates[i].0);
    match previous {
        Some(p) => println!(
            "[sw-7097 mlx] FD {label}: Richardson convergence |R - R_prev| / |R| {:.3e}",
            (extrapolated - p).abs() / extrapolated.abs().max(1e-30)
        ),
        None => println!("[sw-7097 mlx] FD {label}: one measurable pair; no convergence check"),
    }
    Some(Richardson {
        extrapolated,
        pair,
        previous,
    })
}

/// The first step of the ladder: where the predicted change `|g·d|·h` is 1e3x the noise floor,
/// capped at 5% of the factors' norm.
fn first_step(params: &LoraParams, grads: &LoraParams, dir: &LoraParams, floor: f32) -> f32 {
    let a = dot(grads, dir);
    let cap = 0.05 * dot(params, params).sqrt() / dot(dir, dir).sqrt().max(1e-30);
    (1e3 * floor / a.abs().max(1e-30)).min(cap)
}

/// The Richardson verdict's relative error against the analytic derivative (`ρ → 1`), or a failure
/// naming the direction when no two adjacent rungs are both measurable.
fn verdict(label: &str, r: Option<Richardson>) -> f32 {
    let r = r.unwrap_or_else(|| {
        panic!(
            "FD {label}: no two adjacent finite-difference steps are both measurable (|L+ - L-| >= \
             {FD_MEASURABLE_OVER_NOISE}x the noise floor); the loss is not resolvable along it"
        )
    });
    let rel = (r.extrapolated - 1.0).abs();
    println!(
        "[sw-7097 mlx] FD {label}: verdict over steps ({:.3e}, {:.3e}): Richardson / analytic {:.6}, rel {rel:.3e} (previous pair {:?})",
        r.pair.0, r.pair.1, r.extrapolated, r.previous
    );
    rel
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
    // Precondition: IEEE float32 GEMMs. On NAX MLX runs float32 GEMMs as TF32 unless
    // MLX_ENABLE_TF32=0 (v0.32.0 backend/metal/matmul.cpp:915-917); block 49's logits reach ~5e3
    // (learned max |norm_k| 55.5), where TF32's ~5e-4 relative error is O(1) in the logits and the
    // loss jitters by ~1e-3 — no gradient-scale finite difference resolves.
    {
        let a = normal(&[272, 5376], 71, 1.0);
        let b = normal(&[5376, 7168], 72, 0.02);
        let rel = rel_l2(
            &a.matmul(&b).unwrap(),
            &a.matmul_device(&b, mlx_rs::StreamOrDevice::cpu()).unwrap(),
        );
        println!(
            "[sw-7097 mlx] MLX_ENABLE_TF32={}; float32 GEMM default stream vs CPU stream rel L2 {rel:.3e}",
            std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "<unset: MLX default 1>".into())
        );
        assert!(
            rel <= 1e-5,
            "float32 GEMMs here are not IEEE single precision (rel L2 {rel:.3e} against the CPU \
             stream): run this gate with MLX_ENABLE_TF32=0"
        );
    }
    // Every block forward below — the trainer's and the test's own — runs the eager SwiGLU.
    let _eager = crate::dit::layers::EagerSwigluScope::enter();
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
    // Seeded by sorted key, not by HashMap iteration order (randomized per process).
    let mut keys: Vec<_> = params.keys().cloned().collect();
    keys.sort();
    for (i, key) in keys.iter().enumerate() {
        if key.ends_with("lora_b") {
            let shape = params[key].shape().to_vec();
            params.insert(key.clone(), normal(&shape, 100 + i as u64, 0.02));
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

    // The trainer's streamed backward, with the cotangent at every block boundary.
    let mut cot_s = Vec::new();
    let parts_s = parts(&mut streamed, &inputs).unwrap();
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

    // The resident reference: independent whole-graph autodiff over a DiT whose every tensor is
    // forced before the graph is built (the trainer's own resident load), run twice in-process.
    let mut resident = MiniMaxH3ImageDit::load_dir_materialized(&dir, Dtype::Float32).unwrap();
    let modulation_r = resident.training_modulation(false, None).unwrap();
    let (loss_r, grads_r) =
        loss_and_grads(&mut resident, &modulation_r, &params, &spec(false), &inputs).unwrap();
    let (loss_r2, grads_r2) =
        loss_and_grads(&mut resident, &modulation_r, &params, &spec(false), &inputs).unwrap();
    let repeat_worst = grads_r
        .iter()
        .map(|(k, g)| rel_l2(&grads_r2[k], g))
        .fold(0f32, f32::max);
    println!("[sw-7097 mlx] resident repeat: loss {loss_r:.9e} / {loss_r2:.9e}, worst grad rel L2 {repeat_worst:.3e}");

    // Activation and loss compares, and the single-block VJP checks (blocks 0 and 49: the same
    // input and output cotangent through the trainer's per-block vjp and through plain autodiff,
    // on the streamed and on the resident block).
    {
        let parts_r = parts(&mut resident, &inputs).unwrap();
        let bounded = attention(&spec(true), layout.seq_len() as u64);
        let step_index = inputs.step.index;
        spec(false)
            .adapter
            .install_as(
                &mut resident,
                &params,
                cfg.alpha,
                cfg.rank as f32,
                None,
                LOKR_DTYPE,
            )
            .unwrap();
        let mut h = stack_input(&resident, &parts_r, &packed).unwrap();
        let mut inputs_at = vec![h.clone()];
        for (layer, block) in resident.blocks().iter().enumerate() {
            h = run_block(
                &resident,
                block,
                layer,
                &h,
                &modulation_r,
                &parts_r,
                step_index,
                bounded,
            )
            .unwrap();
            mlx_rs::transforms::eval([&h]).unwrap();
            inputs_at.push(h.clone());
        }
        let tail = tail_loss(&resident, &h, &modulation_r, &parts_r, &inputs, false)
            .unwrap()
            .item::<f32>();
        println!("[sw-7097 mlx] loss: untraced resident forward + tail {tail:.9e}, streamed (traced tail) {loss_s:.9e}, resident (traced graph) {loss_r:.9e}");
        let mut hs = stack_input(&streamed, &parts_s, &packed).unwrap();
        let mut worst_h = (0usize, 0f32);
        for layer in 0..=streamed.num_layers() {
            let rel = rel_l2(&hs, &inputs_at[layer]);
            if rel > worst_h.1 {
                worst_h = (layer, rel);
            }
            if layer == streamed.num_layers() {
                break;
            }
            let mut block = materialize(&streamed, layer).unwrap();
            install_concrete(&mut block, layer, &blocks[layer], &params, &spec(true)).unwrap();
            hs = run_block(
                &streamed,
                &block,
                layer,
                &hs,
                &modulation_s,
                &parts_s,
                step_index,
                bounded,
            )
            .unwrap();
            mlx_rs::transforms::eval([&hs]).unwrap();
        }
        println!(
            "[sw-7097 mlx] activations h_00..h_{:02}: worst streamed vs resident rel L2 {:.3e} at h_{:02}",
            streamed.num_layers(),
            worst_h.1,
            worst_h.0
        );
        for layer in [49usize, 0] {
            let cot = &cot_s.iter().find(|(l, _)| *l == layer + 1).unwrap().1;
            let input = &inputs_at[layer];
            let streamed_block = materialize(&streamed, layer).unwrap();
            let resident_block = resident.blocks()[layer].clone();
            let (s_vjp, s_auto) = block_backward_both(
                &streamed,
                &streamed_block,
                layer,
                &blocks[layer],
                &params,
                &spec(true),
                &modulation_s,
                &parts_s,
                step_index,
                bounded,
                input,
                cot,
            );
            let (r_vjp, r_auto) = block_backward_both(
                &resident,
                &resident_block,
                layer,
                &blocks[layer],
                &params,
                &spec(false),
                &modulation_r,
                &parts_r,
                step_index,
                bounded,
                input,
                cot,
            );
            compare_block(
                &format!("block {layer} streamed block: vjp vs autodiff"),
                &s_vjp,
                &s_auto,
            );
            compare_block(
                &format!("block {layer} resident block: vjp vs autodiff"),
                &r_vjp,
                &r_auto,
            );
            compare_block(
                &format!("block {layer} vjp: streamed vs resident block"),
                &s_vjp,
                &r_vjp,
            );
        }
    }
    drop(resident);
    mlx_gen::residency::drain_allocator_cache();

    // Streamed vs resident, per factor and overall.
    for key in &keys {
        let (gs, gr) = (&grads_s[key], &grads_r[key]);
        println!(
            "[sw-7097 mlx] factor {key}: |g_s|/|g_r| {:.6}, cosine {:.6}, rel L2 {:.3e}",
            norm(gs) / norm(gr).max(1e-30),
            cosine(gs, gr),
            rel_l2(gs, gr)
        );
    }
    let loss_rel = (loss_s - loss_r).abs() / loss_r.abs().max(1e-12);
    let worst = keys
        .iter()
        .map(|k| rel_l2(&grads_s[k], &grads_r[k]))
        .fold(0f32, f32::max);
    println!("[sw-7097 mlx] streamed vs resident: loss {loss_s:.6e} / {loss_r:.6e} (rel {loss_rel:.3e}), worst grad rel L2 {worst:.3e}");
    assert!(loss_rel <= LOSS_REL_TOL, "loss rel {loss_rel}");
    assert!(worst <= GRAD_REL_L2_TOL, "grad rel L2 {worst}");

    // ── The finite-difference arbiter: Richardson extrapolation on the halving ladder ──────────
    // Untraced streamed loss (`streamed_loss`, the function the trainer reports and differentiates);
    // the noise floor is that loss's own repeat spread, at least 8 f32 ulps of it.
    let mut loss_at = |p: &LoraParams| {
        streamed_loss(&mut streamed, &modulation_s, p, &spec(true), &inputs).unwrap()
    };
    let floor_of = |l0: f32, again: f32| (again - l0).abs().max(l0.abs() * 8.0 * f32::EPSILON);

    // 1. The seeded point (B ~ N(0, 0.02)): the full gradient direction asserts; each factor
    //    pair's subspace is logged.
    let l0 = loss_at(&params);
    let floor = floor_of(l0, loss_at(&params));
    let seeded = richardson(
        "seeded, along g",
        &params,
        &grads_s,
        &grads_s,
        l0,
        floor,
        first_step(&params, &grads_s, &grads_s, floor),
        &mut loss_at,
    );
    for prefix in [
        "transformer_blocks.49.attn.to_q.",
        "transformer_blocks.49.ff.net.2.",
        "transformer_blocks.0.attn.to_q.",
        "transformer_blocks.0.ff.net.2.",
    ] {
        let dir = restrict(&grads_s, prefix);
        let r = richardson(
            &format!("seeded, along g[{prefix}*]"),
            &params,
            &grads_s,
            &dir,
            l0,
            floor,
            first_step(&params, &grads_s, &dir, floor),
            &mut loss_at,
        );
        if let Some(r) = r {
            println!(
                "[sw-7097 mlx] FD seeded, along g[{prefix}*]: Richardson / analytic {:.6} (logged, not asserted)",
                r.extrapolated
            );
        }
    }

    // 2. The standard LoRA init point (every lora_B zero, as the Candle gate and training start):
    //    the gradient there, then the full gradient direction, asserted.
    let zeroed: LoraParams = params
        .iter()
        .map(|(k, v)| {
            let kept = if k.ends_with("lora_b") {
                Array::zeros::<f32>(v.shape()).unwrap()
            } else {
                v.clone()
            };
            (k.clone(), kept)
        })
        .collect();
    let (loss_z, grads_z) =
        loss_and_grads(&mut streamed, &modulation_s, &zeroed, &spec(true), &inputs).unwrap();
    let mut loss_at = |p: &LoraParams| {
        streamed_loss(&mut streamed, &modulation_s, p, &spec(true), &inputs).unwrap()
    };
    let l0_z = loss_at(&zeroed);
    let floor_z = floor_of(l0_z, loss_at(&zeroed));
    println!(
        "[sw-7097 mlx] B = 0: loss {loss_z:.9e} (untraced {l0_z:.9e}), |g| {:.6e}",
        dot(&grads_z, &grads_z).sqrt()
    );
    let at_init = richardson(
        "B = 0, along g",
        &zeroed,
        &grads_z,
        &grads_z,
        l0_z,
        floor_z,
        first_step(&zeroed, &grads_z, &grads_z, floor_z),
        &mut loss_at,
    );

    let seeded_rel = verdict("seeded, along g", seeded);
    let init_rel = verdict("B = 0, along g", at_init);
    assert!(
        seeded_rel <= FD_REL_TOL,
        "seeded point: Richardson finite difference rel {seeded_rel} exceeds {FD_REL_TOL}"
    );
    assert!(
        init_rel <= FD_REL_TOL,
        "B = 0: Richardson finite difference rel {init_rel} exceeds {FD_REL_TOL}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4");
}

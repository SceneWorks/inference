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

/// The first-order loss change of one step of `unit` along `g`: `unit · |g|²`.
fn analytic_unit(g: &LoraParams, unit: f32) -> f32 {
    unit * dot(g, g)
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

/// `(output rel L2, input-VJP rel L2)` of block 49's q4 `attn.to_q` base against the same weights
/// dequantized to a dense float32 matmul.
fn qmm_vjp_check(dit: &MiniMaxH3ImageDit) -> (f32, f32) {
    use mlx_gen::adapters::AdaptableHost;
    let mut block = super::step::materialize(dit, 49).unwrap();
    let lin = block.adaptable_mut(&["attn", "to_q"]).unwrap();
    let (wq, sc, bi, _, gs, bits) = lin.quantized_params().expect("a q4 leaf");
    let (wq, sc, bi) = (wq.clone(), sc.clone(), bi.clone());
    let dense = mlx_rs::ops::dequantize(&wq, &sc, &bi, gs, bits)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let in_f = dense.shape()[1];
    let x = normal(&[1, 8, in_f], 41, 1.0);
    let cot = normal(&[1, 8, dense.shape()[0]], 42, 1.0);
    let packed = mlx_rs::transforms::vjp(
        |xs: &[Array]| {
            vec![mlx_rs::ops::quantized_matmul(&xs[0], &wq, &sc, &bi, true, gs, bits).unwrap()]
        },
        std::slice::from_ref(&x),
        std::slice::from_ref(&cot),
    )
    .unwrap();
    let plain = mlx_rs::transforms::vjp(
        |xs: &[Array]| vec![mlx_rs::ops::matmul(&xs[0], dense.t()).unwrap()],
        &[x],
        &[cot],
    )
    .unwrap();
    // Linearity of the packed matmul in its input at a small step: an exact float32 kernel gives
    // qmm(x + t·d) − qmm(x) = t·qmm(d) to rounding; a kernel that rounds or quantizes its input
    // does not (a piecewise forward in the activations would make the loss rough in every factor
    // upstream of a q4 projection).
    let qmm = |a: &Array| mlx_rs::ops::quantized_matmul(a, &wq, &sc, &bi, true, gs, bits).unwrap();
    let x2 = normal(&[1, 8, in_f], 43, 1.0);
    let d = normal(&[1, 8, in_f], 44, 1.0);
    for t in [1e-2f32, 1e-3, 1e-4] {
        let moved = x2.add(d.multiply(Array::from_f32(t)).unwrap()).unwrap();
        let lhs = qmm(&moved).subtract(qmm(&x2)).unwrap();
        let rhs = qmm(&d).multiply(Array::from_f32(t)).unwrap();
        println!(
            "[sw-7097 mlx] q4 quantized_matmul linearity at t={t:.0e}: rel L2 {:.3e}",
            rel_l2(&lhs, &rhs)
        );
    }
    (
        rel_l2(&packed.0[0].as_dtype(Dtype::Float32).unwrap(), &plain.0[0]),
        rel_l2(&packed.1[0].as_dtype(Dtype::Float32).unwrap(), &plain.1[0]),
    )
}

/// `(output rel L2, input-VJP rel L2)` of MLX's fused `rms_norm` against `x · rsqrt(mean(x²) + ε) · w`
/// over the per-head q/k-norm shape.
fn rms_norm_vjp_check() -> (f32, f32) {
    let x = normal(&[1, 8, 56, 128], 51, 3.0);
    let w = normal(&[128], 52, 1.0);
    let cot = normal(&[1, 8, 56, 128], 53, 1.0);
    let eps = 1e-5f32;
    let fused = mlx_rs::transforms::vjp(
        |xs: &[Array]| vec![mlx_rs::fast::rms_norm(&xs[0], &w, eps).unwrap()],
        std::slice::from_ref(&x),
        std::slice::from_ref(&cot),
    )
    .unwrap();
    let explicit = mlx_rs::transforms::vjp(
        |xs: &[Array]| {
            let ms = xs[0].square().unwrap().mean_axis(-1, true).unwrap();
            let inv = mlx_rs::ops::rsqrt(ms.add(Array::from_f32(eps)).unwrap()).unwrap();
            vec![xs[0].multiply(&inv).unwrap().multiply(&w).unwrap()]
        },
        &[x],
        &[cot],
    )
    .unwrap();
    (
        rel_l2(&fused.0[0], &explicit.0[0]),
        rel_l2(&fused.1[0], &explicit.1[0]),
    )
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

/// Per tier: the base model's (no LoRA) blocks 0, 25 and 49 attention — max |score|, mean
/// max-probability, the qk-norm weights' largest magnitude — walked through the streamed forward.
/// Returns block 49's attention input and the block itself.
fn tier_walk(
    label: &str,
    dir: &std::path::Path,
    inputs: &StepInputs<'_>,
    cancel: &CancelFlag,
) -> (Array, crate::dit::block::DitBlock) {
    let _eager = crate::dit::layers::EagerSwigluScope::enter();
    let mut dit = MiniMaxH3ImageDit::load_dir_deferred(dir, Dtype::Float32).unwrap();
    let modulation = dit.training_modulation(false, Some((1, cancel))).unwrap();
    let parts = parts(&mut dit, inputs).unwrap();
    let step = inputs.step.index;
    let bounded = mlx_gen::attention::BoundedAttention::UNBOUNDED;
    let mut h = stack_input(&dit, &parts, inputs.packed).unwrap();
    let mut last = None;
    for layer in 0..dit.num_layers() {
        let block = materialize(&dit, layer).unwrap();
        if [0usize, 25, 49].contains(&layer) {
            let m = modulation.cache.modulation(layer).unwrap();
            let sc = block
                .attention_scores(&h, m, &parts.adaln_indices[step], dit.rope(), &parts.tables)
                .unwrap();
            let max_abs = sc.abs().unwrap().max(None).unwrap().item::<f32>();
            let maxp = mlx_rs::ops::softmax_axis(&sc, -1, true)
                .unwrap()
                .max_axis(-1, None)
                .unwrap()
                .mean(None)
                .unwrap()
                .item::<f32>();
            let (nq, nk) = block.attention().qk_norm_weights();
            let wmax = |a: &Array| a.abs().unwrap().max(None).unwrap().item::<f32>();
            println!("[sw-7097 mlx] tier {label}: block {layer} max |score| {max_abs:.4e}, mean max-probability {maxp:.4}, max |norm_q| {:.4e}, max |norm_k| {:.4e}", wmax(nq), wmax(nk));
            if layer == 49 {
                // The same scores on the CPU stream (IEEE float32): the GPU's error in the logits.
                mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
                let sc_cpu = block
                    .attention_scores(&h, m, &parts.adaln_indices[step], dit.rope(), &parts.tables)
                    .unwrap();
                mlx_rs::transforms::eval([&sc_cpu]).unwrap();
                mlx_rs::Device::set_default(&mlx_rs::Device::gpu());
                let err = sc
                    .subtract(&sc_cpu)
                    .unwrap()
                    .abs()
                    .unwrap()
                    .max(None)
                    .unwrap()
                    .item::<f32>();
                println!("[sw-7097 mlx] tier {label}: block 49 scores, default stream vs CPU stream: rel L2 {:.3e}, max |Δlogit| {err:.3e}", rel_l2(&sc, &sc_cpu));
            }
            if layer == 49 {
                let input = block
                    .attention_input(&h, m, &parts.adaln_indices[step])
                    .unwrap();
                mlx_rs::transforms::eval([&input]).unwrap();
                // Which channels make the top score: q ⊙ k at the arg-max (head, query, key).
                let (q, k) = block
                    .attention()
                    .qk(&input, Some((dit.rope(), &parts.tables)))
                    .unwrap();
                let s = sc.shape().to_vec();
                let flat = mlx_rs::ops::indexing::argmax(sc.abs().unwrap(), None)
                    .unwrap()
                    .item::<u32>() as i32;
                let (hh, i, j) = (flat / (s[2] * s[3]), (flat / s[3]) % s[2], flat % s[3]);
                let row = |t: &Array, r: i32| {
                    crate::tensor::slice_axis(
                        &crate::tensor::slice_axis(t, 1, hh, hh + 1).unwrap(),
                        2,
                        r,
                        r + 1,
                    )
                    .unwrap()
                    .reshape(&[-1])
                    .unwrap()
                };
                let contrib = row(&q, i).multiply(row(&k, j)).unwrap();
                let total = contrib.sum(None).unwrap().item::<f32>();
                let mut values: Vec<f32> = contrib.as_slice::<f32>().to_vec();
                let mut idx: Vec<usize> = (0..values.len()).collect();
                idx.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()));
                let top: Vec<String> = idx
                    .iter()
                    .take(5)
                    .map(|&c| format!("ch{c} {:.3e}", values[c]))
                    .collect();
                values.clear();
                println!("[sw-7097 mlx] tier {label}: block 49 top |score| at head {hh}, query {i}, key {j}: q·k {total:.4e} (unscaled); largest channel terms [{}]", top.join(", "));
                last = Some((input, block.clone()));
            }
        }
        h = run_block(&dit, &block, layer, &h, &modulation, &parts, step, bounded).unwrap();
        mlx_rs::transforms::eval([&h]).unwrap();
    }
    last.expect("block 49 was walked")
}

/// The q4 tier's block 49 against the dense bf16 tier's ([`tier_walk`] on both) and, on the same
/// input, the q4 vs bf16 `to_q` / `to_k` projection outputs and weights. Logs only.
fn tier_attention_report(q4_dir: &std::path::Path, inputs: &StepInputs<'_>, cancel: &CancelFlag) {
    let snapshot = std::env::var("MINIMAX_H3_IMAGE_SNAPSHOT").unwrap_or_default();
    let bf16_dir = std::path::Path::new(&snapshot).join(IMAGE_DIT_COMPONENT);
    if !bf16_dir.join("config.json").is_file() {
        println!(
            "[sw-7097 mlx] tier comparison skipped: no bf16 tier at {}",
            bf16_dir.display()
        );
        return;
    }
    let (bf16_input, mut bf16_block) = tier_walk("bf16", &bf16_dir, inputs, cancel);
    let (_, mut q4_block) = tier_walk("q4", q4_dir, inputs, cancel);
    use mlx_gen::adapters::AdaptableHost;
    for leaf in ["to_q", "to_k"] {
        let dense = bf16_block.adaptable_mut(&["attn", leaf]).unwrap().clone();
        let packed = q4_block.adaptable_mut(&["attn", leaf]).unwrap().clone();
        let yd = dense.forward(&bf16_input).unwrap();
        let yq = packed.forward(&bf16_input).unwrap();
        let wd = match dense.quantized_params() {
            None => dense
                .dense_weight()
                .map(|(w, _)| w.as_dtype(Dtype::Float32).unwrap()),
            Some(_) => None,
        };
        let wq = packed.quantized_params().map(|(w, sc, bi, _, gs, bits)| {
            mlx_rs::ops::dequantize(w, sc, bi, gs, bits)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
        });
        let weights = match (&wd, &wq) {
            (Some(d), Some(q)) => {
                let err = q.subtract(d).unwrap().abs().unwrap();
                format!(
                    "weight rel L2 {:.3e}, max |w| {:.4e}, max |w_q4 - w| {:.4e}",
                    rel_l2(q, d),
                    d.abs().unwrap().max(None).unwrap().item::<f32>(),
                    err.max(None).unwrap().item::<f32>()
                )
            }
            _ => "weights not comparable (tier layout)".to_string(),
        };
        println!(
            "[sw-7097 mlx] block 49 {leaf}: q4 vs bf16 projection output rel L2 {:.3e} on the bf16 path's input; {weights}",
            rel_l2(&yq, &yd)
        );
    }
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
    // Untraced, at the FFN's real width: the compiled kernel against its eager body. Nonzero means
    // a forward evaluated through the kernel and differentiated through the traced graph are two
    // functions — what the trainer's EagerSwigluScope removes.
    {
        use crate::dit::layers::{swiglu_product, swiglu_product_impl};
        let gate = normal(&[1, 272, 14336], 61, 4.0);
        let value = normal(&[1, 272, 14336], 62, 4.0);
        let fused = swiglu_product(&gate, &value).unwrap();
        let eager = swiglu_product_impl((&gate, &value)).unwrap();
        println!(
            "[sw-7097 mlx] compiled vs eager SwiGLU forward (untraced): rel L2 {:.3e}",
            rel_l2(&fused, &eager)
        );
    }
    // Precondition: IEEE float32 GEMMs (see the GEMM probe below for why). On NAX MLX runs float32
    // GEMMs as TF32 unless MLX_ENABLE_TF32=0 (v0.32.0 backend/metal/matmul.cpp:915-917).
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
    // Seeded by sorted key, not by HashMap iteration order (which is randomized per process and
    // made earlier runs' parameters — and so their losses and gradients — differ run to run).
    let mut keys_sorted: Vec<_> = params.keys().cloned().collect();
    keys_sorted.sort();
    for (i, key) in keys_sorted.iter().enumerate() {
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

    // Is block 49's near-hard attention the model's or the q4 tier's? The dense bf16 tier against
    // the q4 tier, base model (no LoRA), same fixture — run 43516258270: q4 block 49 max |score|
    // 5.18e3 against ~20-27 at blocks 0 and 25.
    tier_attention_report(&dir, &inputs, &cancel);

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
    // The resident reference: independent whole-graph autodiff over a DiT whose every tensor is
    // forced before the graph is built (the trainer's own resident load).
    let mut resident = MiniMaxH3ImageDit::load_dir_materialized(&dir, Dtype::Float32).unwrap();
    let modulation_r = resident.training_modulation(false, None).unwrap();
    let (loss_r, grads_r) =
        loss_and_grads(&mut resident, &modulation_r, &params, &spec(false), &inputs).unwrap();
    // (a) The resident forward + backward again in the same process.
    let (loss_r2, grads_r2) =
        loss_and_grads(&mut resident, &modulation_r, &params, &spec(false), &inputs).unwrap();
    let repeat_worst = grads_r
        .iter()
        .map(|(k, g)| rel_l2(&grads_r2[k], g))
        .fold(0f32, f32::max);
    println!("[sw-7097 mlx] resident repeat: loss {loss_r:.9e} / {loss_r2:.9e}, worst grad rel L2 {repeat_worst:.3e}");
    let cot_r =
        resident_boundary_cotangents(&mut resident, &modulation_r, &params, &spec(false), &inputs);
    // (b) The post-stack head on its own: the untraced tail loss over the resident's untraced h_50
    // against both paths' reported (traced) losses.
    {
        let parts_r = parts(&mut resident, &inputs).unwrap();
        let bounded = attention(&spec(false), layout.seq_len() as u64);
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
        for (layer, block) in resident.blocks().iter().enumerate() {
            h = run_block(
                &resident,
                block,
                layer,
                &h,
                &modulation_r,
                &parts_r,
                inputs.step.index,
                bounded,
            )
            .unwrap();
            mlx_rs::transforms::eval([&h]).unwrap();
        }
        let tail = tail_loss(&resident, &h, &modulation_r, &parts_r, &inputs, false)
            .unwrap()
            .item::<f32>();
        println!("[sw-7097 mlx] loss: untraced resident forward + tail {tail:.9e}, streamed (traced tail) {loss_s:.9e}, resident (traced graph) {loss_r:.9e}");
    }
    // Single-block isolation, blocks 0 and 49: the same input and output cotangent through the
    // trainer's per-block vjp and through plain autodiff, on the streamed block (rebuilt from the
    // tier) and on the resident one — separating "differentiation mechanism" from "block/state".
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
        // (b) The resident untraced forward's block inputs against the streamed forward's (the
        // trainer's: one rebuilt block at a time, concrete factors installed).
        let mut hs = stack_input(&streamed, &parts_s, &packed).unwrap();
        let mut worst_h = (0usize, 0f32);
        for layer in 0..=streamed.num_layers() {
            let rel = rel_l2(&hs, &inputs_at[layer]);
            if layer % 7 == 0 || layer >= 47 || rel > worst_h.1 {
                println!(
                    "[sw-7097 mlx] activation h_{layer:02}: streamed vs resident rel L2 {rel:.3e}"
                );
            }
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
            "[sw-7097 mlx] activations: worst streamed vs resident rel L2 {:.3e} at h_{:02}",
            worst_h.1, worst_h.0
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

        // Block 49's attention under to_q perturbations along g (restricted to that factor pair):
        // the scores' magnitude and float32 resolution, softmax saturation, and whether any row's
        // winning key (argmax) changes on the 17-point grid — with the loss on the same grid. A
        // value-dependent discrete choice in the attention path would show as argmax flips that
        // line up with loss steps; a smooth path shows none and constant loss differences.
        // Is the near-hard attention the base model's or the fixture's? Blocks 0, 25 and 49's
        // score magnitude and mean max-probability with the LoRA up factors zeroed (the base model
        // plus a zero delta, as training starts), with B seeded only where the Candle gate seeds it
        // (blocks 0-1, the same N(0, 0.02)), and with this gate's seed — plus the seeded block 49
        // to_q delta's size against its weight.
        {
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
            for (label, variant) in [
                ("B = 0 (base)", &zeroed),
                ("B ~ N(0, 0.02) (gate seed)", &params),
            ] {
                spec(false)
                    .adapter
                    .install_as(
                        &mut resident,
                        variant,
                        cfg.alpha,
                        cfg.rank as f32,
                        None,
                        LOKR_DTYPE,
                    )
                    .unwrap();
                let mut h = stack_input(&resident, &parts_r, &packed).unwrap();
                for (layer, block) in resident.blocks().iter().enumerate() {
                    if [0usize, 25, 49].contains(&layer) {
                        let sc = block
                            .attention_scores(
                                &h,
                                modulation_r.cache.modulation(layer).unwrap(),
                                &parts_r.adaln_indices[step_index],
                                resident.rope(),
                                &parts_r.tables,
                            )
                            .unwrap();
                        let max_abs = sc.abs().unwrap().max(None).unwrap().item::<f32>();
                        let maxp = mlx_rs::ops::softmax_axis(&sc, -1, true)
                            .unwrap()
                            .max_axis(-1, None)
                            .unwrap()
                            .mean(None)
                            .unwrap()
                            .item::<f32>();
                        println!("[sw-7097 mlx] scores, {label}: block {layer} max |score| {max_abs:.4e}, mean max-probability {maxp:.4}");
                    }
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
                }
            }
            // Restore the gate's factors on the resident blocks.
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
            use mlx_gen::adapters::AdaptableHost;
            let mut blk = resident.blocks()[49].clone();
            let lin = blk.adaptable_mut(&["attn", "to_q"]).unwrap();
            let (wq, sc, bi, _, gs, bits) = lin.quantized_params().expect("a q4 leaf");
            let w = mlx_rs::ops::dequantize(wq, sc, bi, gs, bits)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let a = &params[&std::rc::Rc::from("transformer_blocks.49.attn.to_q.lora_a")];
            let b = &params[&std::rc::Rc::from("transformer_blocks.49.attn.to_q.lora_b")];
            let delta = b
                .matmul(a)
                .unwrap()
                .multiply(Array::from_f32(cfg.alpha / cfg.rank as f32))
                .unwrap();
            println!(
                "[sw-7097 mlx] block 49 to_q: |ΔW|/|W| {:.3e} with the gate seed (|ΔW| {:.3e}, |W| {:.3e})",
                norm(&delta) / norm(&w).max(1e-30),
                norm(&delta),
                norm(&w)
            );
        }
        let dir49 = restrict(&grads_s, "transformer_blocks.49.attn.to_q.");
        let a49 = dot(&grads_s, &dir49);
        let unit = 1e-4 * loss_s.abs() / a49.abs().max(1e-30);
        let base_block = resident.blocks()[49].clone();
        let scores_at = |p: &LoraParams| -> Array {
            let mut blk = base_block.clone();
            if let Some(adapter) = &blocks[49].adapter {
                adapter
                    .install_as(
                        &mut BlockHost {
                            index: 49,
                            block: &mut blk,
                        },
                        p,
                        cfg.alpha,
                        cfg.rank as f32,
                        None,
                        LOKR_DTYPE,
                    )
                    .unwrap();
            }
            let s = blk
                .attention_scores(
                    &inputs_at[49],
                    modulation_r.cache.modulation(49).unwrap(),
                    &parts_r.adaln_indices[step_index],
                    resident.rope(),
                    &parts_r.tables,
                )
                .unwrap();
            mlx_rs::transforms::eval([&s]).unwrap();
            s
        };
        let s0 = scores_at(&params);
        let max_abs = s0.abs().unwrap().max(None).unwrap().item::<f32>();
        let probs0 = mlx_rs::ops::softmax_axis(&s0, -1, true).unwrap();
        let maxp = probs0.max_axis(-1, None).unwrap();
        let saturated = maxp
            .gt(Array::from_f32(0.99))
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .mean(None)
            .unwrap()
            .item::<f32>();
        println!(
            "[sw-7097 mlx] block 49 attention: max |score| {max_abs:.4e} (float32 spacing there {:.3e}), mean max-probability {:.4}, rows with max-probability > 0.99: {:.2}%",
            max_abs * f32::EPSILON,
            maxp.mean(None).unwrap().item::<f32>(),
            100.0 * saturated
        );
        let arg0 = mlx_rs::ops::indexing::argmax_axis(&s0, -1, None).unwrap();
        let mut losses = Vec::new();
        for k in -8i32..=8 {
            let p = shifted(&params, &dir49, unit * k as f32);
            let sk = scores_at(&p);
            let argk = mlx_rs::ops::indexing::argmax_axis(&sk, -1, None).unwrap();
            let flips = argk
                .ne(&arg0)
                .unwrap()
                .as_dtype(Dtype::Int32)
                .unwrap()
                .sum(None)
                .unwrap()
                .item::<i32>();
            let dprob = mlx_rs::ops::softmax_axis(&sk, -1, true)
                .unwrap()
                .subtract(&probs0)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            let l = streamed_loss(&mut streamed, &modulation_s, &p, &spec(true), &inputs).unwrap();
            losses.push(l);
            println!("[sw-7097 mlx] block 49 to_q grid k={k:+}: step {:.3e}, argmax flips {flips} of {} rows, max |Δprob| {dprob:.3e}, loss {l:.9e}", unit * k as f32, arg0.size());
        }
        let diffs: Vec<String> = losses
            .windows(2)
            .map(|w| format!("{:.3e}", w[1] - w[0]))
            .collect();
        println!(
            "[sw-7097 mlx] block 49 to_q grid: successive loss differences [{}]; smooth expectation {:.3e}",
            diffs.join(", "),
            unit * a49
        );
    }
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

    // The op-level suspects first: the q4 quantized matmul's input VJP against the dequantized
    // dense matmul, and MLX's fused RMSNorm VJP against its explicit formula.
    let (qmm_out, qmm_grad) = qmm_vjp_check(&streamed);
    println!("[sw-7097 mlx] q4 quantized_matmul vs dequantized dense: output rel L2 {qmm_out:.3e}, input VJP rel L2 {qmm_grad:.3e}");
    let (rms_out, rms_grad) = rms_norm_vjp_check();
    println!("[sw-7097 mlx] fast rms_norm vs explicit: output rel L2 {rms_out:.3e}, input VJP rel L2 {rms_grad:.3e}");

    let gemm_rel;
    // Where smoothness goes: (i) dtypes on the LoRA path; (ii) the realized perturbation of one
    // factor against the requested h·g; (iii) this GPU's float32 GEMM against the CPU stream's at a
    // block projection's shape; (iv) the loss along g on a fine grid around θ — its successive
    // differences show whether L is smooth or stepped at the FD's scale.
    {
        let key = keys
            .iter()
            .find(|k| k.ends_with("49.attn.to_q.lora_b"))
            .unwrap()
            .clone();
        let h = 1e-3f32;
        let moved = shifted(&params, &grads_s, h);
        let realized = moved[&key].subtract(&params[&key]).unwrap();
        let wanted = grads_s[&key].multiply(Array::from_f32(h)).unwrap();
        println!(
            "[sw-7097 mlx] dtypes: params {:?}, grads {:?}, perturbed {:?}; realized vs requested perturbation of {key}: rel L2 {:.3e}",
            params[&key].dtype(),
            grads_s[&key].dtype(),
            moved[&key].dtype(),
            rel_l2(&realized, &wanted)
        );
        let a = normal(&[272, 5376], 71, 1.0);
        let b = normal(&[5376, 7168], 72, 0.02);
        let gpu = a.matmul(&b).unwrap();
        let cpu = a.matmul_device(&b, mlx_rs::StreamOrDevice::cpu()).unwrap();
        gemm_rel = rel_l2(&gpu, &cpu);
        // MLX runs float32 GEMMs on NAX as TF32 unless MLX_ENABLE_TF32=0 (v0.32.0
        // backend/metal/matmul.cpp:915-917). Block 49's logits reach ~5e3 (max |norm_k| 55.5), so a
        // TF32 GEMM's ~5e-4 relative error is O(1) in the logits there and the loss jitters by
        // ~1e-3 — no gradient-scale finite difference resolves. The gate measures the IEEE float32
        // function, so it requires IEEE float32 GEMMs (the CPU stream's to within 1e-5).
        println!(
            "[sw-7097 mlx] GPU float32 GEMM precision (evidence): [272,5376]x[5376,7168] default (GPU) stream vs CPU stream rel L2 {gemm_rel:.3e} — IEEE float32 accumulation gives ~1e-6; TF32-class is ~5e-4"
        );
        let unit = 0.183 / 16.0;
        let mut row = Vec::new();
        for k in -8i32..=8 {
            let l = loss_and_grads(
                &mut streamed,
                &modulation_s,
                &shifted(&params, &grads_s, unit * k as f32),
                &spec(true),
                &inputs,
            )
            .unwrap()
            .0;
            row.push(l);
        }
        let diffs: Vec<String> = row
            .windows(2)
            .map(|w| format!("{:.3e}", w[1] - w[0]))
            .collect();
        println!(
            "[sw-7097 mlx] loss along g at steps k*{unit:.3e}, k=-8..8: successive differences [{}] (a smooth L has them ~constant ≈ {:.3e})",
            diffs.join(", "),
            analytic_unit(&grads_s, unit)
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

    // ── The finite-difference arbiter (GPU) ─────────────────────────────────────────────────────
    // Central differences of the untraced streamed loss (`streamed_loss`, the function the trainer
    // reports and differentiates) with the REALIZED perturbation: δ± = (θ ± h·d) − θ evaluated in
    // f32, the prediction g·(δ+ − δ−). The step starts where the predicted change is 1e3x the loss's
    // own rounding and halves until the one-sided changes agree within 10%; only a step that is
    // measurable AND passes that check adjudicates, and none qualifying fails the gate.
    let l0 = streamed_loss(&mut streamed, &modulation_s, &params, &spec(true), &inputs).unwrap();
    let l0_again =
        streamed_loss(&mut streamed, &modulation_s, &params, &spec(true), &inputs).unwrap();
    let floor = (l0_again - l0).abs().max(l0.abs() * 8.0 * f32::EPSILON);
    let params_norm = dot(&params, &params).sqrt();
    let realized = |dir: &LoraParams, eps: f32| -> (LoraParams, f32) {
        let moved = shifted(&params, dir, eps);
        let delta: LoraParams = moved
            .iter()
            .map(|(k, v)| (k.clone(), v.subtract(&params[k]).unwrap()))
            .collect();
        let predicted = dot(&grads_s, &delta);
        (moved, predicted)
    };
    let mut verdicts = Vec::new();
    let mut central = |streamed: &mut MiniMaxH3ImageDit, dir: &LoraParams, label: &str| {
        let a = dot(&grads_s, dir);
        let cap = 0.05 * params_norm / dot(dir, dir).sqrt().max(1e-30);
        let mut eps = (1e3 * floor / a.abs().max(1e-30)).min(cap);
        let mut verdict = None;
        for _ in 0..12 {
            let (plus, pred_plus) = realized(dir, eps);
            let (minus, pred_minus) = realized(dir, -eps);
            let lp = streamed_loss(streamed, &modulation_s, &plus, &spec(true), &inputs).unwrap();
            let lm = streamed_loss(streamed, &modulation_s, &minus, &spec(true), &inputs).unwrap();
            let (up, down) = (lp - l0, l0 - lm);
            let asymmetry = (up - down).abs() / up.abs().max(down.abs()).max(1e-30);
            let measurable = (lp - lm).abs() >= 1e2 * floor;
            let predicted = pred_plus - pred_minus;
            let rel = ((lp - lm) - predicted).abs() / predicted.abs().max(1e-30);
            let ok = measurable && asymmetry <= 0.1;
            println!("[sw-7097 mlx] FD along {label}: L+ - L- {:.6e} vs realized prediction {predicted:.6e} (rel {rel:.3e}) at step {eps:.3e}; L+ - L0 {up:.3e}, L0 - L- {down:.3e}, asymmetry {asymmetry:.3e}, floor {floor:.3e}, measurable {measurable}, qualifies {ok}", lp - lm);
            if ok {
                verdict = Some(rel);
                break;
            }
            if !measurable {
                break;
            }
            eps /= 2.0;
        }
        verdicts.push((label.to_string(), verdict));
    };
    central(&mut streamed, &grads_s, "g_streamed");
    for prefix in [
        "transformer_blocks.49.attn.to_q.",
        "transformer_blocks.49.ff.net.2.",
        "transformer_blocks.0.attn.to_q.",
        "transformer_blocks.0.ff.net.2.",
    ] {
        central(
            &mut streamed,
            &restrict(&grads_s, prefix),
            &format!("g_streamed[{prefix}*]"),
        );
    }
    let full = verdicts[0].1;
    assert!(
        full.is_some(),
        "no finite-difference step along g_streamed is both measurable and within the curvature \
         check; the loss is not resolvable at any tried step"
    );
    let fd_rel = full.unwrap_or(f32::INFINITY);
    assert!(fd_rel <= FD_REL_TOL, "finite difference rel {fd_rel}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED training::real_tests::streamed_backward_matches_resident_and_finite_differences_q4");
}

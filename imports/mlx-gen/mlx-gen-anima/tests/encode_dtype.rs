//! sc-10577 — the CHEAP, deterministic guard for the configurable fp32 encode path.
//!
//! The isolation MEASUREMENT (fp32-TE vs bf16-TE stage-7 residual) is real-weights + Metal + SLOW, so
//! it is `#[ignore]`d in `parity_real_weights.rs`. This file is its CI-runnable counterpart: it builds a
//! TINY synthetic Qwen3 tower + `AnimaTextConditioner` (fixed-seed weights, no licensed snapshot) on the
//! CI Metal GPU and proves the plumbing the measurement relies on:
//!   1. the shipped path records **bf16** as its compute dtype (`compute_dtype() == Bfloat16`) —
//!      default UNCHANGED by this story (the only change to `forward` is the hardcoded `Bfloat16` →
//!      `self.compute_dtype`, byte-identical when that field is bf16);
//!   2. the opt-in path is genuinely **fp32** end-to-end (`from_weights_dtype(.., Float32)` with fp32
//!      weights → `compute_dtype() == Float32`, forward output **Float32** with no bf16 anywhere);
//!   3. the two are the SAME computation at different weight precision (they agree within a loose
//!      bound) — i.e. fp32 is not a divergent code path, it is the reference-precision variant;
//!   4. the fp32 opt-in genuinely uses fp32 **weights**, not merely fp32-typed activations: at equal
//!      (fp32) compute, the fp32-weight tower diverges from a bf16-weight tower by MORE than bf16
//!      rounding. This is the failure-capable half of guard 2 — the `output == Float32` assertion
//!      alone is confounded (see NOTE: the f32 RoPE tables promote a bf16-weight tower's output to f32
//!      too), so on its own it would still pass if `forward` ignored `compute_dtype` and silently kept
//!      bf16 weights. Guard 4 pins the property down: if the fp32 path collapsed to bf16 weights the
//!      two towers would be the SAME fp32 computation → bit-identical output → rel-L2 == 0 → RED.
//!
//! NOTE on dtypes: the shared text RoPE tables (`TextRope::forward`) are f32, and MLX promotes mixed
//! bf16×f32 ops — so the bf16 tower's INTERNAL activations (and hence its output dtype) promote to f32
//! even though its WEIGHTS are bf16. The bf16-vs-fp32 distinction this story isolates is therefore the
//! weight PRECISION (bf16 on disk vs the reference's fp32 upcast), which is exactly what the fp32-TE
//! variant changes. So these guards assert `compute_dtype()`, the fp32 output dtype, AND — because that
//! output-dtype check is confounded by the same promotion — a weight-PRECISION discriminator (guard 4)
//! that a bf16-weight tower could not pass; they do NOT assert the promotion-dependent bf16 output
//! dtype.

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_anima::conditioner::AnimaTextConditioner;
use mlx_gen_anima::config::{ConditionerConfig, Qwen3Config};
use mlx_gen_anima::text_encoder::AnimaQwen3;

/// FNV-1a of the key → a stable, order-independent per-tensor seed (same idea as `parity_goldens.rs`).
fn key_seed(key: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in key.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Deterministic LCG in `[-1, 1)`.
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed & 0x7fff_ffff;
    (0..n)
        .map(|_| {
            s = (s.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fff_ffff;
            (s as f64 / 2147483647.0 * 2.0 - 1.0) as f32
        })
        .collect()
}

/// Insert a synthetic tensor at `dtype`: norm/scale weights near 1.0, everything else small.
fn synth(w: &mut Weights, key: &str, shape: &[i32], dtype: Dtype) {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let raw = lcg(n, key_seed(key));
    let data: Vec<f32> = if key.ends_with("norm.weight") {
        raw.iter().map(|&v| 1.0 + 0.1 * v).collect()
    } else {
        raw.iter().map(|&v| 0.3 * v).collect()
    };
    w.insert(
        key,
        Array::from_slice(&data, shape).as_dtype(dtype).unwrap(),
    );
}

/// A tiny Qwen3 config: hidden 6, 1 layer, GQA 2/1, head_dim 4 (even for the half-split RoPE), vocab 16.
/// hidden (6) ≠ heads·head_dim (8) — the same asymmetry the real Qwen3-0.6B has (1024 vs 2048).
fn tiny_qwen3_cfg() -> Qwen3Config {
    Qwen3Config {
        vocab_size: 16,
        hidden_size: 6,
        n_layers: 1,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 4,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    }
}

/// Build the full dense weight set `AnimaQwen3::from_weights_dtype` requires (prefix `"model"`), at `dtype`.
fn tiny_qwen3_weights(cfg: &Qwen3Config, dtype: Dtype) -> Weights {
    let h = cfg.hidden_size;
    let qd = cfg.n_heads * cfg.head_dim; // 8
    let kvd = cfg.n_kv_heads * cfg.head_dim; // 4
    let inter = 3 * h; // 18
    let mut w = Weights::empty();
    synth(
        &mut w,
        "model.embed_tokens.weight",
        &[cfg.vocab_size, h],
        dtype,
    );
    for i in 0..cfg.n_layers {
        let b = format!("model.layers.{i}");
        synth(&mut w, &format!("{b}.input_layernorm.weight"), &[h], dtype);
        synth(
            &mut w,
            &format!("{b}.post_attention_layernorm.weight"),
            &[h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.q_proj.weight"),
            &[qd, h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.k_proj.weight"),
            &[kvd, h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.v_proj.weight"),
            &[kvd, h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.o_proj.weight"),
            &[h, qd],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.q_norm.weight"),
            &[cfg.head_dim],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.self_attn.k_norm.weight"),
            &[cfg.head_dim],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.mlp.gate_proj.weight"),
            &[inter, h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.mlp.up_proj.weight"),
            &[inter, h],
            dtype,
        );
        synth(
            &mut w,
            &format!("{b}.mlp.down_proj.weight"),
            &[h, inter],
            dtype,
        );
    }
    synth(&mut w, "model.norm.weight", &[h], dtype);
    w
}

/// A tiny conditioner config: model_dim 8, 1 block, 2 heads (head_dim 4), T5 vocab 8, min_seq 8.
fn tiny_cond_cfg() -> ConditionerConfig {
    ConditionerConfig {
        source_dim: 8,
        target_dim: 8,
        model_dim: 8,
        num_layers: 1,
        num_attention_heads: 2,
        mlp_ratio: 2.0,
        target_vocab_size: 8,
        min_sequence_length: 8,
        rope_theta: 10000.0,
        norm_eps: 1e-6,
    }
}

/// Build the dense weight set `AnimaTextConditioner::from_weights` requires (prefix `"net.llm_adapter"`).
fn tiny_cond_weights(cfg: &ConditionerConfig, dtype: Dtype) -> Weights {
    let d = cfg.model_dim as i32;
    let src = cfg.source_dim as i32;
    let hd = cfg.head_dim() as i32;
    let qd = cfg.num_attention_heads as i32 * hd; // heads·head_dim
    let inter = (cfg.mlp_ratio * cfg.model_dim as f32) as i32;
    let p = "net.llm_adapter";
    let mut w = Weights::empty();
    synth(
        &mut w,
        &format!("{p}.embed.weight"),
        &[cfg.target_vocab_size as i32, d],
        dtype,
    );
    for i in 0..cfg.num_layers {
        let b = format!("{p}.blocks.{i}");
        // self-attn: q/k/v from the target hidden (model_dim); cross-attn: k/v from the source.
        for (attn, kv_in) in [("self_attn", d), ("cross_attn", src)] {
            synth(
                &mut w,
                &format!("{b}.{attn}.q_proj.weight"),
                &[qd, d],
                dtype,
            );
            synth(
                &mut w,
                &format!("{b}.{attn}.k_proj.weight"),
                &[qd, kv_in],
                dtype,
            );
            synth(
                &mut w,
                &format!("{b}.{attn}.v_proj.weight"),
                &[qd, kv_in],
                dtype,
            );
            synth(
                &mut w,
                &format!("{b}.{attn}.o_proj.weight"),
                &[d, qd],
                dtype,
            );
            synth(&mut w, &format!("{b}.{attn}.q_norm.weight"), &[hd], dtype);
            synth(&mut w, &format!("{b}.{attn}.k_norm.weight"), &[hd], dtype);
        }
        synth(&mut w, &format!("{b}.norm_self_attn.weight"), &[d], dtype);
        synth(&mut w, &format!("{b}.norm_cross_attn.weight"), &[d], dtype);
        synth(&mut w, &format!("{b}.norm_mlp.weight"), &[d], dtype);
        synth(&mut w, &format!("{b}.mlp.0.weight"), &[inter, d], dtype);
        synth(&mut w, &format!("{b}.mlp.0.bias"), &[inter], dtype);
        synth(&mut w, &format!("{b}.mlp.2.weight"), &[d, inter], dtype);
        synth(&mut w, &format!("{b}.mlp.2.bias"), &[d], dtype);
    }
    synth(&mut w, &format!("{p}.out_proj.weight"), &[d, d], dtype);
    synth(&mut w, &format!("{p}.out_proj.bias"), &[d], dtype);
    synth(&mut w, &format!("{p}.norm.weight"), &[d], dtype);
    w
}

/// Flatten an array to f64 (for the bf16-vs-fp32 relative-L2 sanity check).
fn flat_f64(a: &Array) -> Vec<f64> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&x| x as f64)
        .collect()
}

fn rel_l2(got: &[f64], reference: &[f64]) -> f64 {
    assert_eq!(got.len(), reference.len());
    let (mut num, mut den) = (0f64, 0f64);
    for (&g, &r) in got.iter().zip(reference) {
        num += (g - r) * (g - r);
        den += r * r;
    }
    (num / den.max(1e-12)).sqrt()
}

#[test]
fn qwen3_tower_default_is_bf16_and_fp32_opt_in_is_fp32() {
    let cfg = tiny_qwen3_cfg();
    // Default constructor → bf16 (the shipped path, UNCHANGED by sc-10577).
    let te_bf16 =
        AnimaQwen3::from_weights(&tiny_qwen3_weights(&cfg, Dtype::Bfloat16), "model", &cfg)
            .unwrap();
    assert_eq!(
        te_bf16.compute_dtype(),
        Dtype::Bfloat16,
        "from_weights must default to bf16"
    );
    // Opt-in fp32 reference variant (must supply fp32 weights — matmul dtype must match).
    let te_fp32 = AnimaQwen3::from_weights_dtype(
        &tiny_qwen3_weights(&cfg, Dtype::Float32),
        "model",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();
    assert_eq!(te_fp32.compute_dtype(), Dtype::Float32);

    // Forward on tiny ids with a real padding mask (last token masked out).
    let ids = Array::from_slice(&[1i32, 5, 9], &[1, 3]);
    let mask = Array::from_slice(&[1i32, 1, 0], &[1, 3]);
    let out_bf16 = te_bf16.forward(&ids, &mask).unwrap();
    let out_fp32 = te_fp32.forward(&ids, &mask).unwrap();
    // The fp32 tower has NO bf16 anywhere (fp32 weights + fp32 activations), so its output stays fp32 —
    // this is the "actually fp32" guard. (The bf16 tower's output promotes to f32 via the f32 RoPE
    // tables even though its weights are bf16, so its output dtype is not asserted — see the note above.)
    assert_eq!(
        out_fp32.dtype(),
        Dtype::Float32,
        "fp32 tower output must be Float32"
    );
    assert_eq!(out_bf16.shape(), out_fp32.shape());

    // Same computation, different weight precision: the outputs must agree within a loose bound —
    // proving the fp32 path is the reference-precision variant of the SAME tower, not a divergent branch.
    let r = rel_l2(&flat_f64(&out_bf16), &flat_f64(&out_fp32));
    println!("[encode_dtype] qwen3 tower bf16-weight vs fp32-weight rel-L2 = {r:.3e}");
    assert!(
        r < 1.5e-1,
        "bf16 and fp32 towers must be the same computation (rel-L2 {r:.3e})"
    );

    // Guard 4 — the DISCRIMINATING guard for "fp32 variant uses fp32 WEIGHTS". The
    // `out_fp32.dtype() == Float32` check above is necessary but NOT sufficient: the shared f32 RoPE
    // tables promote activations to f32, so a bf16-weight tower's output is Float32 too — that check
    // would still pass if `forward` ignored `compute_dtype` and kept bf16 weights. Build a reference
    // tower with the SAME fp32 compute but bf16-ROUNDED weights (the ONLY difference from `te_fp32` is
    // weight precision) and require the outputs to diverge by more than bf16 rounding. If the fp32
    // path silently used bf16 weights, the two towers would be one bit-identical fp32 computation
    // (rel-L2 == 0) and this assertion goes RED.
    let te_bf16_weights = AnimaQwen3::from_weights_dtype(
        &tiny_qwen3_weights(&cfg, Dtype::Bfloat16),
        "model",
        &cfg,
        Dtype::Float32,
    )
    .unwrap();
    let out_bf16_weights = te_bf16_weights.forward(&ids, &mask).unwrap();
    let r_weight = rel_l2(&flat_f64(&out_fp32), &flat_f64(&out_bf16_weights));
    println!(
        "[encode_dtype] qwen3 fp32-weight vs bf16-weight (both fp32 compute) rel-L2 = {r_weight:.3e}"
    );
    assert!(
        r_weight > 5e-4,
        "fp32 variant must genuinely use fp32 WEIGHTS: at equal (fp32) compute the fp32-weight and \
         bf16-weight towers must differ by more than bf16 rounding (rel-L2 {r_weight:.3e}); a value \
         near 0 means the fp32 path collapsed to bf16 weights"
    );
}

#[test]
fn conditioner_honors_dtype_arg_bf16_and_fp32() {
    let cfg = tiny_cond_cfg();
    let cond_bf16 = AnimaTextConditioner::from_weights(
        &tiny_cond_weights(&cfg, Dtype::Bfloat16),
        "net.llm_adapter",
        cfg,
    )
    .unwrap();
    let cond_fp32 = AnimaTextConditioner::from_weights(
        &tiny_cond_weights(&cfg, Dtype::Float32),
        "net.llm_adapter",
        cfg,
    )
    .unwrap();

    // source_hidden [1, 2, source_dim]; target_ids [1, 3] (< min_seq 8 → right-padded to 8).
    let source_bf16 = Array::from_slice(&lcg(16, 42), &[1, 2, 8])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let source_fp32 = source_bf16.as_dtype(Dtype::Float32).unwrap();
    let target = Array::from_slice(&[1i32, 2, 3], &[1, 3]);

    let out_bf16 = cond_bf16
        .forward(&source_bf16, &target, Dtype::Bfloat16)
        .unwrap();
    let out_fp32 = cond_fp32
        .forward(&source_fp32, &target, Dtype::Float32)
        .unwrap();
    // fp32 source + fp32 weights + Float32 dtype arg → the fp32 conditioner output is Float32 (the
    // "actually fp32" guard). The bf16 conditioner's output promotes to f32 internally (f32 RoPE), so
    // its output dtype is not asserted.
    assert_eq!(
        out_fp32.dtype(),
        Dtype::Float32,
        "fp32 conditioner output must be Float32"
    );
    assert_eq!(
        out_bf16.shape(),
        &[1, 8, 8],
        "right-pad to min_sequence_length"
    );
    assert_eq!(out_fp32.shape(), &[1, 8, 8]);

    let r = rel_l2(&flat_f64(&out_bf16), &flat_f64(&out_fp32));
    println!("[encode_dtype] conditioner bf16-weight vs fp32-weight rel-L2 = {r:.3e}");
    assert!(
        r < 1.5e-1,
        "conditioner bf16 and fp32 must be the same computation (rel-L2 {r:.3e})"
    );

    // Guard 4 — the DISCRIMINATING guard for "fp32 variant uses fp32 WEIGHTS" (see the qwen3 tower
    // test). Re-run the bf16-WEIGHTS conditioner at fp32 compute (fp32 source + `Float32` dtype) so
    // the ONLY difference from `cond_fp32` is weight precision, and require the outputs to genuinely
    // diverge. If the fp32 path silently used bf16 weights, both would be one bit-identical fp32
    // computation (rel-L2 == 0) and this assertion goes RED.
    let out_bf16_weights = cond_bf16
        .forward(&source_fp32, &target, Dtype::Float32)
        .unwrap();
    let r_weight = rel_l2(&flat_f64(&out_fp32), &flat_f64(&out_bf16_weights));
    println!(
        "[encode_dtype] conditioner fp32-weight vs bf16-weight (both fp32 compute) rel-L2 = {r_weight:.3e}"
    );
    assert!(
        r_weight > 5e-4,
        "fp32 conditioner must genuinely use fp32 WEIGHTS: at equal (fp32) compute the fp32-weight \
         and bf16-weight outputs must differ by more than bf16 rounding (rel-L2 {r_weight:.3e})"
    );
}

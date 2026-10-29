use super::*;

#[test]
fn stg_control_selects_only_configured_self_attention_block() {
    // Mutation-sensitive for the SC-18759 block-28 row: changing either the
    // actual block index passed by a resident/streamed loop or the selected
    // index makes this fail. STG suppresses self attention only.
    let control = AvPerturbation::stg(&[28]);
    let before = control.attention_plan(27);
    let target = control.attention_plan(28);
    let after = control.attention_plan(29);

    assert_eq!(
        before,
        AvAttentionPlan {
            run_self: true,
            run_cross_modal: true,
        }
    );
    assert_eq!(
        target,
        AvAttentionPlan {
            run_self: false,
            run_cross_modal: true,
        }
    );
    assert_eq!(after, before);
}

#[test]
fn runtime_stg_override_selects_every_requested_block() {
    let control = AvPerturbation::stg_blocks(&[3, 7]).unwrap();
    assert!(!control.attention_plan(3).run_self);
    assert!(!control.attention_plan(7).run_self);
    assert!(control.attention_plan(28).run_self);
}

#[test]
fn modality_isolation_suppresses_only_cross_modal_attention() {
    // This must remain distinct from STG: self/text/FF still execute at
    // every block, while both A2V and V2A calls are skipped.
    let normal = AvPerturbation::NONE.attention_plan(28);
    let isolated = AvPerturbation::modality_isolated().attention_plan(28);
    assert_eq!(
        normal,
        AvAttentionPlan {
            run_self: true,
            run_cross_modal: true,
        }
    );
    assert_eq!(
        isolated,
        AvAttentionPlan {
            run_self: true,
            run_cross_modal: false,
        }
    );
    assert_ne!(normal, isolated);
}

#[test]
fn controlled_path_uses_actual_indices_for_resident_and_streamed_blocks() {
    // Structural mutation proof: the resident and streamed loops for both AV and video-only
    // execution must derive their per-block calls from the materialized/enumerated `index`,
    // not a fixed STG layer or a mode that silently becomes a whole-block skip.
    // The tests live in their own file, so `transformer.rs` and its production submodules
    // (sc-27342) hold only production uses.
    let source = [
        include_str!("../transformer.rs"),
        include_str!("attention.rs"),
        include_str!("av_block.rs"),
        include_str!("av_dit.rs"),
        include_str!("linear.rs"),
        include_str!("video_block.rs"),
        include_str!("video_dit.rs"),
    ]
    .concat();
    let production_index_uses = source.matches("perturbation.attention_plan(index)").count();
    assert_eq!(
        production_index_uses, 4,
        "resident and streamed AV/video-only loops must all apply the actual block index"
    );
    assert!(
        source.contains("if run_self {") && source.contains("if control.run_cross_modal {"),
        "self and cross-modal calls must be independently controlled inside AvBlock"
    );
    assert!(
        source.contains("let cross = attn2.forward") && source.contains("self.feed_forward"),
        "STG/modality controls must retain text cross-attention and feed-forward"
    );
}

/// sc-18789: the keyframe absolute-position marker lands on exactly the tokens the mask
/// marks — marked tokens shift by the embedding, unmarked tokens are bit-identical, and an
/// absent mask (or absent embedding) is an exact no-op.
#[test]
fn keyframes_embedding_applies_only_to_marked_tokens() {
    use mlx_rs::ops::indexing::IndexOp;
    struct CpuGuard(mlx_rs::Device);
    impl Drop for CpuGuard {
        fn drop(&mut self) {
            mlx_rs::Device::set_default(&self.0);
        }
    }
    let _cpu = CpuGuard(mlx_rs::Device::try_default().expect("default device"));
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 3, 2]);
    let emb = Array::from_slice(&[10.0f32, 20.0], &[1, 2]);
    let mask = Array::from_slice(&[0.0f32, 1.0, 0.0], &[1, 3, 1]);
    let out = apply_keyframes_embedding(&x, Some(&emb), Some(&mask)).unwrap();
    let got: Vec<f32> = (0..3)
        .flat_map(|t| (0..2).map(move |c| (t, c)))
        .map(|(t, c)| out.index((0, t, c)).item::<f32>())
        .collect();
    assert_eq!(got, vec![1.0, 2.0, 13.0, 24.0, 5.0, 6.0]);
    for out in [
        apply_keyframes_embedding(&x, Some(&emb), None).unwrap(),
        apply_keyframes_embedding(&x, None, Some(&mask)).unwrap(),
    ] {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(&out, &x).unwrap()).unwrap();
        assert_eq!(mlx_rs::ops::max(&d, None).unwrap().item::<f32>(), 0.0);
    }
}

#[test]
fn rms_norm_noweight_matches_fresh_ones_and_caches() {
    // sc-4052: the cached `(dim, dtype)` ones must give bit-identical results to a freshly built
    // ones, and a second call (cache hit) must match the first.
    use mlx_rs::ops::array_eq;
    use mlx_rs::random;
    use mlx_rs::transforms::eval;

    let x = random::normal::<f32>(&[2, 8], None, None, Some(&random::key(0).unwrap())).unwrap();
    let eps = 1e-6;
    let got = rms_norm_noweight(&x, eps).unwrap();
    let ones = Array::ones::<f32>(&[8]).unwrap();
    let want = fast_rms_norm(&x, &ones, eps).unwrap();
    eval([&got, &want]).unwrap();
    assert!(
        array_eq(&got, &want, None).unwrap().item::<bool>(),
        "cached ones == fresh ones"
    );
    let got2 = rms_norm_noweight(&x, eps).unwrap();
    eval([&got2]).unwrap();
    assert!(
        array_eq(&got, &got2, None).unwrap().item::<bool>(),
        "cache hit is identical"
    );
}

#[test]
fn rope_memo_content_path_caches_and_recomputes_on_change() {
    // sc-3995: the `None` (content) path recomputes the per-stage RoPE tables only when `positions`
    // changes. A constant positions (every denoise step) hits the cache; a new positions recomputes.
    let memo = RopeMemo::default();
    let calls = std::cell::Cell::new(0);
    let compute = |tag: f32| {
        // Distinct table content per "stage" so we can tell a hit from a recompute.
        (
            Array::from_slice(&[tag, tag + 1.0], &[2]),
            Array::from_slice(&[tag + 2.0, tag + 3.0], &[2]),
        )
    };

    let pos_a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
    let pos_a2 = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]); // same content
    let pos_b = Array::from_slice(&[9.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]); // different

    let (c0, _) = memo
        .get_or_compute(&pos_a, None, || {
            calls.set(calls.get() + 1);
            Ok(compute(10.0))
        })
        .unwrap();
    // Same positions content → cache hit, compute not re-run, identical table returned.
    let (c1, _) = memo
        .get_or_compute(&pos_a2, None, || {
            calls.set(calls.get() + 1);
            Ok(compute(99.0)) // would differ if (wrongly) recomputed
        })
        .unwrap();
    assert_eq!(calls.get(), 1, "same positions must hit the cache");
    assert_eq!(c0.as_slice::<f32>(), c1.as_slice::<f32>());

    // Different positions → recompute.
    let (c2, _) = memo
        .get_or_compute(&pos_b, None, || {
            calls.set(calls.get() + 1);
            Ok(compute(20.0))
        })
        .unwrap();
    assert_eq!(calls.get(), 2, "changed positions must recompute");
    assert_eq!(c2.as_slice::<f32>(), &[20.0, 21.0]);
}

#[test]
fn rope_memo_epoch_path_keys_on_epoch_not_positions() {
    // sc-7141: with `Some(epoch)` the memo keys on the epoch token, NOT the positions content. The
    // same epoch hits even if positions content differs (the caller guarantees epoch↔positions is
    // 1:1, so this can't happen in practice — it proves the key is the epoch); a new epoch recomputes.
    let memo = RopeMemo::default();
    let calls = std::cell::Cell::new(0);
    let compute = |tag: f32| {
        (
            Array::from_slice(&[tag, tag + 1.0], &[2]),
            Array::from_slice(&[tag + 2.0, tag + 3.0], &[2]),
        )
    };
    let pos_a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
    let pos_b = Array::from_slice(&[9.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]); // different content

    // Epoch 0 computes and caches.
    let (c0, _) = memo
        .get_or_compute(&pos_a, Some(0), || {
            calls.set(calls.get() + 1);
            Ok(compute(10.0))
        })
        .unwrap();
    // Same epoch → hit, even though positions content differs (proves the key is the epoch, O(1),
    // no positions copy). Returns the epoch-0 table.
    let (c1, _) = memo
        .get_or_compute(&pos_b, Some(0), || {
            calls.set(calls.get() + 1);
            Ok(compute(99.0))
        })
        .unwrap();
    assert_eq!(
        calls.get(),
        1,
        "same epoch must hit regardless of positions"
    );
    assert_eq!(c0.as_slice::<f32>(), c1.as_slice::<f32>());

    // New epoch (next stage) → recompute, even with the same positions as epoch 0.
    let (c2, _) = memo
        .get_or_compute(&pos_a, Some(1), || {
            calls.set(calls.get() + 1);
            Ok(compute(20.0))
        })
        .unwrap();
    assert_eq!(calls.get(), 2, "changed epoch must recompute");
    assert_eq!(c2.as_slice::<f32>(), &[20.0, 21.0]);

    // Switching from `Some` to `None` is a key-variant change → recompute (never a false hit).
    let (_c3, _) = memo
        .get_or_compute(&pos_a, None, || {
            calls.set(calls.get() + 1);
            Ok(compute(30.0))
        })
        .unwrap();
    assert_eq!(calls.get(), 3, "epoch→content key change must recompute");
}

#[test]
fn modulate_closed_form() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
    let scale = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[1, 1, 4]);
    let shift = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 4]);
    let got = modulate(&x, &scale, &shift).unwrap();
    assert_eq!(got.as_slice::<f32>(), &[2.0, 4.0, 2.0, 8.0]);
}

// sc-2963: the compiled glue helpers (`modulate`/`gated`/`gelu_ffn`) the AvDiT forward composes
// are bit-identical to eager (`max|Δ|=0`) at both LTX compute dtypes — f32 (the `quant_f32`
// quality target) and bf16 (the `quant_bf16` speed path). The block forward has no committed
// weights (the Q8 block lives in the 20 GB checkpoint), so this gates the helpers directly.
#[test]
fn compiled_glue_bit_identical_to_eager() {
    use mlx_rs::random;
    let rnd = |shape: &[i32], dt: Dtype| -> Array {
        let k = random::key(0).unwrap();
        random::normal::<f32>(shape, None, None, Some(&k))
            .unwrap()
            .as_dtype(dt)
            .unwrap()
    };
    let max_abs = |a: &Array, b: &Array| -> f32 {
        let d = mlx_rs::ops::abs(subtract(a, b).unwrap()).unwrap();
        mlx_rs::ops::max(&d, None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>()
    };
    for dt in [Dtype::Float32, Dtype::Bfloat16] {
        let (b, s, dim, ffn) = (1i32, 32i32, 64i32, 256i32);
        // modulate: x·(1+scale)+shift, scale/shift broadcast [B,1,dim].
        let x = rnd(&[b, s, dim], dt);
        let scale = rnd(&[b, 1, dim], dt);
        let shift = rnd(&[b, 1, dim], dt);
        crate::set_compile_glue(false);
        let me = modulate(&x, &scale, &shift).unwrap();
        crate::set_compile_glue(true);
        let mc = modulate(&x, &scale, &shift).unwrap();
        crate::set_compile_glue(false);
        assert_eq!(mc.dtype(), dt, "modulate dtype {dt:?}");
        assert_eq!(max_abs(&mc, &me), 0.0, "modulate {dt:?}");

        // gated: x + out·gate.
        let out = rnd(&[b, s, dim], dt);
        let gate = rnd(&[b, 1, dim], dt);
        crate::set_compile_glue(false);
        let ge = gated(&x, &out, &gate).unwrap();
        crate::set_compile_glue(true);
        let gc = gated(&x, &out, &gate).unwrap();
        crate::set_compile_glue(false);
        assert_eq!(max_abs(&gc, &ge), 0.0, "gated {dt:?}");

        // gelu_ffn: the tanh-GELU FFN activation (eager defers to core gelu_tanh).
        let h = rnd(&[b, s, ffn], dt);
        crate::set_compile_glue(false);
        let fe = gelu_ffn(&h).unwrap();
        crate::set_compile_glue(true);
        let fc = gelu_ffn(&h).unwrap();
        crate::set_compile_glue(false);
        assert_eq!(fc.dtype(), dt, "gelu_ffn dtype {dt:?}");
        // sc-12747: under MLX 0.32.0 the compiled tanh-GELU FFN rounds ~1 ULP-f32 differently
        // from eager (0-ULP on the prior 0.31.2 pin); bf16 stays bit-identical. f32 takes the
        // shared re-baselined tolerance; bf16 stays exact.
        let tol = if dt == Dtype::Float32 {
            mlx_gen::nn::COMPILED_GLUE_F32_ULP_TOL
        } else {
            0.0
        };
        let rel = mlx_gen::nn::max_rel_diff(&fc, &fe);
        assert!(
            rel <= tol,
            "gelu_ffn {dt:?}: rel|Δ|={rel:e} exceeds {tol:e}"
        );
    }
}

#[test]
fn ada_values_splits_rows() {
    let table = Array::from_slice(&(0..18).map(|v| v as f32).collect::<Vec<_>>(), &[9, 2]);
    let ts = Array::zeros::<f32>(&[1, 1, 18]).unwrap();
    let vals = ada_values(&table, &ts, 0, 3).unwrap();
    assert_eq!(vals.len(), 3);
    assert_eq!(vals[0].as_slice::<f32>(), &[0.0, 1.0]);
    assert_eq!(vals[2].as_slice::<f32>(), &[4.0, 5.0]);
}

#[test]
fn timestep_embedding_shape_and_pad() {
    // (N=2,) → (2, 256), cos-first (t=0 → cos=1, sin=0).
    let t = Array::from_slice(&[0.0f32, 1.0], &[2]);
    let emb = timestep_embedding(&t).unwrap();
    assert_eq!(emb.shape(), &[2, 256]);
    let row0 = emb.index_axis(0, 0).unwrap();
    let s = row0.as_slice::<f32>();
    assert!((s[0] - 1.0).abs() < 1e-6); // cos(0)
    assert!(s[128].abs() < 1e-6); // sin(0)
}

use mlx_rs::ops::{all_close, array_eq};

/// A bare dense Linear (no bias contribution) for the residual-math gates.
fn dense(w: Array) -> Linear {
    let out = w.shape()[0];
    let dt = w.dtype();
    Linear {
        kind: LinearKind::Dense {
            b: Some(Array::zeros::<f32>(&[out]).unwrap().as_dtype(dt).unwrap()),
            w,
        },
        lora: None,
    }
}

/// The sc-15265 contract this crate's `Linear::forward` implements, spelled out independently
/// of the implementation: the residual is scaled in its own (promoted) dtype and then **cast to
/// the host output's dtype before the add**, so the host's dtype is what comes out.
fn want_adapted(base: &Array, resid: &Array, s: f32) -> Array {
    let r = multiply(resid, scalar(s).as_dtype(resid.dtype()).unwrap()).unwrap();
    add(base, r.as_dtype(base.dtype()).unwrap()).unwrap()
}

/// Bit-pattern (not value) equality — `array_eq` would accept `-0.0 == +0.0`.
fn bit_exact(got: &Array, want: &Array) -> bool {
    if got.dtype() != want.dtype() || got.shape() != want.shape() {
        return false;
    }
    let bits = |a: &Array| match a.dtype() {
        Dtype::Bfloat16 | Dtype::Float16 => a.view::<u16>().unwrap(),
        Dtype::Float32 => a.view::<u32>().unwrap(),
        _ => a.clone(),
    };
    array_eq(bits(got), bits(want), false)
        .unwrap()
        .item::<bool>()
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_scale_zero_is_bit_exact_noop_and_nonzero_has_effect() {
    // base [out=2, in=3]; LoRA a=[in=3, rank=2], b=[rank=2, out=2] (residual form, alpha/rank
    // already folded into the per-pass scale by the loader — here it's the raw scale).
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let base = dense(w.clone()).forward(&x).unwrap();

    // scale 0 → bit-exact no-op. sc-15265: the residual is SKIPPED, not added-as-zero (that
    // stronger claim is what `disabled_ltx_adapters_residual_is_never_evaluated` gates; this
    // fixture is all-f32 and so cannot tell the two apart on its own).
    let mut lin0 = dense(w.clone());
    lin0.push_lora(a.clone(), b.clone(), vec![0.0]);
    assert!(bit_exact(&lin0.forward(&x).unwrap(), &base));

    // scale 0.5 → differs from base and equals the sc-15265 contract exactly:
    // `base + host_dtype(0.5 · (x·a)·b)`. NOT the pre-fix `add(&base, &resid)`, which promotes
    // whenever the factors are wider than the host (this all-f32 fixture makes the two
    // coincide — `dtype_narrowing_*` below is the leg that separates them).
    let mut lin1 = dense(w);
    lin1.push_lora(a.clone(), b.clone(), vec![0.5]);
    let got = lin1.forward(&x).unwrap();
    assert!(!array_eq(&got, &base, false).unwrap().item::<bool>());
    let resid = matmul(matmul(&x, &a).unwrap(), &b).unwrap();
    let want = want_adapted(&base, &resid, 0.5);
    assert_eq!(got.dtype(), base.dtype());
    assert!(all_close(&got, &want, 1e-6, 1e-6, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::lora_scale_zero_is_bit_exact_noop_and_nonzero_has_effect");
}

// ── sc-15265: LTX carries its own adapter stack, so the shared seam's two rules are
// DUPLICATED in `Linear::forward`. The pair below is the runnable gate for that copy — the
// only other test that touches a zero pass-scale on real factors is
// `tests/lora_real_weights.rs::…`, which is `#[ignore]`d behind a 20 GB checkpoint. Both use
// cheap synthetic fixtures and no weights.

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn dtype_narrowing_keeps_a_bf16_ltx_linear_bf16_under_f32_factors() {
    // Rule 2. A bf16 host with f32 LoRA factors: `add(bf16, f32)` promotes, so pre-sc-15265
    // this Linear — and every op downstream of it — silently became f32 the moment a LoRA was
    // installed. Delete the `.as_dtype(out.dtype())` in `Linear::forward` and the dtype
    // assertion below fails.
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    // f32 factors, left f32 on purpose — this is what the golden LoRA files ship.
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);

    let base = dense(w.clone()).forward(&x).unwrap();
    assert_eq!(base.dtype(), Dtype::Bfloat16, "host output is bf16");

    let mut lin = dense(w);
    lin.push_lora(a.clone(), b.clone(), vec![0.75]);
    let got = lin.forward(&x).unwrap();
    assert_eq!(
        got.dtype(),
        Dtype::Bfloat16,
        "installing an LTX adapter must not widen the host Linear's output dtype"
    );
    assert!(
        !bit_exact(&got, &base),
        "fixture check: the adapter must actually be live"
    );
    // The residual itself still promotes — only the hand-off to the host is narrowed.
    let resid = matmul(matmul(&x, &a).unwrap(), &b).unwrap();
    assert_eq!(resid.dtype(), Dtype::Float32);
    assert!(
        bit_exact(&got, &want_adapted(&base, &resid, 0.75)),
        "forward must be base + bf16(0.75·residual)"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::dtype_narrowing_keeps_a_bf16_ltx_linear_bf16_under_f32_factors");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn disabled_ltx_adapters_residual_is_never_evaluated() {
    // Rule 1. A pass whose scale is exactly 0 must be SKIPPED, not added as a zero — the
    // distinction the all-f32 fixtures above cannot see, because casting an exact-zero
    // residual is already bit-exact. An `INFINITY` factor makes the residual non-finite, so
    // `Inf · 0.0 = NaN`: staying bit-exact to the unadapted base proves the residual was never
    // formed. Delete the `if s == 0.0 { continue; }` in `Linear::forward` and this fails.
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let base = dense(w.clone()).forward(&x).unwrap();

    // LoRA with an Inf in `a`.
    let a_inf = Array::from_slice(&[f32::INFINITY; 6], &[3, 2]);
    let mut lin = dense(w.clone());
    lin.push_lora(a_inf, b, vec![0.0]);
    assert!(
        bit_exact(&lin.forward(&x).unwrap(), &base),
        "a zero-scale LTX LoRA pass must not evaluate its residual"
    );

    // LoKr with an Inf delta, and a per-pass schedule whose OFF pass must also skip.
    let delta_inf = Array::from_slice(&[f32::INFINITY; 6], &[2, 3]);
    let mut lin = dense(w);
    lin.push_lokr(delta_inf, vec![0.0, 1.0]);
    lin.set_lora_pass(0);
    assert!(
        bit_exact(&lin.forward(&x).unwrap(), &base),
        "an OFF pass in a per-pass schedule must not evaluate its residual either"
    );
    lin.set_lora_pass(1);
    assert!(
        !lin.forward(&x)
            .unwrap()
            .is_finite()
            .unwrap()
            .all(None)
            .unwrap()
            .item::<bool>(),
        "fixture check: the ON pass DOES evaluate the poisoned residual"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::disabled_ltx_adapters_residual_is_never_evaluated");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn empty_pass_scale_forward_errors_not_panics() {
    // A malformed adapter with an empty pass_scale must surface a clean error from forward,
    // not underflow `ps.len() - 1` (usize) into an OOB index and panic (sc-5259).
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let mut lin = dense(w);
    lin.push_lora(a, b, vec![]);
    assert!(lin.forward(&x).is_err());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::empty_pass_scale_forward_errors_not_panics");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_per_pass_selects_strength() {
    // pass_scale [0.0, 1.0]: pass 0 is a no-op, pass 1 applies the full residual (sc-2687).
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let base = dense(w.clone()).forward(&x).unwrap();

    let mut lin = dense(w);
    lin.push_lora(a, b, vec![0.0, 1.0]);
    lin.set_lora_pass(0);
    assert!(
        array_eq(lin.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>(),
        "pass 0 (strength 0) must be a no-op"
    );
    lin.set_lora_pass(1);
    assert!(
        !array_eq(lin.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>(),
        "pass 1 (strength 1) must change the output"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::lora_per_pass_selects_strength");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_pass_index_clamps_into_uniform_scale() {
    // A uniform (length-1) pass_scale is used regardless of the selected pass — the clamp keeps a
    // uniform adapter working even when the pipeline advances to pass 1.
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let mut lin = dense(w);
    lin.push_lora(a, b, vec![0.5]);
    lin.set_lora_pass(0);
    let p0 = lin.forward(&x).unwrap();
    lin.set_lora_pass(1);
    let p1 = lin.forward(&x).unwrap();
    assert!(array_eq(&p0, &p1, false).unwrap().item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::lora_pass_index_clamps_into_uniform_scale");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lokr_scale_zero_is_noop_nonzero_matches_delta_and_per_pass() {
    // sc-2393: a LoKr residual carries a precomputed `[out,in]` delta (alpha/rank baked in). The
    // forward is `out + pass_scale · x·ΔWᵀ` — scale 0 a bit-exact no-op, scale s = `base + s·x·ΔWᵀ`,
    // and the per-pass strength selects exactly like LoRA. `base [out=2,in=3]`, `ΔW [2,3]`.
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let delta = Array::from_slice(&[0.05f32, -0.1, 0.2, 0.15, -0.25, 0.3], &[2, 3]);
    let base = dense(w.clone()).forward(&x).unwrap();

    // scale 0 → bit-exact no-op (skipped outright, sc-15265).
    let mut lin0 = dense(w.clone());
    lin0.push_lokr(delta.clone(), vec![0.0]);
    assert!(bit_exact(&lin0.forward(&x).unwrap(), &base));

    // scale 0.5 → `base + host_dtype(0.5·x·ΔWᵀ)` (sc-15265), and differs from base.
    let mut lin = dense(w.clone());
    lin.push_lokr(delta.clone(), vec![0.5]);
    let got = lin.forward(&x).unwrap();
    assert!(!array_eq(&got, &base, false).unwrap().item::<bool>());
    let resid = matmul(&x, delta.t()).unwrap();
    let want = want_adapted(&base, &resid, 0.5);
    assert_eq!(got.dtype(), base.dtype());
    assert!(all_close(&got, &want, 1e-6, 1e-6, false)
        .unwrap()
        .item::<bool>());

    // Per-pass [0.0, 1.0]: pass 0 no-op, pass 1 applies.
    let mut linp = dense(w);
    linp.push_lokr(delta, vec![0.0, 1.0]);
    linp.set_lora_pass(0);
    assert!(array_eq(linp.forward(&x).unwrap(), &base, false)
        .unwrap()
        .item::<bool>());
    linp.set_lora_pass(1);
    assert!(!array_eq(linp.forward(&x).unwrap(), &base, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::lokr_scale_zero_is_noop_nonzero_matches_delta_and_per_pass");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn lora_and_lokr_stack_on_one_linear() {
    // Both adapter kinds sum onto a single base: `out + lora_resid + lokr_resid`.
    let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
    let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
    let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
    let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
    let delta = Array::from_slice(&[0.05f32, -0.1, 0.2, 0.15, -0.25, 0.3], &[2, 3]);
    let base = dense(w.clone()).forward(&x).unwrap();

    let mut lin = dense(w);
    lin.push_lora(a.clone(), b.clone(), vec![0.5]);
    lin.push_lokr(delta.clone(), vec![0.25]);
    let got = lin.forward(&x).unwrap();

    // sc-15265: each residual is narrowed to the running output's dtype before ITS add, so the
    // contract composes per-adapter (not "sum the residuals, then add once").
    let after_lora = want_adapted(&base, &matmul(matmul(&x, &a).unwrap(), &b).unwrap(), 0.5);
    let want = want_adapted(&after_lora, &matmul(&x, delta.t()).unwrap(), 0.25);
    assert_eq!(got.dtype(), base.dtype());
    assert!(all_close(&got, &want, 1e-6, 1e-6, false)
        .unwrap()
        .item::<bool>());
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED transformer::tests::lora_and_lokr_stack_on_one_linear"
    );
}

// --- sc-18758: ff_bias / audio_ff_bias / connector_ff_bias, keyframes_abs_pos_embedding ---------

/// A minimal 2-linear FeedForward-shaped weights map: `proj_in`/`proj_out`, each `[4,4]` weight +
/// `[4]` bias — `with_bias:false` also skips inserting the bias tensors, matching a real LTX-2.5
/// checkpoint (the tensor is genuinely absent, not merely unread).
fn ff_weights(with_bias: bool) -> Weights {
    let mut m = std::collections::HashMap::new();
    m.insert(
        "ff.proj_in.weight".to_string(),
        Array::from_slice(
            &[
                0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
            ],
            &[4, 4],
        ),
    );
    m.insert(
        "ff.proj_out.weight".to_string(),
        Array::from_slice(
            &[
                0.2f32, -0.1, 0.05, 0.15, -0.2, 0.3, 0.1, -0.05, 0.25, -0.15, 0.2, -0.1, 0.05, 0.1,
                -0.2, 0.15,
            ],
            &[4, 4],
        ),
    );
    if with_bias {
        m.insert(
            "ff.proj_in.bias".to_string(),
            Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]),
        );
        m.insert(
            "ff.proj_out.bias".to_string(),
            Array::from_slice(&[0.5f32, -0.5, 0.25, -0.25], &[4]),
        );
    }
    Weights::from_map(m)
}

/// Mutation test (sc-18758): flipping `ff_bias` genuinely changes the constructed parameter set —
/// not merely a defaulted-past flag. With a checkpoint that carries bias tensors, `bias:true` reads
/// and applies them (the forward differs from the bias-free result); `bias:false` never reads
/// `.bias` at all, so its `LinearKind` structurally omits the bias term (`Dense { b: None, .. }`)
/// even though the tensor was present in the source map — proving the flag, not the checkpoint
/// content, decides whether the bias parameter exists in the built module.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn ff_bias_flag_changes_constructed_params_and_forward() {
    let prec = Precision::dense_f32(4, 64);
    let w = ff_weights(true);
    let x = Array::from_slice(&[1.0f32, -1.0, 0.5, 2.0], &[1, 4]);

    let ff_biased = FeedForward::load(&w, "ff", prec, true).expect("bias:true loads");
    let ff_unbiased = FeedForward::load(&w, "ff", prec, false).expect("bias:false loads");

    // Structural: bias:false never installs a bias term, regardless of the source map.
    assert!(matches!(
        &ff_unbiased.proj_in.kind,
        LinearKind::Dense { b: None, .. }
    ));
    assert!(matches!(
        &ff_biased.proj_in.kind,
        LinearKind::Dense { b: Some(_), .. }
    ));

    // Forward: the two constructions produce numerically different output on the same input —
    // the flag is exercised, not defaulted past.
    let out_biased = ff_biased.forward(&x).unwrap();
    let out_unbiased = ff_unbiased.forward(&x).unwrap();
    mlx_rs::transforms::eval([&out_biased, &out_unbiased]).unwrap();
    assert!(
        !array_eq(&out_biased, &out_unbiased, None)
            .unwrap()
            .item::<bool>(),
        "ff_bias:true vs false must produce different output on a checkpoint with real bias values"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED transformer::tests::ff_bias_flag_changes_constructed_params_and_forward");
}

/// The other half of the mutation proof: `bias:false` on a checkpoint that genuinely carries no
/// `.bias` tensor (the real LTX-2.5 shape) must NOT `require` it — and `bias:true` on that same
/// checkpoint must error, proving the strict-loader contract (never silently zero-fill a bias that
/// was never shipped, never silently skip one the config says must be there).
#[test]
fn ff_bias_false_never_requires_an_absent_bias_tensor() {
    let prec = Precision::dense_f32(4, 64);
    let w = ff_weights(false);

    FeedForward::load(&w, "ff", prec, false).expect("bias:false must not require the tensor");
    let err = FeedForward::load(&w, "ff", prec, true);
    assert!(
        err.is_err(),
        "bias:true against a checkpoint with no `.bias` tensor must error, not silently proceed"
    );
}

#[test]
fn apply_keyframes_embedding_is_a_no_op_without_embedding_or_mask() {
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let embedding = Array::from_slice(&[10.0f32, 20.0], &[1, 2]);
    let mask = Array::from_slice(&[1.0f32, 0.0], &[1, 2, 1]);

    // No embedding configured (a pre-sc-18758 / 2.3 model) → exact passthrough.
    let got = apply_keyframes_embedding(&x, None, Some(&mask)).unwrap();
    mlx_rs::transforms::eval([&got, &x]).unwrap();
    assert!(array_eq(&got, &x, None).unwrap().item::<bool>());

    // Embedding configured but no mask supplied (a grid path with no generated slots) → exact
    // passthrough.
    let got2 = apply_keyframes_embedding(&x, Some(&embedding), None).unwrap();
    mlx_rs::transforms::eval([&got2]).unwrap();
    assert!(array_eq(&got2, &x, None).unwrap().item::<bool>());
}

#[test]
fn apply_keyframes_embedding_marks_only_gated_tokens() {
    // (1, 2, 2): token 0 marked (mask>0), token 1 not.
    let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let embedding = Array::from_slice(&[10.0f32, 20.0], &[1, 2]);
    let mask = Array::from_slice(&[1.0f32, 0.0], &[1, 2, 1]);

    let got = apply_keyframes_embedding(&x, Some(&embedding), Some(&mask)).unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    let out = got.as_slice::<f32>();
    // Token 0 gets `x + embedding`.
    assert!((out[0] - 11.0).abs() < 1e-6);
    assert!((out[1] - 22.0).abs() < 1e-6);
    // Token 1 (mask == 0) is untouched.
    assert!((out[2] - 3.0).abs() < 1e-6);
    assert!((out[3] - 4.0).abs() < 1e-6);
}

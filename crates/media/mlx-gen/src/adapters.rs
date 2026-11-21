//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base, re-exported from [`mlx_layers::adapters`] with [`loader`] hosted in `mlx-gen` (`SW-6955`).

pub mod facts;
pub mod loader;

pub use mlx_layers::adapters::{
    build_lokr_factors, conv_lora_delta, install_adapter, lokr_structured_layout, prefixed_paths,
    quantized_matmul_with_bias, read_scalar_alpha, reconstruct_loha_delta, reconstruct_lokr_delta,
    reconstruct_lokr_delta_scaled, validate_lokr_metadata_scale, AdaptableConv2d, AdaptableHost,
    AdaptableLinear, Adapter, AdapterStack, BflTarget, DiffPatchPart, Linear, LinearBase,
    LinearFacts, LokrFactors, LoraRowSlice, ScaleContext, DEFAULT_LYCORIS_DTYPE,
};

#[cfg(test)]
use crate::array::scalar;
#[cfg(test)]
use mlx_rs::{
    ops::{add, addmm, matmul, multiply},
    Array, Dtype,
};

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};

    #[test]
    fn read_scalar_alpha_errors_on_empty_or_non_finite_and_reads_bf16() {
        // sc-11129: a size-0 `.alpha` from a malformed/truncated adapter file must produce
        // a TYPED error, not the `as_slice()[0]` panic that would abort the worker. (safetensors
        // cannot serialize an empty array, so exercise the reader on the in-memory empty tensor —
        // the exact value a truncated file would deserialize to.)
        let empty = Array::from_slice::<f32>(&[], &[0]);
        let err = read_scalar_alpha(&empty, "k.alpha").expect_err("an empty .alpha must error");
        assert!(
            matches!(&err, crate::Error::Msg(m) if m.contains("k.alpha") && m.contains("empty")),
            "expected a typed Msg error, got {err:?}"
        );
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = read_scalar_alpha(&Array::from_slice(&[bad], &[1]), "k.alpha")
                .expect_err("a non-finite .alpha must error");
            assert!(
                matches!(&err, crate::Error::Msg(m) if m.contains("not finite")),
                "{bad}: {err:?}"
            );
        }
        // Happy path: a well-formed bf16 scalar still reads its value.
        let good = Array::from_slice(&[8.0f32], &[1])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        assert_eq!(read_scalar_alpha(&good, "k.alpha").unwrap(), 8.0);
    }

    /// sc-27320: `quantize_if_aligned` packs a dense base only when its in-features divide the
    /// group, reports whether it did, and leaves an already-packed base alone; `is_quant_aligned`
    /// agrees with it. (Load/quantize only — no forward.)
    #[test]
    fn quantize_if_aligned_packs_only_dense_aligned_bases() {
        let mut aligned = AdaptableLinear::dense(Array::zeros::<f32>(&[4, 64]).unwrap(), None);
        assert!(aligned.is_quant_aligned(None));
        assert!(aligned.quantize_if_aligned(8, None).unwrap());
        assert!(aligned.is_quantized());
        assert_eq!(aligned.base_shape(), vec![4, 64]);
        assert!(
            aligned.is_quant_aligned(None),
            "a packed base keeps its logical in-features"
        );
        assert!(
            !aligned.quantize_if_aligned(8, None).unwrap(),
            "an already-packed base is left untouched"
        );

        let mut misaligned = AdaptableLinear::dense(Array::zeros::<f32>(&[4, 48]).unwrap(), None);
        assert!(!misaligned.is_quant_aligned(None));
        assert!(!misaligned.quantize_if_aligned(8, None).unwrap());
        assert!(!misaligned.is_quantized());
        assert!(misaligned.is_quant_aligned(Some(16)));
    }

    #[derive(Clone, Copy, Debug)]
    enum ResidualPayloadKind {
        Lora,
        MaterializedLokr,
        StructuredLokr,
    }

    fn write_lazy_adapter_payload(path: &std::path::Path, value: f32) {
        let mut header = br#"{"a":{"dtype":"F32","shape":[8,2],"data_offsets":[0,64]},"b":{"dtype":"F32","shape":[2,8],"data_offsets":[64,128]},"delta":{"dtype":"F32","shape":[8,8],"data_offsets":[128,384]},"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[384,400]},"w2":{"dtype":"F32","shape":[4,4],"data_offsets":[400,464]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = Vec::with_capacity(8 + header.len() + 464);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for _ in 0..116 {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn adapter_from_lazy_payload(path: &std::path::Path, kind: ResidualPayloadKind) -> Adapter {
        let weights = crate::weights::Weights::from_file(path).unwrap();
        match kind {
            ResidualPayloadKind::Lora => Adapter::Lora {
                a: weights.require("a").unwrap().clone(),
                b: weights.require("b").unwrap().clone(),
                scale: 1.0,
            },
            ResidualPayloadKind::MaterializedLokr => Adapter::Lokr {
                delta: weights.require("delta").unwrap().clone(),
                scale: 1.0,
            },
            ResidualPayloadKind::StructuredLokr => Adapter::LokrStructured {
                factors: LokrFactors {
                    w1: weights.require("w1").unwrap().clone(),
                    w2: weights.require("w2").unwrap().clone(),
                    a: 2,
                    b: 4,
                    c: 2,
                    d: 4,
                    scale: 1.0,
                },
            },
        }
    }

    #[test]
    fn adapter_payload_materialization_stays_inside_pin_for_every_residual_form() {
        for kind in [
            ResidualPayloadKind::Lora,
            ResidualPayloadKind::MaterializedLokr,
            ResidualPayloadKind::StructuredLokr,
        ] {
            let dir = tempfile::tempdir().unwrap();
            let source = dir.path().join("adapter.safetensors");
            let replacement = dir.path().join("replacement.safetensors");
            write_lazy_adapter_payload(&source, 0.25);
            write_lazy_adapter_payload(&replacement, -0.75);
            let pin = crate::PinnedWeightsFile::pin(&source).unwrap();

            let result: crate::Result<()> = pin.read_unchanged(|path| {
                let adapter = adapter_from_lazy_payload(path, kind);
                std::fs::rename(&replacement, &source).unwrap();
                adapter.materialize()
            });
            let error = result.expect_err("A to B replacement must invalidate the adapter pin");
            match error {
                crate::Error::Unsupported(reason)
                    if reason.starts_with("artifact seal mismatch after load: ") => {}
                crate::Error::Unsupported(reason) => {
                    panic!("{kind:?}: unexpected artifact-seal reason: {reason}")
                }
                other => {
                    panic!("{kind:?}: expected a typed artifact-seal rejection, got: {other:?}")
                }
            }
        }
    }

    fn lokr_2x2() -> Array {
        reconstruct_lokr_delta(
            8.0,
            4.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap()
    }

    #[test]
    fn lokr_delta_stored_bf16() {
        assert_eq!(lokr_2x2().dtype(), Dtype::Bfloat16);
    }

    // ── sc-15265: adapter installation must not change the host Linear's output dtype ──────────
    //
    // A bf16 base with f32 LoRA/LoKr factors used to promote the WHOLE forward to f32, so
    // "install an adapter and set its scale to 0" was NOT the same as "don't install it"
    // (measured drift ≈1.9e-4 dense / 2.9e-4 Q4 e2e), and a *live* adapter silently widened
    // every downstream activation. The helpers below build a 64-wide (group-size-aligned, so the
    // same base can be quantized) bf16 host and f32 rank-4 factors — the exact shape of the bug.

    /// Deterministic `[rows, cols]` f32 array from a cheap analytic pattern.
    fn synth(rows: i32, cols: i32, seed: f32) -> Array {
        let n = (rows * cols) as usize;
        let v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.017 + seed).sin() * 0.05)
            .collect();
        Array::from_slice(&v, &[rows, cols])
    }

    /// A bf16 host Linear (`[64, 64]`, no bias), bf16 activations `[4, 64]`, and f32 rank-4
    /// LoRA factors `a: [64, 4]`, `b: [4, 64]` — the promoting combination.
    fn bf16_host() -> (AdaptableLinear, Array, Array, Array) {
        let w = synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap();
        let x = synth(4, 64, 1.1).as_dtype(Dtype::Bfloat16).unwrap();
        (
            AdaptableLinear::dense(w, None),
            x,
            synth(64, 4, 2.7),
            synth(4, 64, 3.9),
        )
    }

    /// `true` iff the two arrays are bit-identical in dtype, shape, and every element's **raw bit
    /// pattern**.
    ///
    /// This deliberately does NOT use `array_eq`, which is *value* equality and therefore accepts
    /// `-0.0 == +0.0` (and, for that matter, rejects `NaN == NaN`). The contract these tests pin is
    /// byte identity — "installed at scale 0" produces the same bytes as "not installed" — and the
    /// sign bit is exactly where a value comparison would let a real regression through: a LoKr
    /// residual is built as `multiply(&factor2, scalar(0.0))`, which yields **signed** zeros, so
    /// `base(-0.0) + (+0.0) = +0.0` flips a sign bit that `array_eq` cannot see. Reinterpret through
    /// `view` (an unsigned integer of the same width) and compare those instead.
    fn bit_exact(got: &Array, want: &Array) -> bool {
        if got.dtype() != want.dtype() || got.shape() != want.shape() {
            return false;
        }
        let bits = |a: &Array| match a.dtype() {
            Dtype::Bfloat16 | Dtype::Float16 => a.view::<u16>().unwrap(),
            Dtype::Float32 => a.view::<u32>().unwrap(),
            Dtype::Float64 => a.view::<u64>().unwrap(),
            // Integer/bool dtypes have no redundant encodings — value equality IS bit equality.
            _ => a.clone(),
        };
        array_eq(bits(got), bits(want), false)
            .unwrap()
            .item::<bool>()
    }

    #[test]
    fn bit_exact_rejects_a_signed_zero_flip_that_value_equality_accepts() {
        // Pins the helper above (MINOR 5): the gates below are only as strong as this comparison.
        let neg = Array::from_slice(&[-0.0f32, 1.0], &[2]);
        let pos = Array::from_slice(&[0.0f32, 1.0], &[2]);
        assert!(
            array_eq(&neg, &pos, false).unwrap().item::<bool>(),
            "value equality cannot see the sign bit — which is why bit_exact must not use it"
        );
        assert!(!bit_exact(&neg, &pos), "bit_exact must see the sign bit");
        assert!(bit_exact(&neg, &neg.clone()));
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn scale_zero_lora_is_bit_exact_noop_on_a_bf16_dense_base() {
        let (mut lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        assert_eq!(base.dtype(), Dtype::Bfloat16, "host output is bf16");

        lin.push(Adapter::Lora {
            a: a.clone(),
            b: b.clone(),
            scale: 0.0,
        });
        let zero = lin.forward(&x).unwrap();
        assert_eq!(
            zero.dtype(),
            Dtype::Bfloat16,
            "installing an adapter must not widen the host's output dtype"
        );
        assert!(
            bit_exact(&zero, &base),
            "a scale-0 adapter must be bit-exact to no adapter at all"
        );

        // MUTATION: the same adapter at a live scale must actually change the output — otherwise
        // the bit-exactness above would be a false green from a dead residual.
        let mut live =
            AdaptableLinear::dense(synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap(), None);
        live.push(Adapter::Lora { a, b, scale: 1.0 });
        assert!(
            !bit_exact(&live.forward(&x).unwrap(), &base),
            "a live adapter must change the forward (residual is not identically zero)"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::scale_zero_lora_is_bit_exact_noop_on_a_bf16_dense_base");
    }

    /// A `[rows, cols]` f32 array whose every entry is `+INFINITY`.
    fn inf(rows: i32, cols: i32) -> Array {
        Array::from_slice(&vec![f32::INFINITY; (rows * cols) as usize], &[rows, cols])
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn a_disabled_adapters_residual_is_never_evaluated() {
        // sc-15265, MAJOR-1 gate. This is the ONLY test that discriminates the `is_disabled()`
        // short-circuit in `apply_adapters` from the narrowing cast beside it. Every other
        // scale-0 gate here (and the krea e2e tier sweep) stays green with the short-circuit
        // deleted, because `0 · finite = 0` and casting an exact-zero residual to the host dtype is
        // *already* bit-exact — the cast alone carries them.
        //
        // The discriminator is a residual that is NOT zero when you compute it: a factor of
        // `+INFINITY` makes the low-rank product non-finite, and `Inf · 0.0 = NaN`. So a scale-0
        // adapter whose forward is still bit-exact to the unadapted base proves the residual was
        // never formed at all — no arithmetic identity can rescue a NaN. Delete the
        // `if adapter.is_disabled() { continue; }` in `apply_adapters` and every assertion below
        // fails with a NaN output.
        //
        // This is not a synthetic-only worry: it is exactly the failure mode a third-party
        // checkpoint carrying an `Inf` weight would hit at the moment a user turns the adapter OFF.
        let (lin, x, _, b) = bf16_host();
        let base = lin.forward(&x).unwrap();

        // (1) LoRA — an `Inf` in `a`.
        let poison_lora = Adapter::Lora {
            a: inf(64, 4),
            b: b.clone(),
            scale: 0.0,
        };
        // The fixture must actually be a trap, or this whole test is vacuous.
        let raw = poison_lora.residual(&x).unwrap();
        assert!(
            !raw.is_finite().unwrap().all(None).unwrap().item::<bool>(),
            "fixture check: evaluating this residual must produce a non-finite value"
        );
        let mut poisoned = lin.clone();
        poisoned.push(poison_lora);
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled LoRA's residual must never be evaluated (short-circuit deleted?)"
        );

        // (2) Materialized LoKr — an `Inf` in the delta.
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::Lokr {
            delta: inf(64, 64).as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.0,
        });
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled LoKr's residual must never be evaluated"
        );

        // (3) Structured LoKr — an `Inf` in `w2`. This arm is why `LokrFactors` retains its
        // pre-bake `scale` (sc-15265): the scale is folded into `w2` at build time, so `Inf · 0.0`
        // bakes a NaN straight into the factor. Without a recoverable disabled flag there is no
        // way to skip it, and "turn the adapter off" would output NaN.
        let factors = build_lokr_factors(
            0.0,
            &[64, 64],
            Some(&synth(8, 8, 6.1)),
            None,
            None,
            Some(&inf(8, 8)),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("8×8 ⊗ 8×8 factors a 64×64 base");
        assert!(
            !factors
                .w2
                .is_finite()
                .unwrap()
                .all(None)
                .unwrap()
                .item::<bool>(),
            "fixture check: Inf · 0.0 must bake a NaN into w2"
        );
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::LokrStructured { factors });
        assert!(
            bit_exact(&poisoned.forward(&x).unwrap(), &base),
            "a disabled structured LoKr's residual must never be evaluated"
        );

        // And `forward_upcast` runs the same accumulation, so it inherits the same guarantee.
        let upcast_base = lin.forward_upcast(&x).unwrap();
        let mut poisoned = lin.clone();
        poisoned.push(Adapter::Lora {
            a: inf(64, 4),
            b,
            scale: 0.0,
        });
        assert!(
            bit_exact(&poisoned.forward_upcast(&x).unwrap(), &upcast_base),
            "forward_upcast must short-circuit a disabled adapter too"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::a_disabled_adapters_residual_is_never_evaluated");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn scale_zero_lora_is_bit_exact_noop_on_a_packed_base() {
        for bits in [4, 8] {
            let (mut lin, x, a, b) = bf16_host();
            lin.quantize(bits, None).unwrap();
            let base = lin.forward(&x).unwrap();
            assert_eq!(base.dtype(), Dtype::Bfloat16, "q{bits} host output is bf16");

            lin.push(Adapter::Lora { a, b, scale: 0.0 });
            let zero = lin.forward(&x).unwrap();
            assert_eq!(
                zero.dtype(),
                Dtype::Bfloat16,
                "q{bits}: installing an adapter must not widen the host's output dtype"
            );
            assert!(
                bit_exact(&zero, &base),
                "q{bits}: a scale-0 adapter must be bit-exact to no adapter at all"
            );
        }
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::scale_zero_lora_is_bit_exact_noop_on_a_packed_base");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn scale_zero_lokr_variants_are_bit_exact_noops_on_a_bf16_base() {
        // The two LoKr arms, both of which are pure regression pins rather than reproductions of
        // the sc-15265 promotion:
        //   * materialized LoKr casts its delta to the ACTIVATION dtype in `Adapter::residual`, so
        //     its residual was already bf16 on a bf16 host — this arm never promoted;
        //   * structured LoKr builds its factors at an explicit `out_dtype` (f32 here), but
        //     `LokrFactors::residual` likewise casts them to the activation dtype, so this arm
        //     never promoted either. Building it f32 over a bf16 host does NOT reproduce the bug —
        //     the arm is exercised only to pin that it *stays* a bit-exact no-op at `scale = 0`.
        // (The arm that actually promoted is LoRA, whose factors are used at their file dtype; see
        // `live_adapter_keeps_the_host_output_dtype_and_matches_a_cast_residual`.)
        let (lin, x, _, _) = bf16_host();
        let base = lin.forward(&x).unwrap();

        let mut mat = lin.clone();
        mat.push(Adapter::Lokr {
            delta: synth(64, 64, 5.5).as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.0,
        });
        assert!(
            bit_exact(&mat.forward(&x).unwrap(), &base),
            "a scale-0 materialized LoKr must be bit-exact to no adapter"
        );

        let factors = build_lokr_factors(
            0.0,
            &[64, 64],
            Some(&synth(8, 8, 6.1)),
            None,
            None,
            Some(&synth(8, 8, 7.3)),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("8×8 ⊗ 8×8 factors a 64×64 base");
        let mut structured = lin.clone();
        structured.push(Adapter::LokrStructured { factors });
        let got = structured.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "a structured LoKr must not widen the host's output dtype"
        );
        assert!(
            bit_exact(&got, &base),
            "a scale-0 structured LoKr must be bit-exact to no adapter"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::scale_zero_lokr_variants_are_bit_exact_noops_on_a_bf16_base");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn live_adapter_keeps_the_host_output_dtype_and_matches_a_cast_residual() {
        // Half 2 (sc-15265): the widening is NOT confined to scale 0. A live f32-factor adapter
        // over a bf16 host must land its residual in the host dtype — `bf16(base) + bf16(residual)`
        // — rather than promoting the whole downstream chain to f32.
        let (mut lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        let adapter = Adapter::Lora { a, b, scale: 0.75 };
        let residual = adapter.residual(&x).unwrap();
        assert_eq!(
            residual.dtype(),
            Dtype::Float32,
            "f32 factors still promote the RESIDUAL itself (Adapter::residual is unchanged)"
        );

        lin.push(adapter);
        let got = lin.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "a live adapter must not widen the host's output dtype"
        );
        let want = add(&base, residual.as_dtype(Dtype::Bfloat16).unwrap()).unwrap();
        assert!(
            bit_exact(&got, &want),
            "the forward must be base + host-dtype(residual)"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::live_adapter_keeps_the_host_output_dtype_and_matches_a_cast_residual");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn a_training_stack_keeps_its_pre_sc15265_numerics() {
        // sc-15265 rule 0. The trainers ship f32 master factors over bf16 block streams
        // (`install_training_lora` passes `dtype = None`); narrowing that residual would round the
        // adapter's whole contribution away while it is still small, so a training stack opts out
        // of BOTH rules. This pins that: `set_training_adapters` must reproduce the pre-fix
        // `add(base, residual)` exactly — promotion and all — while the same adapters installed the
        // inference way must not.
        let (lin, x, a, b) = bf16_host();
        let base = lin.forward(&x).unwrap();
        let adapter = Adapter::Lora {
            a: a.clone(),
            b: b.clone(),
            scale: 1.0,
        };
        let residual = adapter.residual(&x).unwrap();
        assert_eq!(residual.dtype(), Dtype::Float32);

        let mut train = lin.clone();
        train.set_training_adapters(vec![adapter]);
        let got = train.forward(&x).unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Float32,
            "a training stack must keep the promoted (pre-fix) accumulation dtype"
        );
        assert!(
            bit_exact(&got, &add(&base, &residual).unwrap()),
            "a training stack must be bit-identical to the pre-sc-15265 add"
        );

        // Rule 1 is opted out of too, so a training install is untouched even at scale 0 — an f32
        // zero residual still promotes, exactly as it did before. No partial exemption.
        let mut train0 = lin.clone();
        train0.set_training_adapters(vec![Adapter::Lora { a, b, scale: 0.0 }]);
        assert_eq!(
            train0.forward(&x).unwrap().dtype(),
            Dtype::Float32,
            "the training exemption covers the scale-0 skip as well"
        );

        // And `set_adapters` (the inference / clear path) must NOT inherit the flag.
        let mut back = train;
        back.set_adapters(vec![Adapter::Lora {
            a: synth(64, 4, 2.7),
            b: synth(4, 64, 3.9),
            scale: 1.0,
        }]);
        assert_eq!(
            back.forward(&x).unwrap().dtype(),
            Dtype::Bfloat16,
            "set_adapters must clear the training flag, not leave it latched"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::a_training_stack_keeps_its_pre_sc15265_numerics");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn forward_upcast_also_keeps_its_own_output_dtype() {
        // `forward_upcast` widens the BASE to the activation dtype on purpose; the adapter must
        // still not widen it any further than that.
        let w = synth(64, 64, 0.3).as_dtype(Dtype::Bfloat16).unwrap();
        let x = synth(4, 64, 1.1).as_dtype(Dtype::Bfloat16).unwrap();
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward_upcast(&x).unwrap();
        assert_eq!(base.dtype(), Dtype::Bfloat16);
        lin.push(Adapter::Lora {
            a: synth(64, 4, 2.7),
            b: synth(4, 64, 3.9),
            scale: 0.0,
        });
        let zero = lin.forward_upcast(&x).unwrap();
        assert!(
            bit_exact(&zero, &base),
            "forward_upcast: a scale-0 adapter must be a bit-exact no-op too"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::forward_upcast_also_keeps_its_own_output_dtype");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn forward_upcast_uses_activation_dtype_without_mutating_storage() {
        let weight = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let lin = AdaptableLinear::dense(weight, None);
        let x = Array::from_slice(&[0.25f32, 0.5], &[1, 2]);

        let got = lin.forward_upcast(&x).unwrap();
        let want = Array::from_slice(&[1.25f32, 2.75], &[1, 2]);

        assert_eq!(got.dtype(), Dtype::Float32);
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
        let LinearBase::Dense(base) = &lin.base else {
            panic!("expected dense base");
        };
        assert_eq!(base.weight.value.dtype(), Dtype::Bfloat16);
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::forward_upcast_uses_activation_dtype_without_mutating_storage");
    }

    #[test]
    fn reconstruct_lokr_delta_rejects_zero_rank() {
        // F-141 (sc-11129): a `rank = 0` metadata (alpha then defaults to rank) makes the scale
        // `0/0 = NaN`, which would bake into the reconstructed delta and NaN-poison every render. The
        // shared seam must reject it with a typed error so every consumer (SDXL/Kolors/InstantID/Wan)
        // inherits the guard, retiring the fixes-don't-travel class.
        let err = reconstruct_lokr_delta(
            0.0,
            0.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            Dtype::Bfloat16,
        )
        .expect_err("rank 0 must be rejected, not produce a NaN delta");
        assert!(
            matches!(err, crate::Error::Msg(_)),
            "expected a typed Msg error"
        );
        // Regression: a well-formed alpha/rank still reconstructs (the happy path is unaffected).
        assert!(lokr_2x2().sum(None).unwrap().item::<f32>().is_finite());
    }

    #[test]
    fn build_lokr_factors_rejects_non_finite_scale() {
        // F-141 (sc-11129): the packed/deferred path receives the pre-derived `(alpha/rank)·strength`
        // scale, so a `rank = 0` yields a NaN scale that would bake into the structured `w2`. The seam
        // must reject a non-finite scale rather than install a NaN factor. (`LokrFactors` is not Debug,
        // so match rather than `expect_err`.)
        let result = build_lokr_factors(
            f32::NAN,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
            None,
            Dtype::Bfloat16,
        );
        assert!(
            matches!(result, Err(crate::Error::Msg(_))),
            "a non-finite scale must be rejected with a typed Msg error"
        );
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn scale_zero_lokr_is_bit_exact_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        lin.push(Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.0,
        });
        let out = lin.forward(&x).unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
        crate::model_inference_device::record_model_inference_device();
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::scale_zero_lokr_is_bit_exact_noop"
        );
    }

    #[test]
    fn lokr_residual_runs_in_activation_dtype() {
        // sc-2718: the f32 bug-workaround is gone (NAX 16-bit dense GEMM fixed at the toolchain
        // level, sc-2772). A LoKr residual now runs in the ACTIVATION dtype — bf16 on the bf16 path
        // — mirroring the fork's `scale · matmul(x, delta.astype(x.dtype).T)`. So a bf16-input LoKr
        // residual must (a) return bf16 and (b) match the f32 reference within bf16 rounding — NOT
        // diverge (which is what the old buggy bf16 GEMM produced and the f32 detour avoided).
        let delta = lokr_2x2(); // bf16
        let x32 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2]);
        let lokr = Adapter::Lokr {
            delta: delta.clone(),
            scale: 0.5,
        };

        let got = lokr
            .residual(&x32.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "bf16-input LoKr residual runs in the activation dtype"
        );

        let want = multiply(
            matmul(&x32, delta.as_dtype(Dtype::Float32).unwrap().t()).unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got, &want, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoKr residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn lora_residual_is_fork_faithful_no_forced_dtype() {
        // sc-2718: LoRA factors keep their loaded dtype and the result is NOT cast back, replicating
        // the fork's `scale · matmul(matmul(x, lora_A), lora_B)` byte-for-byte.
        let a32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1 - 0.4).collect::<Vec<_>>(),
            &[2, 4],
        );
        let b32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
            &[4, 2],
        );
        let x_bf16 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // f32 factors (the goldens' dtype): a bf16 `x` promotes the residual to f32 — and it is
        // byte-exact to the fork's `scale · (x·A)·B` (no forced dtype, no cast-back).
        let lora_f32 = Adapter::Lora {
            a: a32.clone(),
            b: b32.clone(),
            scale: 0.5,
        };
        let got_f32 = lora_f32.residual(&x_bf16).unwrap();
        assert_eq!(
            got_f32.dtype(),
            Dtype::Float32,
            "f32 factors promote the residual to f32 (fork-faithful, not forced)"
        );
        let want_f32 = multiply(
            matmul(matmul(&x_bf16, &a32).unwrap(), &b32).unwrap(),
            scalar(0.5),
        )
        .unwrap();
        assert!(
            array_eq(&got_f32, &want_f32, false).unwrap().item::<bool>(),
            "LoRA residual must be byte-exact to the fork's scale·(x·A)·B"
        );

        // bf16 factors: the residual runs bf16 — the `[seq,r]·[r,out]` (K=rank=4≤512, M=seq=3≥2)
        // shape the NAX build mis-ran before sc-2772 — and matches the f32 reference within bf16
        // rounding (NOT garbage), proving the GEMM bug is gone so the f32 detour is unneeded.
        let lora_bf16 = Adapter::Lora {
            a: a32.as_dtype(Dtype::Bfloat16).unwrap(),
            b: b32.as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.5,
        };
        let got_bf16 = lora_bf16.residual(&x_bf16).unwrap();
        assert_eq!(
            got_bf16.dtype(),
            Dtype::Bfloat16,
            "bf16 factors keep the residual in the activation dtype"
        );
        let want_bf16 = multiply(
            matmul(
                matmul(x_bf16.as_dtype(Dtype::Float32).unwrap(), &a32).unwrap(),
                &b32,
            )
            .unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got_bf16, &want_bf16, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 LoRA residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn biased_dense_forward_is_fused_addmm() {
        // sc-2779: the biased dense base must be a FUSED `addmm(bias, x, Wᵀ)`, not `matmul`+`add`.
        // In bf16 the two differ (double-rounding), so feed bf16 activations and assert the forward
        // is bit-exact to `addmm` and bit-distinct from `matmul`+`add` — i.e. the fusion is real.
        let n = 4 * 64;
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..n)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let lin = AdaptableLinear::dense(w.clone(), Some(bias.clone()));
        let got = lin.forward(&x).unwrap();

        let want_addmm = addmm(&bias, &x, w.t(), 1.0, 1.0).unwrap();
        assert!(
            array_eq(&got, &want_addmm, false).unwrap().item::<bool>(),
            "biased dense forward must be bit-exact to addmm(bias, x, Wᵀ)"
        );

        // And it must NOT be the double-rounding matmul+add (which is what the bug looked like).
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            !array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "bf16 addmm should differ from matmul+add (double-rounding) — fusion not applied?"
        );
        crate::model_inference_device::record_model_inference_device();
        println!(
            "\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::biased_dense_forward_is_fused_addmm"
        );
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn biased_dense_forward_f32_acts_match_matmul_add_bit_exact() {
        // sc-2779 golden-safety invariant: with f32 activations (the current Z-Image/Qwen/FLUX path,
        // even over bf16 weights), `addmm == matmul+add` bit-for-bit — nothing rounds to bf16
        // mid-op. This is why lifting the core to addmm cannot regress any current f32-act golden.
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap(); // bf16 weights
        let bias = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..4 * 64)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        ); // f32 activations

        let got = AdaptableLinear::dense(w.clone(), Some(bias.clone()))
            .forward(&x)
            .unwrap();
        let matmul_add = add(matmul(&x, w.t()).unwrap(), &bias).unwrap();
        assert!(
            array_eq(&got, &matmul_add, false).unwrap().item::<bool>(),
            "f32-activation addmm must be bit-exact to matmul+add (no golden regression)"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::biased_dense_forward_f32_acts_match_matmul_add_bit_exact");
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn merge_dense_delta_adds_to_weight_and_zero_is_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);

        // A zero delta is a bit-exact no-op (`W + 0 == W`) — the scale-0 LoRA invariant.
        let mut lin = AdaptableLinear::dense(w.clone(), None);
        let base = lin.forward(&x).unwrap();
        lin.merge_dense_delta(&Array::from_slice(&[0.0f32; 4], &[2, 2]))
            .unwrap();
        assert!(array_eq(lin.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());

        // A nonzero delta is exactly `(W + δ)·x`.
        let delta = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let mut lin2 = AdaptableLinear::dense(w.clone(), None);
        lin2.merge_dense_delta(&delta).unwrap();
        let want = AdaptableLinear::dense(add(&w, &delta).unwrap(), None)
            .forward(&x)
            .unwrap();
        assert!(array_eq(lin2.forward(&x).unwrap(), &want, false)
            .unwrap()
            .item::<bool>());

        // Merging into a quantized base is rejected (must merge before quantization).
        let mut lin3 = AdaptableLinear::dense(
            Array::from_slice(
                &(0..4096).map(|i| i as f32 * 1e-3).collect::<Vec<_>>(),
                &[64, 64],
            ),
            None,
        );
        lin3.quantize(8, None).unwrap();
        assert!(lin3
            .merge_dense_delta(&Array::from_slice(&[0.0f32; 4096], &[64, 64]))
            .is_err());
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::merge_dense_delta_adds_to_weight_and_zero_is_noop");
    }

    #[test]
    fn conv_lora_delta_one_by_one_matches_hand_fold() {
        // sc-2919: a 1×1 conv LoRA (rank 2, in 2, out 2). down/up are `[*, *, 1, 1]`; the fused
        // delta is `Σ_r up[o,r]·down[r,i]`, scaled by alpha/rank. Hand-computed independently:
        //   down2 = [[1,2],[3,4]] (rank,in); up2 = [[5,6],[7,8]] (out,rank)
        //   δ[0,0]=5·1+6·3=23  δ[0,1]=5·2+6·4=34  δ[1,0]=7·1+8·3=31  δ[1,1]=7·2+8·4=46
        //   eff = alpha/rank = 4/2 = 2 → [[46,68],[62,92]]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
        let up = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 4.0, 2.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 2, 1, 1]);
        let want = Array::from_slice(&[46.0f32, 68.0, 62.0, 92.0], &[2, 2, 1, 1]);
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn conv_lora_delta_kxk_rank1_broadcasts_spatial_kernel() {
        // sc-2919: a 3×3-shaped (here 2×2) conv LoRA with rank 1 reduces to `δ[o,i,y,x] =
        // up[o]·down[0,i,y,x]` — proving the spatial kernel is preserved (not collapsed). in=1, out=2.
        //   down[0,0,:,:] = [[1,2],[3,4]]; up = [10, 20]
        //   δ[0] = 10·[1,2,3,4] = [10,20,30,40];  δ[1] = 20·[...] = [20,40,60,80]
        let down = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let up = Array::from_slice(&[10.0f32, 20.0], &[2, 1, 1, 1]);
        let delta = conv_lora_delta(&down, &up, 1.0, 1.0, 1.0).unwrap();
        assert_eq!(delta.shape(), &[2, 1, 2, 2]);
        let want = Array::from_slice(
            &[10.0f32, 20.0, 30.0, 40.0, 20.0, 40.0, 60.0, 80.0],
            &[2, 1, 2, 2],
        );
        assert!(all_close(&delta, &want, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>());
        // The user scale composes multiplicatively (scale 0 ⇒ a zero delta ⇒ no-op merge).
        let zero = conv_lora_delta(&down, &up, 1.0, 1.0, 0.0).unwrap();
        assert!(
            array_eq(&zero, Array::zeros::<f32>(&[2, 1, 2, 2]).unwrap(), false)
                .unwrap()
                .item::<bool>()
        );
    }

    #[test]
    fn conv_lora_delta_rejects_non_4d_factors() {
        // F-006: a malformed conv LoRA with 2-D factors must surface a typed error, not panic on the
        // `ds[2]`/`ds[3]` slice. And a rank mismatch between the factors is rejected too.
        let down2d = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let up = Array::from_slice(&[10.0f32, 20.0], &[2, 1, 1, 1]);
        let err = conv_lora_delta(&down2d, &up, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("4-D factors"), "got: {err}");

        // 4-D but mismatched rank: down rank 1, up rank 2.
        let down = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let up_bad = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2, 1, 1]);
        let err = conv_lora_delta(&down, &up_bad, 1.0, 1.0, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("rank mismatch"), "got: {err}");
    }

    #[test]
    fn merge_conv_delta_transposes_nchw_and_adds() {
        // sc-2919: NHWC weight `[out=1, kH=1, kW=1, in=2]` = [1, 2]; a 1×1 NCHW delta
        // `[out=1, in=2, kH=1, kW=1]` = [0.5, 0.25] transposes to NHWC [0.5, 0.25] and adds → [1.5, 2.25].
        let w = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 2]);
        let mut conv = AdaptableConv2d::new(w.clone(), None);
        let delta = Array::from_slice(&[0.5f32, 0.25], &[1, 2, 1, 1]);
        conv.merge_conv_delta(&delta).unwrap();
        let want = Array::from_slice(&[1.5f32, 2.25], &[1, 1, 1, 2]);
        assert!(array_eq(conv.weight(), &want, false)
            .unwrap()
            .item::<bool>());

        // A zero delta is a bit-exact no-op.
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::zeros::<f32>(&[1, 2, 1, 1]).unwrap())
            .unwrap();
        assert!(array_eq(conv2.weight(), &w, false).unwrap().item::<bool>());
    }

    #[test]
    fn merge_conv_delta_kxk_zero_is_noop_and_nonzero_lands_in_nhwc() {
        // sc-2919 regression: with a k×k (here 3×3, out≠in) kernel the NCHW→NHWC transpose is a real
        // permutation (not the trivial 1×1 case), so a zero delta must STILL be a bit-exact no-op — i.e.
        // the merge must not permute/scramble the weight. NHWC weight [out=2, kH=3, kW=3, in=4].
        let n = 2 * 3 * 3 * 4;
        let wv: Vec<f32> = (0..n).map(|i| i as f32 * 0.01 - 0.3).collect();
        let w = Array::from_slice(&wv, &[2, 3, 3, 4]);

        let mut conv = AdaptableConv2d::new(w.clone(), None);
        conv.merge_conv_delta(&Array::zeros::<f32>(&[2, 4, 3, 3]).unwrap())
            .unwrap();
        assert_eq!(
            conv.weight().as_slice::<f32>(),
            w.as_slice::<f32>(),
            "a zero k×k conv delta must be a bit-exact no-op (no permutation/scramble)"
        );

        // A nonzero NCHW delta must land at the matching NHWC position: δ_nchw[o,i,y,x] adds to
        // weight_nhwc[o,y,x,i]. Put a single spike at nchw (o=1,i=2,y=0,x=2) → nhwc (1,0,2,2).
        let mut spike = vec![0f32; n];
        // nchw flat index for [2,4,3,3] at (1,2,0,2) = ((1*4+2)*3+0)*3+2 = 56.
        spike[56] = 5.0;
        let mut conv2 = AdaptableConv2d::new(w.clone(), None);
        conv2
            .merge_conv_delta(&Array::from_slice(&spike, &[2, 4, 3, 3]))
            .unwrap();
        // nhwc flat index for [2,3,3,4] at (1,0,2,2) = ((1*3+0)*3+2)*4+2 = 46.
        let nhwc_idx = 46usize;
        let got = conv2.weight().as_slice::<f32>();
        for (j, (&g, &b)) in got.iter().zip(&wv).enumerate() {
            let want = if j == nhwc_idx { b + 5.0 } else { b };
            assert_eq!(
                g, want,
                "conv delta landed at wrong NHWC index (got change at {j})"
            );
        }
    }

    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn stacks_mixed_lora_and_lokr_summing_residuals() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        let lora = Adapter::Lora {
            a: Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
            b: Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]),
            scale: 0.5,
        };
        let lokr = Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.7,
        };
        let lora_r = lora.residual(&x).unwrap();
        let lokr_r = lokr.residual(&x).unwrap();
        lin.push(lora);
        lin.push(lokr);
        assert_eq!(lin.adapters().len(), 2);
        let expected = add(add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(
            all_close(lin.forward(&x).unwrap(), &expected, 1e-4, 1e-2, false)
                .unwrap()
                .item::<bool>()
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::stacks_mixed_lora_and_lokr_summing_residuals");
    }

    // ---- Structured (deferred) LoKr — the vec-trick (sc-10050) ------------------------------------

    /// The vec-trick `residual` must equal the materialized-delta `residual` for a **full** `w1⊗w2`
    /// LoKr, proving `vec(w1·X·w2ᵀ) == x·(w1⊗w2)ᵀ` under row-major kron ordering — the core identity
    /// this story rests on. `out = a·b = 2·3 = 6`, `in = c·d = 4·5 = 20`.
    #[test]
    fn structured_lokr_full_matches_materialized_delta() {
        let (a, b, c, d) = (2i32, 3, 4, 5);
        let (out, inp) = (a * b, c * d);
        let w1 = Array::from_slice(
            &(0..a * c)
                .map(|i| (i as f32 * 0.11).sin())
                .collect::<Vec<_>>(),
            &[a, c],
        );
        let w2 = Array::from_slice(
            &(0..b * d)
                .map(|i| (i as f32 * 0.07).cos())
                .collect::<Vec<_>>(),
            &[b, d],
        );
        let scale = 0.9f32;
        // Materialized reference delta and its residual `x·ΔWᵀ`.
        let delta = reconstruct_lokr_delta_scaled(
            scale,
            &[out, inp],
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap();
        // Structured factors — no [out,in] tensor built.
        let factors = build_lokr_factors(
            scale,
            &[out, inp],
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Float32,
        )
        .unwrap()
        .expect("a plain linear LoKr is deferrable");
        // The built factors are the SMALL Kronecker factors, never the [out,in] delta.
        assert_eq!(factors.w1.shape(), &[a, c]);
        assert_eq!(factors.w2.shape(), &[b, d]);

        let x = Array::from_slice(
            &(0..2 * inp)
                .map(|i| (i as f32 * 0.013 - 0.5).sin())
                .collect::<Vec<_>>(),
            &[2, inp],
        );
        let want = Adapter::Lokr { delta, scale: 1.0 }.residual(&x).unwrap();
        let got = factors.residual(&x).unwrap();
        assert_eq!(got.shape(), &[2, out]);
        // Metal reduced-precision matmul (~1e-3 relative) — the vec-trick and the materialized delta
        // take different matmul shapes, so compare at the Metal tolerance, not bit-exact.
        assert!(
            all_close(&got, &want, 2e-2, 2e-3, false)
                .unwrap()
                .item::<bool>(),
            "structured LoKr residual must match the materialized-delta residual"
        );
    }

    /// Same identity for a **decomposed** LoKr (`w1_a·w1_b`, `w2_a·w2_b`) — the low-rank inner factors
    /// are materialized only as the SMALL `[a,c]`/`[b,d]` Kronecker factors, never the `[out,in]` delta.
    #[test]
    fn structured_lokr_decomposed_matches_materialized_delta() {
        let (a, b, c, d, r) = (3i32, 2, 5, 4, 2);
        let (out, inp) = (a * b, c * d);
        let mk = |rows: i32, cols: i32, seed: f32| {
            Array::from_slice(
                &(0..rows * cols)
                    .map(|i| (i as f32 * 0.09 + seed).sin() * 0.3)
                    .collect::<Vec<_>>(),
                &[rows, cols],
            )
        };
        let (w1a, w1b) = (mk(a, r, 0.1), mk(r, c, 0.2)); // w1 = [a,c]
        let (w2a, w2b) = (mk(b, r, 0.3), mk(r, d, 0.4)); // w2 = [b,d]
        let scale = 1.3f32;
        let delta = reconstruct_lokr_delta_scaled(
            scale,
            &[out, inp],
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap();
        let factors = build_lokr_factors(
            scale,
            &[out, inp],
            None,
            Some(&w1a),
            Some(&w1b),
            None,
            None,
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap()
        .expect("a decomposed linear LoKr is deferrable");
        assert_eq!(factors.w1.shape(), &[a, c]);
        assert_eq!(factors.w2.shape(), &[b, d]);

        let x = Array::from_slice(
            &(0..inp)
                .map(|i| (i as f32 * 0.02).cos())
                .collect::<Vec<_>>(),
            &[1, inp],
        );
        let want = Adapter::Lokr { delta, scale: 1.0 }.residual(&x).unwrap();
        let got = factors.residual(&x).unwrap();
        assert!(
            all_close(&got, &want, 2e-2, 2e-3, false)
                .unwrap()
                .item::<bool>(),
            "decomposed structured LoKr must match the materialized delta"
        );
    }

    /// A **tucker/CP** `w2` (`w2_t2`, conv-only) is NOT deferrable via the 2-D vec-trick → `Ok(None)`,
    /// so the caller can fall back to materialization (dense) or a clear error (packed). Never a panic.
    #[test]
    fn structured_lokr_tucker_is_not_deferrable() {
        let t2 = Array::from_slice(&[0.1f32; 2 * 2 * 3 * 3], &[2, 2, 3, 3]);
        let w2a = Array::from_slice(&[0.2f32; 2 * 4], &[2, 4]);
        let w2b = Array::from_slice(&[0.3f32; 2 * 5], &[2, 5]);
        let w1 = Array::from_slice(&[0.4f32; 3 * 4], &[3, 4]);
        let got = build_lokr_factors(
            1.0,
            &[24, 180], // conv-ish; the tucker guard fires before any shape check
            Some(&w1),
            None,
            None,
            None,
            Some(&t2),
            Some(&w2a),
            Some(&w2b),
            Dtype::Float32,
        )
        .unwrap();
        assert!(
            got.is_none(),
            "tucker/CP LoKr must be reported non-deferrable"
        );
    }

    /// End-to-end on a **quantized (Q8)** base: a structured LoKr applies with a non-degenerate,
    /// finite output and the base stays packed — the memory-safe packed-tier path (sc-10050). The
    /// residual is computed only from the small factors (asserted above), so no `[out,in]` is built.
    #[test]
    #[ignore = "model inference; explicit model-proof workflow"]
    fn structured_lokr_applies_on_quantized_base() {
        let (a, b, c, d) = (4i32, 8, 8, 8); // out=32, in=64 (in % group_size(64)==0)
        let (out, inp) = (a * b, c * d);
        let wv: Vec<f32> = (0..out * inp).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut lin = AdaptableLinear::dense(Array::from_slice(&wv, &[out, inp]), None);
        lin.quantize(8, Some(64)).unwrap();
        assert!(lin.is_quantized());

        let w1 = Array::from_slice(
            &(0..a * c)
                .map(|i| (i as f32 * 0.05).sin() * 0.1)
                .collect::<Vec<_>>(),
            &[a, c],
        );
        let w2 = Array::from_slice(
            &(0..b * d)
                .map(|i| (i as f32 * 0.03).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[b, d],
        );
        let factors = build_lokr_factors(
            0.8,
            &lin.base_shape(),
            Some(&w1),
            None,
            None,
            Some(&w2),
            None,
            None,
            None,
            Dtype::Bfloat16,
        )
        .unwrap()
        .unwrap();
        lin.push(Adapter::LokrStructured { factors });

        let x = Array::from_slice(
            &(0..inp)
                .map(|i| (i as f32 * 0.01).sin())
                .collect::<Vec<_>>(),
            &[1, inp],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let y = lin.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, out]);
        assert!(lin.is_quantized(), "base must stay packed after LoKr apply");
        // Non-degenerate + finite.
        let ys = y.as_dtype(Dtype::Float32).unwrap();
        let sum_abs = ys.abs().unwrap().sum(None).unwrap().item::<f32>();
        assert!(
            sum_abs.is_finite() && sum_abs > 0.0,
            "output must be finite and non-zero"
        );
        crate::model_inference_device::record_model_inference_device();
        println!("\nMODEL_INFERENCE_BODY_COMPLETED adapters::tests::structured_lokr_applies_on_quantized_base");
    }
}

use super::*;
use crate::primitives::kv_cache::testing::{host, on_cpu, tok, ConcatReference};
use serde_json::json;
use std::collections::HashMap;

#[test]
fn attn_kv_matches_concat_reference_across_a_block_boundary() {
    on_cpu(|| {
        // The full-attention slot is a one-layer block cache: 3-token prefill + enough single
        // tokens to cross the block boundary twice, every returned K/V equal to a naive concat,
        // and `offset` reporting live positions rather than the padded buffer length.
        let block = 4;
        let mut slot = AttnKv::with_block_tokens(block);
        let mut reference = ConcatReference::new();
        assert_eq!(slot.offset(), 0);
        assert_eq!(slot.batch_size(), 0);

        let prefill_k = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[1, 1, 3, 2]);
        let prefill_v = Array::from_slice(&[9.0f32, 9.5, 8.0, 8.5, 7.0, 7.5], &[1, 1, 3, 2]);
        let (sk, sv) = slot.update(&prefill_k, &prefill_v).unwrap();
        let (rk, rv) = reference.update(&prefill_k, &prefill_v);
        assert_eq!(host(&sk), host(&rk));
        assert_eq!(host(&sv), host(&rv));
        assert_eq!(slot.offset(), 3);
        assert_eq!(slot.batch_size(), 1);

        for i in 0..7 {
            let k = tok(10.0 + i as f32);
            let v = tok(20.0 + i as f32);
            let (sk, sv) = slot.update(&k, &v).unwrap();
            let (rk, rv) = reference.update(&k, &v);
            assert_eq!(sk.shape(), rk.shape(), "update {i}: shape");
            assert_eq!(host(&sk), host(&rk), "update {i}: keys");
            assert_eq!(host(&sv), host(&rv), "update {i}: values");
            assert_eq!(slot.offset(), 4 + i, "update {i}: offset is live positions");
        }
        // 10 live positions in a 12-position buffer: offset must not read the padding.
        assert_eq!(slot.offset(), 10);
        assert_ne!(slot.offset(), 12);

        slot.reset().unwrap();
        assert_eq!(slot.offset(), 0);
        assert_eq!(slot.batch_size(), 0);
    })
}

#[test]
fn attn_kv_clone_snapshot_survives_in_place_updates_and_rolls_back() {
    on_cpu(|| {
        // A snapshot rollback (the MTP predictor cache, the engine's generic `SnapshotRollback`)
        // restores a `clone()` taken before the trial. While the trial writes in place, the
        // snapshot must stay exactly what it was, and continuing from it must match a reference
        // that never saw the trial — across a block boundary, so the trial both overwrites
        // padding and grows.
        let block = 4;
        let mut slot = AttnKv::with_block_tokens(block);
        let mut reference = ConcatReference::new();
        let prompt = Array::from_slice(&[0.0f32, 0.5, 1.0, 1.5, 2.0, 2.5], &[1, 1, 3, 2]);
        slot.update(&prompt, &prompt).unwrap();
        reference.update(&prompt, &prompt);

        let snapshot = slot.clone();
        let snapshot_k_before = host(reference.k.as_ref().unwrap());

        // Trial: 3 draft tokens (positions 3..6) — fills the block and grows into a second one.
        for i in 0..3 {
            let d = tok(100.0 + i as f32);
            slot.update(&d, &d).unwrap();
        }
        assert_eq!(slot.offset(), 6);
        assert_eq!(
            snapshot.offset(),
            3,
            "snapshot offset untouched by the trial"
        );
        let (snap_k, _) = snapshot.kv.peek(0).unwrap().unwrap();
        assert_eq!(
            host(&snap_k),
            snapshot_k_before,
            "snapshot contents untouched by the trial's in-place writes"
        );

        // Reject everything: restore the snapshot and replay the accepted path.
        let mut slot = snapshot;
        for i in 0..5 {
            let t = tok(200.0 + i as f32);
            let (sk, sv) = slot.update(&t, &t).unwrap();
            let (rk, rv) = reference.update(&t, &t);
            assert_eq!(host(&sk), host(&rk), "replay {i}: keys");
            assert_eq!(host(&sv), host(&rv), "replay {i}: values");
        }
        assert_eq!(slot.offset(), 8);
    })
}

#[test]
fn published_prism_norm_multiplier_matches_independent_rms_oracle() {
    let source = vec![1.0583496f32, 0.9418945, 1.3125, 1.957_031_3];
    let x = vec![0.25f32, -0.5, 1.25, -2.0];
    let weight = checkpoint_norm_weight(Array::from_slice(&source, &[4]), true).unwrap();
    let got = rms_norm(&Array::from_slice(&x, &[1, 4]), &weight, 1e-6).unwrap();
    mlx_rs::transforms::eval([&got]).unwrap();
    let inv = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-6)
        .sqrt()
        .recip();
    for (i, value) in got.as_slice::<f32>().iter().enumerate() {
        let expected = x[i] * inv * source[i];
        assert!(
            (value - expected).abs() < 1e-5,
            "lane {i}: {value} != {expected}"
        );
    }
    let dense = checkpoint_norm_weight(Array::from_slice(&[0.0583496f32], &[1]), false).unwrap();
    mlx_rs::transforms::eval([&dense]).unwrap();
    assert!((dense.item::<f32>() - source[0]).abs() < 1e-6);
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn headless_decoder_preserves_normalized_hidden_and_hybrid_cache_on_cpu() {
    on_cpu(|| {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let prefix = "language_model";
        let mut tensors = synthetic_tensors_with_prefix(&cfg, prefix);
        let lm = Qwen35Model::build_lazy(
            &Weights::from_map(tensors.clone()),
            prefix,
            cfg.clone(),
            None,
        )
        .unwrap();
        tensors.remove("lm_head.weight");
        let weights = Weights::from_map(tensors);
        assert!(Qwen35Model::build_lazy(&weights, prefix, cfg.clone(), None).is_err());
        // The public constructor explicitly verifies GPU views; test the shared loader and
        // decoder on their CPU stream without invoking that Metal-only boundary.
        let encoder = Qwen35Model::build_headless_lazy(&weights, prefix, cfg.clone()).unwrap();
        assert!(encoder.lm_head.is_none());
        let mut lm_cache = lm.new_cache();
        let mut cache = encoder.new_cache();
        let ids = Array::from_slice(&[1i32, 7, 3], &[1, 3]);
        let (expected, logits) = lm.hidden_and_logits(&ids, &mut lm_cache, 0).unwrap();
        let hidden = encoder.forward_hidden(&ids, &mut cache, 0).unwrap();
        assert_eq!(hidden.shape(), &[1, 3, cfg.hidden_size]);
        assert_eq!(logits.shape(), &[1, 3, cfg.vocab_size]);
        assert_eq!(host(&hidden), host(&expected));
        assert_eq!(cache.offset(), 3);
        let next_ids = Array::from_slice(&[42i32], &[1, 1]);
        let next = encoder.forward_hidden(&next_ids, &mut cache, 3).unwrap();
        let (expected, _) = lm.hidden_and_logits(&next_ids, &mut lm_cache, 3).unwrap();
        assert_eq!(host(&next), host(&expected));
        assert_eq!(cache.offset(), 4);
        cache.reset().unwrap();
        let repeated = encoder.forward_hidden(&ids, &mut cache, 0).unwrap();
        assert_eq!(host(&repeated), host(&hidden));
        assert!(encoder.forward(&ids, &mut encoder.new_cache(), 0).is_err());
        crate::model_inference_device::record_model_inference_device();
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::headless_decoder_preserves_normalized_hidden_and_hybrid_cache_on_cpu");
    });
}

#[test]
fn moss_frozen_hf_precision_matches_independent_cpu_cast_fixtures() {
    on_cpu(|| {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/moss_qwen_precision.json")).unwrap();
        let data = |case: &str, key: &str| -> Vec<f32> {
            fixture["cases"][case][key]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let bf = |v: &[f32], shape: &[i32]| {
            Array::from_slice(v, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let check = |case: &str, got: &Array, tolerance: f32| {
            let expected = data(case, "expected");
            let old = data(case, "old");
            assert!(expected
                .iter()
                .zip(&old)
                .any(|(a, b)| (a - b).abs() > tolerance));
            for (actual, expected) in host(got).iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{case}: {actual} != {expected}"
                );
            }
        };
        let hf = PrecisionProfile::MossFrozenHf;
        let native = PrecisionProfile::default();
        assert_eq!(native, PrecisionProfile::Native);
        let a = bf(&data("decay", "a"), &[1, 1, 16]);
        let log = bf(&data("decay", "a_log"), &[16]);
        let bias = bf(&data("decay", "bias"), &[16]);
        let decay = hf.decay(&a, &log, &bias).unwrap();
        assert_eq!(decay.dtype(), Dtype::Float32);
        check("decay", &decay, 2e-7);
        assert_eq!(
            host(&native.decay(&a, &log, &bias).unwrap()),
            host(&compute_g(&a, &log, &bias).unwrap())
        );
        assert_ne!(host(&decay), host(&native.decay(&a, &log, &bias).unwrap()));

        let q = bf(&data("query", "normalized"), &[1, 128]);
        let inverse = fixture["cases"]["query"]["inverse_head_dim"]
            .as_f64()
            .unwrap() as f32;
        let scaled = hf.scaled_query(&q, inverse).unwrap();
        assert_eq!(scaled.dtype(), Dtype::Float32);
        check("query", &scaled, 1e-8);
        assert_eq!(
            host(&native.scaled_query(&q, inverse).unwrap()),
            host(&scale(&q, inverse).unwrap())
        );
        assert_ne!(
            host(&scaled),
            host(&native.scaled_query(&q, inverse).unwrap())
        );

        let x = bf(&data("norm", "x"), &[2, 8]);
        let source = bf(&data("norm", "weight"), &[8]);
        let multiplier = hf.norm_weight(source.clone(), false).unwrap();
        let old_multiplier = native.norm_weight(source.clone(), false).unwrap();
        assert_eq!(multiplier.dtype(), Dtype::Float32);
        assert_eq!(old_multiplier.dtype(), Dtype::Bfloat16);
        assert_eq!(
            host(&old_multiplier),
            host(&checkpoint_norm_weight(source.clone(), false).unwrap())
        );
        assert_eq!(
            host(&native.norm_weight(source.clone(), true).unwrap()),
            host(&source)
        );
        let norm = hf.norm(&x, &multiplier, 1e-6).unwrap();
        assert_eq!(norm.dtype(), Dtype::Bfloat16);
        check("norm", &norm, 0.0);
        assert_ne!(
            host(&norm),
            host(
                &hf.norm(&x, &old_multiplier.as_dtype(Dtype::Float32).unwrap(), 1e-6)
                    .unwrap()
            )
        );
        assert_eq!(
            host(&native.norm(&x, &old_multiplier, 1e-6).unwrap()),
            host(&rms_norm(&x, &old_multiplier, 1e-6).unwrap())
        );
    });
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moss_delta_output_norm_matches_frozen_hf_cpu_and_preserves_native() {
    on_cpu(|| {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/moss_qwen_gated_norm.json")).unwrap();
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let prefix = "language_model";
        let weights = Weights::from_map(synthetic_tensors_with_prefix(&cfg, prefix));
        let mut model = Qwen35Model::from_weights_layout(
            &weights,
            prefix,
            cfg,
            None,
            None,
            false,
            PrecisionProfile::MossFrozenHf,
        )
        .unwrap();
        let delta = model
            .layers
            .iter_mut()
            .find_map(|layer| {
                if let Mixer::Delta(delta) = &mut layer.mixer {
                    Some(delta)
                } else {
                    None
                }
            })
            .unwrap();
        assert_eq!(delta.precision, PrecisionProfile::MossFrozenHf);
        let mut distinguished_native = false;
        for case in fixture["cases"].as_array().unwrap() {
            let values = |key: &str| -> Vec<f32> {
                case[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .flat_map(|value| {
                        value.as_array().map_or_else(
                            || vec![value.as_f64().unwrap() as f32],
                            |row| row.iter().map(|x| x.as_f64().unwrap() as f32).collect(),
                        )
                    })
                    .collect()
            };
            let bf = |key: &str, shape: &[i32]| {
                Array::from_slice(&values(key), shape)
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
            };
            let x = bf("x", &[1, 2]);
            let gate = bf("gate", &[1, 2]);
            delta.norm_weight = bf("weight", &[2]);
            delta.eps = case["eps"].as_f64().unwrap() as f32;
            let expected = values("expected");
            assert_ne!(
                expected,
                values("wrong_fused_norm_early_gate"),
                "fixture must reject incorrect casts"
            );
            delta.precision = PrecisionProfile::MossFrozenHf;
            let actual = delta.normalize_output(&x, &gate).unwrap();
            assert_eq!(actual.dtype(), Dtype::Bfloat16);
            assert_eq!(actual.shape(), x.shape());
            assert_eq!(host(&actual), expected, "{}", case["name"]);
            delta.precision = PrecisionProfile::Native;
            let native = delta.normalize_output(&x, &gate).unwrap();
            assert_eq!(
                host(&native),
                host(&rms_norm_gated(&x, &delta.norm_weight, &gate, delta.eps).unwrap()),
            );
            distinguished_native |= host(&native) != expected;
            if case["name"] == "sigmoid_rounding_dense_ffn" {
                let identity = Array::from_slice(&[1f32, 0., 0., 1.], &[2, 2])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                let gates = values("gate");
                let gate_weight = Array::from_slice(&[gates[0], 0., 0., gates[1]], &[2, 2])
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap();
                let ffn = Ffn::Dense(Mlp {
                    gate: Projection::load(gate_weight, None).unwrap(),
                    up: Projection::load(identity.clone(), None).unwrap(),
                    down: Projection::load(identity, None).unwrap(),
                });
                assert_eq!(
                    host(&ffn.forward(&x, PrecisionProfile::MossFrozenHf).unwrap()),
                    expected,
                );
                let native = ffn.forward(&x, PrecisionProfile::Native).unwrap();
                assert_eq!(host(&native), values("wrong_fused_norm_early_gate"));
            }
        }
        assert!(
            distinguished_native,
            "fixtures must discriminate the unchanged Native path"
        );
    });
    #[cfg(feature = "model-inference-proof")]
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::moss_delta_output_norm_matches_frozen_hf_cpu_and_preserves_native");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moss_frozen_hf_conv_matches_cpu_oracle_and_preserves_native_trace() {
    on_cpu(|| {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../testdata/moss_qwen_conv.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let values = |key: &str| -> Vec<f32> {
                case[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect()
            };
            let bf = |key: &str, shape: &[i32]| {
                Array::from_slice(&values(key), shape)
                    .as_dtype(Dtype::Bfloat16)
                    .unwrap()
            };
            let x = bf("x", &[1, 4, 1]);
            let weight = bf("weight", &[1, 4]);
            let state = bf("state", &[1, 3, 1]);
            let (actual, trace) = PrecisionProfile::MossFrozenHf
                .convolve(&x, &weight, &state)
                .unwrap();
            assert_eq!(actual.dtype(), Dtype::Bfloat16);
            assert_eq!(actual.shape(), x.shape());
            assert_eq!(host(&actual), values("expected"));
            if case["name"] == "tap_rounding" {
                assert_ne!(host(&actual), values("old"));
            } else {
                assert_ne!(host(&actual), values("wrong_post_silu_narrowing"));
            }
            let (native, native_trace) = PrecisionProfile::Native
                .convolve(&x, &weight, &state)
                .unwrap();
            let (old, old_trace) = causal_depthwise_conv_traced(&x, &weight, &state).unwrap();
            assert_eq!(host(&native), values("old"));
            assert_eq!(host(&native), host(&old));
            assert_eq!(trace.tokens(), 4);
            assert_eq!(trace.tail_len(), 3);
            for token in 0..4 {
                let tail = trace.tail_after(token).unwrap();
                assert_eq!(tail.dtype(), Dtype::Bfloat16);
                assert_eq!(tail.shape(), &[1, 3, 1]);
                assert_eq!(host(&tail), host(&native_trace.tail_after(token).unwrap()));
                assert_eq!(host(&tail), host(&old_trace.tail_after(token).unwrap()));
            }
            assert_eq!(host(&trace.tail_after(3).unwrap()), values("tail"));
            let first = x.take_axis(Array::from_slice(&[0i32, 1], &[2]), 1).unwrap();
            let second = x.take_axis(Array::from_slice(&[2i32, 3], &[2]), 1).unwrap();
            let (head, head_trace) = PrecisionProfile::MossFrozenHf
                .convolve(&first, &weight, &state)
                .unwrap();
            let (tail, tail_trace) = PrecisionProfile::MossFrozenHf
                .convolve(&second, &weight, &head_trace.tail_after(1).unwrap())
                .unwrap();
            assert_eq!(
                host(&concatenate_axis(&[head, tail], 1).unwrap()),
                values("expected")
            );
            assert_eq!(host(&tail_trace.tail_after(1).unwrap()), values("tail"));
        }
    });
    #[cfg(feature = "model-inference-proof")]
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::moss_frozen_hf_conv_matches_cpu_oracle_and_preserves_native_trace");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moss_frozen_hf_headless_profile_is_scoped_and_accounts_all_norm_arrays() {
    on_cpu(|| {
        let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
        let prefix = "language_model";
        let mut tensors = synthetic_tensors_with_prefix(&cfg, prefix);
        tensors.remove("lm_head.weight");
        let w = Weights::from_map(tensors);
        let native = Qwen35Model::build_headless_lazy(&w, prefix, cfg.clone()).unwrap();
        let hf = Qwen35Model::from_weights_layout(
            &w,
            prefix,
            cfg.clone(),
            None,
            None,
            false,
            PrecisionProfile::MossFrozenHf,
        )
        .unwrap();
        assert_eq!(native.precision, PrecisionProfile::Native);
        assert_eq!(hf.precision, PrecisionProfile::MossFrozenHf);
        assert!(hf.lm_head.is_none() && hf.mtp.is_none() && !hf.prism && !hf.quantized);
        assert_eq!(hf.norm.dtype(), Dtype::Float32);
        let mut norm_elements = cfg.hidden_size as usize;
        for layer in &hf.layers {
            assert_eq!(layer.precision, PrecisionProfile::MossFrozenHf);
            assert_eq!(layer.input_ln.dtype(), Dtype::Float32);
            assert_eq!(layer.post_ln.dtype(), Dtype::Float32);
            norm_elements += 2 * cfg.hidden_size as usize;
            match &layer.mixer {
                Mixer::Attn(a) => {
                    assert_eq!(a.precision, PrecisionProfile::MossFrozenHf);
                    assert_eq!(a.q_norm.dtype(), Dtype::Float32);
                    assert_eq!(a.k_norm.dtype(), Dtype::Float32);
                    norm_elements += 2 * cfg.head_dim as usize;
                }
                Mixer::Delta(d) => {
                    assert_eq!(d.precision, PrecisionProfile::MossFrozenHf);
                    assert_eq!(d.norm_weight.dtype(), Dtype::Bfloat16);
                    assert_eq!(d.conv_weight.dtype(), Dtype::Bfloat16);
                    assert_eq!(d.a_log.dtype(), Dtype::Bfloat16);
                    assert_eq!(d.dt_bias.dtype(), Dtype::Bfloat16);
                }
            }
        }
        let bytes = |m: &Qwen35Model| {
            m.param_groups()
                .iter()
                .flatten()
                .map(Array::nbytes)
                .sum::<usize>()
        };
        assert_eq!(bytes(&hf) - bytes(&native), norm_elements * 2);
        let ids = Array::from_slice(&[1i32, 7, 3], &[1, 3]);
        let hidden = hf.forward_hidden(&ids, &mut hf.new_cache(), 0).unwrap();
        assert_eq!(hidden.dtype(), Dtype::Bfloat16);
        assert_eq!(hidden.shape(), &[1, 3, cfg.hidden_size]);
        assert!(host(&hidden).iter().all(|x| x.is_finite()));
        assert!(Qwen35Model::from_weights_layout(
            &w,
            prefix,
            cfg.clone(),
            None,
            None,
            true,
            PrecisionProfile::MossFrozenHf
        )
        .is_err());
        let mut quantized = cfg.clone();
        quantized.quantization = Some(QuantSpec::q4());
        assert!(Qwen35Model::from_weights_layout(
            &w,
            prefix,
            quantized,
            None,
            None,
            false,
            PrecisionProfile::MossFrozenHf
        )
        .is_err());
        assert!(Qwen35Model::from_weights_layout(
            &w,
            prefix,
            Qwen35Config::from_json(&cfg_json_moe()).unwrap(),
            None,
            None,
            false,
            PrecisionProfile::MossFrozenHf
        )
        .is_err());
        assert!(
            with_compute_dtype(Dtype::Float32, || Qwen35Model::from_weights_layout(
                &w,
                prefix,
                cfg,
                None,
                None,
                false,
                PrecisionProfile::MossFrozenHf
            ))
            .is_err()
        );
        crate::model_inference_device::record_model_inference_device();
        #[cfg(feature = "model-inference-proof")]
        println!("\nMODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::moss_frozen_hf_headless_profile_is_scoped_and_accounts_all_norm_arrays");
    });
}

pub(crate) fn cfg_json() -> serde_json::Value {
    // 4 layers → schedule (interval 4): layers 0,1,2 linear, layer 3 full attention.
    json!({
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 32, "num_hidden_layers": 4, "intermediate_size": 64,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "vocab_size": 50, "rms_norm_eps": 1e-6, "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.5, "max_position_embeddings": 128,
            "tie_word_embeddings": false, "full_attention_interval": 4,
            "linear_num_value_heads": 4, "linear_num_key_heads": 2,
            "linear_key_head_dim": 4, "linear_value_head_dim": 4, "linear_conv_kernel_dim": 4
        },
        "vision_config": { "model_type": "qwen3_5" }
    })
}

#[test]
fn frozen_qwen38_dense_config_matches_qwen35_decoder() {
    let value: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../docs/reference/qwen38/config.json"
    )))
    .unwrap();
    let cfg = Qwen35Config::from_json(&value).unwrap();

    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_layers, 64);
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(cfg.num_heads, 24);
    assert_eq!(cfg.num_kv_heads, 4);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(cfg.max_position_embeddings, 262144);
    assert_eq!(cfg.mrope_section_resolved(), [11, 11, 10]);
    assert_eq!(cfg.mtp_num_hidden_layers, 1);
    assert!(!cfg.mtp_use_dedicated_embeddings);
    assert!(cfg.moe.is_none());
    assert!(cfg.quantization.is_none());
}

/// A deterministic small tensor `[shape]` (finite, non-degenerate).
fn t(map: &mut HashMap<String, Array>, key: &str, shape: &[i32]) {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.02).collect();
    map.insert(key.to_string(), Array::from_slice(&data, shape));
}

fn synthetic_weights_with_prefix(cfg: &Qwen35Config, pfx: &str) -> Weights {
    Weights::from_map(synthetic_tensors_with_prefix(cfg, pfx))
}

fn synthetic_tensors_with_prefix(cfg: &Qwen35Config, pfx: &str) -> HashMap<String, Array> {
    let h = cfg.hidden_size;
    let key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads;
    let value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads;
    let conv_dim = key_dim * 2 + value_dim;
    let mut m = HashMap::new();
    // The decoder can live under either VLM-wrapped or flat text-only roots; `lm_head` and
    // `mtp.*` remain at the checkpoint root in both layouts.
    t(
        &mut m,
        &format!("{pfx}.embed_tokens.weight"),
        &[cfg.vocab_size, h],
    );
    t(&mut m, &format!("{pfx}.norm.weight"), &[h]);
    t(&mut m, "lm_head.weight", &[cfg.vocab_size, h]);
    for i in 0..cfg.num_layers {
        let lp = |s: &str| format!("{pfx}.layers.{i}.{s}");
        t(&mut m, &lp("input_layernorm.weight"), &[h]);
        t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
        match &cfg.moe {
            None => {
                t(
                    &mut m,
                    &lp("mlp.gate_proj.weight"),
                    &[cfg.intermediate_size, h],
                );
                t(
                    &mut m,
                    &lp("mlp.up_proj.weight"),
                    &[cfg.intermediate_size, h],
                );
                t(
                    &mut m,
                    &lp("mlp.down_proj.weight"),
                    &[h, cfg.intermediate_size],
                );
            }
            Some(moe) => {
                let mi = moe.moe_intermediate_size;
                let si = moe.shared_expert_intermediate_size;
                t(
                    &mut m,
                    &lp("mlp.experts.gate_up_proj"),
                    &[moe.num_experts, 2 * mi, h],
                );
                t(
                    &mut m,
                    &lp("mlp.experts.down_proj"),
                    &[moe.num_experts, h, mi],
                );
                t(&mut m, &lp("mlp.gate.weight"), &[moe.num_experts, h]);
                t(&mut m, &lp("mlp.shared_expert.gate_proj.weight"), &[si, h]);
                t(&mut m, &lp("mlp.shared_expert.up_proj.weight"), &[si, h]);
                t(&mut m, &lp("mlp.shared_expert.down_proj.weight"), &[h, si]);
                t(&mut m, &lp("mlp.shared_expert_gate.weight"), &[1, h]);
            }
        }
        if cfg.is_linear(i) {
            // 4-way split projections (real qwen3_5 layout).
            t(
                &mut m,
                &lp("linear_attn.in_proj_qkv.weight"),
                &[conv_dim, h],
            );
            t(&mut m, &lp("linear_attn.in_proj_z.weight"), &[value_dim, h]);
            t(
                &mut m,
                &lp("linear_attn.in_proj_a.weight"),
                &[cfg.linear_num_value_heads, h],
            );
            t(
                &mut m,
                &lp("linear_attn.in_proj_b.weight"),
                &[cfg.linear_num_value_heads, h],
            );
            t(
                &mut m,
                &lp("linear_attn.conv1d.weight"),
                &[conv_dim, 1, cfg.linear_conv_kernel_dim],
            );
            t(
                &mut m,
                &lp("linear_attn.A_log"),
                &[cfg.linear_num_value_heads],
            );
            t(
                &mut m,
                &lp("linear_attn.dt_bias"),
                &[cfg.linear_num_value_heads],
            );
            t(
                &mut m,
                &lp("linear_attn.norm.weight"),
                &[cfg.linear_value_head_dim],
            );
            t(&mut m, &lp("linear_attn.out_proj.weight"), &[h, value_dim]);
        } else {
            t(
                &mut m,
                &lp("self_attn.q_proj.weight"),
                &[cfg.num_heads * cfg.head_dim * 2, h],
            );
            t(
                &mut m,
                &lp("self_attn.k_proj.weight"),
                &[cfg.num_kv_heads * cfg.head_dim, h],
            );
            t(
                &mut m,
                &lp("self_attn.v_proj.weight"),
                &[cfg.num_kv_heads * cfg.head_dim, h],
            );
            t(
                &mut m,
                &lp("self_attn.o_proj.weight"),
                &[h, cfg.num_heads * cfg.head_dim],
            );
            t(&mut m, &lp("self_attn.q_norm.weight"), &[cfg.head_dim]);
            t(&mut m, &lp("self_attn.k_norm.weight"), &[cfg.head_dim]);
        }
    }
    if cfg.mtp_num_hidden_layers == 1 {
        t(&mut m, "mtp.fc.weight", &[h, 2 * h]);
        t(&mut m, "mtp.pre_fc_norm_embedding.weight", &[h]);
        t(&mut m, "mtp.pre_fc_norm_hidden.weight", &[h]);
        t(&mut m, "mtp.norm.weight", &[h]);
        let lp = |s: &str| format!("mtp.layers.0.{s}");
        t(&mut m, &lp("input_layernorm.weight"), &[h]);
        t(&mut m, &lp("post_attention_layernorm.weight"), &[h]);
        t(
            &mut m,
            &lp("self_attn.q_proj.weight"),
            &[cfg.num_heads * cfg.head_dim * 2, h],
        );
        t(
            &mut m,
            &lp("self_attn.k_proj.weight"),
            &[cfg.num_kv_heads * cfg.head_dim, h],
        );
        t(
            &mut m,
            &lp("self_attn.v_proj.weight"),
            &[cfg.num_kv_heads * cfg.head_dim, h],
        );
        t(
            &mut m,
            &lp("self_attn.o_proj.weight"),
            &[h, cfg.num_heads * cfg.head_dim],
        );
        t(&mut m, &lp("self_attn.q_norm.weight"), &[cfg.head_dim]);
        t(&mut m, &lp("self_attn.k_norm.weight"), &[cfg.head_dim]);
        t(
            &mut m,
            &lp("mlp.gate_proj.weight"),
            &[cfg.intermediate_size, h],
        );
        t(
            &mut m,
            &lp("mlp.up_proj.weight"),
            &[cfg.intermediate_size, h],
        );
        t(
            &mut m,
            &lp("mlp.down_proj.weight"),
            &[h, cfg.intermediate_size],
        );
    }
    m
}

pub(crate) fn synthetic_weights(cfg: &Qwen35Config) -> Weights {
    synthetic_weights_with_prefix(cfg, "model.language_model")
}

pub(crate) fn cfg_json_mtp() -> serde_json::Value {
    let mut v = cfg_json();
    let tc = v["text_config"].as_object_mut().unwrap();
    tc.insert("mtp_num_hidden_layers".into(), json!(1));
    tc.insert("mtp_use_dedicated_embeddings".into(), json!(false));
    v
}

#[test]
fn config_parses_and_schedules_3_linear_1_full() {
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    assert_eq!(cfg.hidden_size, 32);
    assert_eq!(cfg.full_attention_interval, 4);
    assert_eq!(cfg.rotary_dim(), 4); // head_dim 8 * 0.5
                                     // 3 linear : 1 full.
    assert!(cfg.is_linear(0) && cfg.is_linear(1) && cfg.is_linear(2));
    assert!(!cfg.is_linear(3));
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn assembled_forward_produces_finite_logits() {
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let w = synthetic_weights(&cfg);
    let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
    let mut cache = model.new_cache();
    // A 5-token prefill.
    let ids = Array::from_slice(&[1i32, 7, 3, 42, 9], &[1, 5]);
    let logits = model.forward(&ids, &mut cache, 0).unwrap();
    assert_eq!(logits.shape(), &[1, 5, cfg.vocab_size]);
    for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
        assert!(x.is_finite(), "non-finite logit: {x}");
    }
    // The full-attention layer (layer 3) advanced the KV cache to 5 positions.
    assert_eq!(cache.offset(), 5);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::assembled_forward_produces_finite_logits");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn flat_text_and_wrapped_qwen35_match_target_and_mtp() {
    let wrapped_json = cfg_json_mtp();
    let flat_json = wrapped_json["text_config"].clone();
    let wrapped_cfg = Qwen35Config::from_json(&wrapped_json).unwrap();
    let flat_cfg = Qwen35Config::from_json(&flat_json).unwrap();
    let wrapped = Qwen35Model::from_weights(
        &synthetic_weights_with_prefix(&wrapped_cfg, "model.language_model"),
        "model.language_model",
        wrapped_cfg,
    )
    .unwrap();
    let flat = Qwen35Model::from_weights(
        &synthetic_weights_with_prefix(&flat_cfg, "model"),
        "model",
        flat_cfg,
    )
    .unwrap();
    let ids = Array::from_slice(&[1i32, 7, 3], &[1, 3]);
    let wrapped_logits = wrapped.forward(&ids, &mut wrapped.new_cache(), 0).unwrap();
    let flat_logits = flat.forward(&ids, &mut flat.new_cache(), 0).unwrap();
    let values = |a: &Array| {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    };
    assert_eq!(values(&wrapped_logits), values(&flat_logits));

    let shifted_ids = Array::from_slice(&[2i32, 3], &[1, 2]);
    let aligned_hidden = Array::from_slice(&vec![0f32; 2 * 32], &[1, 2, 32]);
    let (_, wrapped_draft) = wrapped
        .mtp_step(
            &shifted_ids,
            &aligned_hidden,
            &mut wrapped.new_mtp_cache().unwrap(),
            0,
        )
        .unwrap();
    let (_, flat_draft) = flat
        .mtp_step(
            &shifted_ids,
            &aligned_hidden,
            &mut flat.new_mtp_cache().unwrap(),
            0,
        )
        .unwrap();
    assert_eq!(values(&wrapped_draft), values(&flat_draft));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::flat_text_and_wrapped_qwen35_match_target_and_mtp");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn mtp_loads_all_frozen_components_and_executes() {
    let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let model = Qwen35Model::from_weights(
        &synthetic_weights(&cfg),
        "model.language_model",
        cfg.clone(),
    )
    .unwrap();
    assert!(model.has_mtp());

    let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
    let mut target_cache = model.new_cache();
    let (target_hidden, _) = model.hidden_and_logits(&ids, &mut target_cache, 0).unwrap();
    let shifted_ids = Array::from_slice(&[2i32, 3], &[1, 2]);
    let aligned_hidden = target_hidden
        .take_axis(Array::from_slice(&[0i32, 1], &[2]), 1)
        .unwrap();
    let mut mtp_cache = model.new_mtp_cache().unwrap();
    let (mtp_hidden, logits) = model
        .mtp_step(&shifted_ids, &aligned_hidden, &mut mtp_cache, 0)
        .unwrap();
    assert_eq!(mtp_hidden.shape(), &[1, 2, cfg.hidden_size]);
    assert_eq!(logits.shape(), &[1, cfg.vocab_size]);
    assert_eq!(mtp_cache.layers[0].offset(), 2);
    for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
        assert!(x.is_finite());
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::mtp_loads_all_frozen_components_and_executes");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn mtp_prefill_projects_only_last_row_and_warm_cache_matches_full() {
    let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let model = Qwen35Model::from_weights(
        &synthetic_weights(&cfg),
        "model.language_model",
        cfg.clone(),
    )
    .unwrap();
    let ids = Array::from_slice(&[1i32, 2, 3], &[1, 3]);
    let as_f32 = |array: &Array| {
        array
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec()
    };
    let mut full_cache = model.new_cache();
    let (full_hidden, full_logits) = model.hidden_and_logits(&ids, &mut full_cache, 0).unwrap();
    let mut prefill_cache = model.new_cache();
    let (hidden, last_logits) = model
        .prefill_hidden_and_last_logits(&ids, &mut prefill_cache, 0)
        .unwrap();
    assert_eq!(last_logits.shape(), &[1, cfg.vocab_size]);
    assert_eq!(as_f32(&hidden), as_f32(&full_hidden));
    let expected_last = full_logits
        .take_axis(Array::from_slice(&[2i32], &[1]), 1)
        .unwrap();
    assert_eq!(as_f32(&last_logits), as_f32(&expected_last));
    assert_eq!(prefill_cache.offset(), full_cache.offset());

    let embeds = model.embed_input_ids(&ids).unwrap();
    let positions = [0, 1, 2];
    let mut full_visual_cache = model.new_cache();
    let (visual_hidden, visual_logits) = model
        .hidden_and_logits_from_embeds_with_deepstack(
            &embeds,
            [&positions, &positions, &positions],
            &mut full_visual_cache,
            &[false; 3],
            &[],
        )
        .unwrap();
    let mut visual_cache = model.new_cache();
    let (visual_prefill_hidden, visual_last) = model
        .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
            &embeds,
            [&positions, &positions, &positions],
            &mut visual_cache,
            &[false; 3],
            &[],
        )
        .unwrap();
    assert_eq!(as_f32(&visual_prefill_hidden), as_f32(&visual_hidden));
    let visual_expected = visual_logits
        .take_axis(Array::from_slice(&[2i32], &[1]), 1)
        .unwrap();
    assert_eq!(as_f32(&visual_last), as_f32(&visual_expected));
    assert_eq!(visual_cache.offset(), full_visual_cache.offset());

    let shifted = embeds
        .take_axis(Array::from_slice(&[1i32, 2], &[2]), 1)
        .unwrap();
    let aligned = hidden
        .take_axis(Array::from_slice(&[0i32, 1], &[2]), 1)
        .unwrap();
    let seed_positions = [1, 2];
    let mut full_mtp_cache = model.new_mtp_cache().unwrap();
    let (full_mtp_hidden, last_mtp_logits) = model
        .mtp_step_from_embeds(
            &shifted,
            &aligned,
            &mut full_mtp_cache,
            [&seed_positions, &seed_positions, &seed_positions],
        )
        .unwrap();
    let all_mtp_logits = model
        .language_head()
        .unwrap()
        .forward(&full_mtp_hidden)
        .unwrap();
    let expected_mtp_last = all_mtp_logits
        .take_axis(Array::from_slice(&[1i32], &[1]), 1)
        .unwrap();
    assert_eq!(last_mtp_logits.shape(), &[1, cfg.vocab_size]);
    assert_eq!(as_f32(&last_mtp_logits), as_f32(&expected_mtp_last));
    let mut warm_mtp_cache = model.new_mtp_cache().unwrap();
    let warm_hidden = model
        .mtp_warm_from_embeds(
            &shifted,
            &aligned,
            &mut warm_mtp_cache,
            [&seed_positions, &seed_positions, &seed_positions],
        )
        .unwrap();
    assert_eq!(as_f32(&warm_hidden), as_f32(&full_mtp_hidden));
    assert_eq!(warm_mtp_cache.layers[0].offset(), 2);
    assert_eq!(full_mtp_cache.layers[0].offset(), 2);
    let previous = hidden
        .take_axis(Array::from_slice(&[2i32], &[1]), 1)
        .unwrap();
    let next = Array::from_slice(&[1i32], &[1, 1]);
    let (_, full_next) = model
        .mtp_step(&next, &previous, &mut full_mtp_cache, 3)
        .unwrap();
    let (_, warm_next) = model
        .mtp_step(&next, &previous, &mut warm_mtp_cache, 3)
        .unwrap();
    assert_eq!(as_f32(&warm_next), as_f32(&full_next));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::mtp_prefill_projects_only_last_row_and_warm_cache_matches_full");
}

/// E2 (sc-24432 feature-end review): a config declaring a native head the snapshot stores
/// no tensor of — or a variant this runtime does not run (two predictor layers, dedicated MTP
/// embeddings) — builds the target plain with a named `mtp:` fallback and prices no head
/// state; a PARTIAL `mtp.*` set still fails (integrity); `mtp.*` tensors under a config that
/// disables MTP stay a refused contradiction.
#[test]
fn an_mtp_head_the_snapshot_cannot_run_is_a_named_fallback_but_a_partial_one_fails() {
    let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let base_cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let weights = synthetic_weights(&base_cfg);
    let model = Qwen35Model::from_weights(&weights, "model.language_model", cfg.clone())
        .expect("a configured head the snapshot does not carry loads the target plain");
    assert!(!model.has_mtp());
    let why = model.mtp_fallback().expect("the fallback is named");
    assert!(
        why.starts_with("mtp: ") && why.contains("stores no `mtp.*` tensor"),
        "{why}"
    );
    assert_eq!(
        model.config().mtp_num_hidden_layers,
        0,
        "no head state priced"
    );

    let full = synthetic_weights(&cfg);
    let tensors = |keep: &dyn Fn(&str) -> bool| {
        Weights::from_map(
            full.keys()
                .filter(|k| keep(k))
                .map(|k| (k.to_string(), full.get(k).unwrap().clone()))
                .collect(),
        )
    };
    for (edit, named) in [
        (
            (|c: &mut Qwen35Config| c.mtp_num_hidden_layers = 2) as fn(&mut Qwen35Config),
            "declares 2 predictor layers",
        ),
        (
            |c: &mut Qwen35Config| c.mtp_use_dedicated_embeddings = true,
            "dedicated MTP embeddings",
        ),
    ] {
        let mut variant = cfg.clone();
        edit(&mut variant);
        let model = Qwen35Model::from_weights(&tensors(&|_| true), "model.language_model", variant)
            .unwrap_or_else(|e| panic!("{named}: the target still loads (E2): {e}"));
        assert!(!model.has_mtp(), "{named}");
        let why = model.mtp_fallback().unwrap_or_default();
        assert!(why.starts_with("mtp: ") && why.contains(named), "{why}");
    }
    let partial = tensors(&|k| k != "mtp.fc.weight");
    let Err(err) = Qwen35Model::from_weights(&partial, "model.language_model", cfg.clone()) else {
        panic!("a partial `mtp.*` set fails the load");
    };
    assert!(err.to_string().contains("mtp.fc.weight"), "{err}");
    let complete = Qwen35Model::from_weights(&full, "model.language_model", cfg).unwrap();
    assert!(complete.has_mtp() && complete.mtp_fallback().is_none());

    let mtp_cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let base_cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let err = Qwen35Model::from_weights(
        &synthetic_weights(&mtp_cfg),
        "model.language_model",
        base_cfg,
    )
    .unwrap_err();
    assert!(err.to_string().contains("config disables MTP"), "{err}");
}

fn greedy_config(max_new_tokens: usize) -> crate::decode::GenerationConfig {
    crate::decode::GenerationConfig {
        max_new_tokens,
        sampling: crate::primitives::sampler::SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            presence_penalty_prompt_len: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        seed: Some(7),
        stop_tokens: Vec::new(),
    }
}

fn bits(a: &Array) -> Vec<u32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

const PRISM_PROMPT: [i32; 7] = [1, 5, 9, 2, 33, 17, 4];

/// Story sc-24444 AC1: on a Prism fixture (packed two-bit projections with random codes and
/// signs, one Gated DeltaNet and one attention layer) the fused rotation leaves the prefill
/// logits bit-identical and the greedy tokens unchanged.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn prism_greedy_tokens_and_logits_are_unchanged_by_the_fused_rotation() {
    use crate::decode::{generate, CancelFlag};
    use crate::primitives::prism::tests::with_unfused_rotation;
    use crate::primitives::prism::{recording_rotation_routes, RotationRoute};

    let fixture = crate::synthetic::prism_qwen35(11);
    let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
    let model = Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
    assert!(model.is_prism());
    let run = || {
        let mut cache = model.new_cache();
        let (hidden, logits) = model
            .hidden_and_logits(&crate::primitives::input_ids(&PRISM_PROMPT), &mut cache, 0)
            .unwrap();
        let out = generate(
            &model,
            &PRISM_PROMPT,
            &greedy_config(24),
            &CancelFlag::new(),
            &mut |_| {},
        )
        .unwrap();
        (bits(&hidden), bits(&logits), out.tokens)
    };
    // Count the routes, not just the numbers: the two paths are bit-identical by
    // construction, so only the route shows the fused kernel actually ran.
    let fused_count = |routes: &[RotationRoute]| {
        routes
            .iter()
            .filter(|r| **r == RotationRoute::Fused)
            .count()
    };
    let (fused, fused_routes) = mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
        crate::model_inference_device::record_model_inference_device();
        recording_rotation_routes(run)
    });
    let (unfused, unfused_routes) = mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
        crate::model_inference_device::record_model_inference_device();
        recording_rotation_routes(|| with_unfused_rotation(run))
    });
    assert!(
        fused_count(&fused_routes) > 0,
        "the fused rotation never ran: {fused_routes:?}"
    );
    assert!(
        !fused_routes.contains(&RotationRoute::Unfused),
        "every rotation at the fixture's widths is fused: {fused_routes:?}"
    );
    assert_eq!(
        fused_count(&unfused_routes),
        0,
        "the oracle run used the kernel"
    );
    assert!(!unfused_routes.is_empty());
    assert!(fused.0 == unfused.0, "prefill hidden states differ");
    assert!(fused.1 == unfused.1, "prefill logits differ");
    assert_eq!(fused.2, unfused.2, "greedy tokens differ");
    let distinct: std::collections::BTreeSet<_> = fused.2.iter().collect();
    assert!(
        distinct.len() > 2,
        "fixture must not be degenerate: {:?}",
        fused.2
    );

    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::prism_greedy_tokens_and_logits_are_unchanged_by_the_fused_rotation");
}

/// Story sc-24444 AC2 (model level): a companion MTP head attached to a Prism target borrows
/// the packed embedding and `lm_head`, drafts, and every depth emits exactly the target's
/// greedy tokens.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn prism_target_with_a_companion_head_drafts_and_keeps_greedy_tokens() {
    use crate::decode::{generate, generate_qwen35_mtp, CancelFlag};

    let fixture = crate::synthetic::prism_qwen35(11);
    let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
    let mut model = Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
    assert!(!model.has_mtp());
    assert!(model.new_mtp_cache().is_none());
    let head = crate::test_fixture::Fixture::new("mlx-companion-head-", None);
    crate::synthetic::write_companion_head(&head, &fixture.config["text_config"], 3, None);
    model.attach_companion_mtp(&head).unwrap();
    assert!(model.has_mtp());
    assert!(
        model.is_prism(),
        "the target stays the packed Prism decoder"
    );

    let greedy = greedy_config(24);
    let plain = generate(
        &model,
        &PRISM_PROMPT,
        &greedy,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    for depth in [1, 3, 7] {
        let (speculative, stats) = generate_qwen35_mtp(
            &model,
            &PRISM_PROMPT,
            &greedy,
            depth,
            &CancelFlag::new(),
            &mut |_| {},
            None,
            None,
        )
        .unwrap();
        assert_eq!(speculative.tokens, plain.tokens, "depth {depth}");
        assert!(stats.proposed > 0, "depth {depth}: the head drafted");
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::prism_target_with_a_companion_head_drafts_and_keeps_greedy_tokens");
}

/// Story sc-24444 AC2 / E2: a head whose geometry disagrees with the target is refused with a
/// reason naming each disagreement, and the target is left exactly as it was — still loaded,
/// no predictor, same greedy tokens.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_mismatched_companion_head_is_refused_by_name_and_leaves_the_target_intact() {
    use crate::decode::{generate, CancelFlag};

    let fixture = crate::synthetic::prism_qwen35(11);
    let text = &fixture.config["text_config"];
    let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
    let mut model = Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
    let greedy = greedy_config(12);
    let before = generate(
        &model,
        &PRISM_PROMPT,
        &greedy,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();

    // The head's own config disagrees (a head for another model size).
    let mut other = text.clone();
    other["intermediate_size"] = json!(128);
    other["num_key_value_heads"] = json!(2);
    let config_mismatch = crate::test_fixture::Fixture::new("mlx-head-geometry-", None);
    crate::synthetic::write_companion_head(&config_mismatch, &other, 3, None);
    let err = model
        .attach_companion_mtp(&config_mismatch)
        .unwrap_err()
        .to_string();
    assert!(err.contains("geometry does not match"), "{err}");
    assert!(err.contains("intermediate_size 128 != target 256"), "{err}");
    assert!(err.contains("num_key_value_heads 2 != target 1"), "{err}");

    // Each field the predictor computes with but no tensor shape reveals is its own named
    // refusal: `vocab_size` is the only guard against a head for another tokenizer.
    for (field, value, named) in [
        ("vocab_size", json!(65), "vocab_size 65 != target 64"),
        (
            "partial_rotary_factor",
            json!(0.25),
            "rotary_dim 16 != target 32",
        ),
        (
            "rope_theta",
            json!(1000000.0),
            "rope_theta 1000000 != target 10000000",
        ),
        (
            "rms_norm_eps",
            json!(1e-5),
            "rms_norm_eps 0.00001 != target 0.000001",
        ),
    ] {
        let mut other = text.clone();
        other[field] = value;
        let head = crate::test_fixture::Fixture::new("mlx-head-field-", None);
        crate::synthetic::write_companion_head(&head, &other, 3, None);
        let err = model.attach_companion_mtp(&head).unwrap_err().to_string();
        assert!(err.contains("geometry does not match"), "{field}: {err}");
        assert!(err.contains(named), "{field}: {err}");
    }

    // The config matches but a stored tensor does not.
    let tensor_mismatch = crate::test_fixture::Fixture::new("mlx-head-tensors-", None);
    crate::synthetic::write_companion_head(&tensor_mismatch, text, 3, Some(128));
    let err = model
        .attach_companion_mtp(&tensor_mismatch)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("`layers.0.self_attn.q_proj` is [128, 128], the target needs [256, 128]"),
        "{err}"
    );

    // Not a companion head at all.
    let wrong_type = crate::test_fixture::Fixture::new("mlx-head-type-", None);
    crate::synthetic::write_companion_head(&wrong_type, text, 3, None);
    let path = wrong_type.join("config.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    value["model_type"] = json!("qwen3_5");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let err = model
        .attach_companion_mtp(&wrong_type)
        .unwrap_err()
        .to_string();
    assert!(err.contains("must be `qwen3_5_mtp`"), "{err}");

    assert!(!model.has_mtp());
    let after = generate(
        &model,
        &PRISM_PROMPT,
        &greedy,
        &CancelFlag::new(),
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(after.tokens, before.tokens);

    // A target with its own native predictor never takes a second one.
    let mut native = crate::decode::engine::tests::qwen35(true);
    let native_head = crate::test_fixture::Fixture::new("mlx-head-native-", None);
    crate::synthetic::write_companion_head(&native_head, text, 3, None);
    let err = native
        .attach_companion_mtp(&native_head)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("already carries its own MTP predictor"),
        "{err}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::a_mismatched_companion_head_is_refused_by_name_and_leaves_the_target_intact");
}

/// Story sc-24444: a companion head's RMSNorm vectors follow the zero-centred Qwen3.8
/// checkpoint convention (`1 + w`) whatever the target's own convention — a Prism target's
/// norms are direct multipliers, the head's are not (the published Qwen3.8 heads'
/// `pre_fc_norm_embedding` values are all negative, mean ≈ −0.46).
#[test]
fn a_companion_heads_norms_are_applied_as_one_plus_w() {
    let fixture = crate::synthetic::prism_qwen35(11);
    let cfg = Qwen35Config::from_json(&fixture.config).unwrap();
    let mut model = Qwen35Model::from_prism_weights(&fixture.weights, cfg, &fixture.pack).unwrap();
    let head = crate::test_fixture::Fixture::new("mlx-head-norm-", None);
    crate::synthetic::write_companion_head(&head, &fixture.config["text_config"], 3, None);
    let path = head.join("model.safetensors");
    let mut tensors = Array::load_safetensors(&path).unwrap();
    // The load is lazy: materialize every tensor before overwriting the file it reads.
    mlx_rs::transforms::eval(tensors.values()).unwrap();
    tensors.insert(
        "pre_fc_norm_embedding.weight".into(),
        Array::from_slice(&[-0.5f32; 128], &[128])
            .as_dtype(Dtype::Bfloat16)
            .unwrap(),
    );
    Array::save_safetensors(tensors.iter().map(|(k, v)| (k.as_str(), v)), None, &path).unwrap();
    model.attach_companion_mtp(&head).unwrap();
    let applied = model
        .mtp
        .as_ref()
        .unwrap()
        .pre_fc_norm_embedding
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_eq!(applied.as_slice::<f32>(), [0.5f32; 128]);
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn mtp_generation_covers_verification_rollback_stop_and_cancel() {
    use crate::decode::{
        generate, generate_qwen35_mtp, generate_speculative, CancelFlag, ConstraintMask,
        EngineOptions, FinishReason, GenerationConfig, MtpProposer, RewindableConstraintMask,
        SpeculativePrompt,
    };
    use crate::primitives::sampler::SamplingParams;

    struct RecordingConstraint {
        allow: Vec<bool>,
        accepted: Vec<i32>,
        rewinds: usize,
    }

    impl ConstraintMask for RecordingConstraint {
        fn allowed(&mut self) -> &[bool] {
            &self.allow
        }

        fn accept(&mut self, token: i32) {
            self.accepted.push(token);
        }
    }

    impl RewindableConstraintMask for RecordingConstraint {
        fn checkpoint(&self) -> usize {
            self.accepted.len()
        }

        fn rewind(&mut self, checkpoint: usize) {
            self.accepted.truncate(checkpoint);
            self.rewinds += 1;
        }
    }

    let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let vocab_size = cfg.vocab_size as usize;
    let model =
        Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap();
    let greedy = GenerationConfig {
        max_new_tokens: 8,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            presence_penalty_prompt_len: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        seed: Some(7),
        stop_tokens: Vec::new(),
    };
    let target_only =
        generate(&model, &[1, 2, 3], &greedy, &CancelFlag::new(), &mut |_| {}).unwrap();
    let mut timed = generate_speculative(
        &model,
        &mut MtpProposer::new(),
        SpeculativePrompt::Tokens(&[1, 2, 3]),
        &greedy,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        EngineOptions {
            prefill_clock: Some(std::time::Instant::now()),
            ..EngineOptions::default()
        },
    )
    .unwrap();
    let _timings = timed
        .take_timer()
        .expect("a timed run keeps its timer")
        .finish();
    let (speculative, stats) = (timed.output, timed.stats);
    assert_eq!(
        speculative.tokens, target_only.tokens,
        "adversarial MTP drafts must not change greedy target output"
    );
    assert!(stats.proposed > 0);
    assert!(
        stats.accepted < stats.proposed,
        "fixture must exercise rejection and clone/replay rollback: {stats:?}"
    );
    assert!(stats.forwards >= 3, "rejection must add a replay forward");

    let mut constrained = RecordingConstraint {
        allow: vec![true; vocab_size],
        accepted: Vec::new(),
        rewinds: 0,
    };
    let (constrained_greedy, constrained_stats) = generate_qwen35_mtp(
        &model,
        &[1, 2, 3],
        &greedy,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        Some(&mut constrained),
        None,
    )
    .unwrap();
    assert_eq!(constrained_greedy.tokens, target_only.tokens);
    assert!(constrained_stats.accepted < constrained_stats.proposed);
    assert_eq!(constrained.accepted, constrained_greedy.tokens);
    assert!(
        constrained.rewinds >= 2,
        "proposal and verification must both rewind provisional constraint state"
    );

    let mut with_stop = greedy.clone();
    with_stop.stop_tokens = vec![target_only.tokens[2]];
    let (stopped, _) = generate_qwen35_mtp(
        &model,
        &[1, 2, 3],
        &with_stop,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        None,
    )
    .unwrap();
    assert_eq!(stopped.tokens, target_only.tokens[..2]);
    assert_eq!(stopped.finish_reason, FinishReason::StopToken);

    let cancel = CancelFlag::new();
    let cancel_from_sink = cancel.clone();
    let (cancelled, _) = generate_qwen35_mtp(
        &model,
        &[1, 2, 3],
        &greedy,
        2,
        &cancel,
        &mut |event| {
            if matches!(event, crate::decode::StreamEvent::Token { step: 0, .. }) {
                cancel_from_sink.cancel();
            }
        },
        None,
        None,
    )
    .unwrap();
    assert_eq!(cancelled.tokens.len(), 1);
    assert_eq!(cancelled.finish_reason, FinishReason::Cancelled);

    let stochastic = GenerationConfig {
        sampling: SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 20,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            presence_penalty_prompt_len: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        ..greedy
    };
    let (sampled, sampled_stats) = generate_qwen35_mtp(
        &model,
        &[1, 2, 3],
        &stochastic,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        None,
        None,
    )
    .unwrap();
    assert_eq!(sampled.tokens.len(), 8);
    assert!(sampled_stats.proposed > 0);
    assert!(sampled_stats.accepted <= sampled_stats.proposed);

    let mut sampled_constraint = RecordingConstraint {
        allow: vec![true; vocab_size],
        accepted: Vec::new(),
        rewinds: 0,
    };
    let (constrained_sampled, _) = generate_qwen35_mtp(
        &model,
        &[1, 2, 3],
        &stochastic,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        Some(&mut sampled_constraint),
        None,
    )
    .unwrap();
    assert_eq!(
        constrained_sampled.tokens, sampled.tokens,
        "a permissive transactional constraint must preserve stochastic p/q sampling"
    );
    assert_eq!(sampled_constraint.accepted, constrained_sampled.tokens);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::mtp_generation_covers_verification_rollback_stop_and_cancel");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn multimodal_mtp_seed_uses_fused_embeddings_and_matches_text_equivalent() {
    use std::time::Instant;

    use crate::decode::{
        generate_speculative, CancelFlag, EngineOptions, GenerationConfig, MtpProposer,
        Qwen35MtpMultimodalPrompt, SpeculativePrompt,
    };
    use crate::primitives::sampler::SamplingParams;

    let cfg = Qwen35Config::from_json(&cfg_json_mtp()).unwrap();
    let model =
        Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap();
    let ids = [1, 2, 3];
    let embeds = model
        .embed_input_ids(&crate::primitives::input_ids(&ids))
        .unwrap();
    let positions = [0, 1, 2];
    let prompt = Qwen35MtpMultimodalPrompt {
        input_ids: &ids,
        embeddings: &embeds,
        positions: [&positions, &positions, &positions],
        visual_pos_mask: &[false, false, false],
        deepstack: &[],
        continuation_delta: 0,
    };
    let config = GenerationConfig {
        max_new_tokens: 6,
        sampling: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            presence_penalty: 0.0,
            presence_penalty_generated_only: false,
            presence_penalty_prompt_len: 0,
            repetition_penalty: 1.0,
            repetition_context: 0,
        },
        seed: Some(17),
        stop_tokens: Vec::new(),
    };
    let clock = || EngineOptions {
        prefill_clock: Some(Instant::now()),
        ..EngineOptions::default()
    };
    let mut text = generate_speculative(
        &model,
        &mut MtpProposer::new(),
        SpeculativePrompt::Tokens(&ids),
        &config,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        clock(),
    )
    .unwrap();
    // The multimodal route: the caller prefills the fused embeddings under explicit M-RoPE
    // positions and the proposer seeds from those same embeddings.
    let mut visual_cache = model.new_cache();
    let (visual_hidden, visual_logits) = model
        .prefill_hidden_and_last_logits_from_embeds_with_deepstack(
            prompt.embeddings,
            prompt.positions,
            &mut visual_cache,
            prompt.visual_pos_mask,
            prompt.deepstack,
        )
        .unwrap();
    let mut visual = generate_speculative(
        &model,
        &mut MtpProposer::multimodal(&prompt),
        SpeculativePrompt::Prefilled {
            cache: &mut visual_cache,
            logits: visual_logits,
            hidden: Some(visual_hidden),
            history: prompt.input_ids,
            position_delta: prompt.continuation_delta,
        },
        &config,
        2,
        &CancelFlag::new(),
        &mut |_| {},
        clock(),
    )
    .unwrap();
    assert_eq!(visual.output.tokens, text.output.tokens);
    assert_eq!(visual.stats, text.stats);
    let _ = text.take_timer().unwrap().finish();
    let _ = visual.take_timer().unwrap().finish();

    // Alter the middle row as an encoded visual feature. Both the target prefill and shifted
    // MTP seed must consume that fused row rather than re-embedding placeholder token id 2.
    let shape = embeds.shape().to_vec();
    let hidden = shape[2] as usize;
    let mut fused_values = embeds
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    for (i, value) in fused_values[hidden..2 * hidden].iter_mut().enumerate() {
        *value += 0.25 + i as f32 * 0.01;
    }
    let fused = Array::from_slice(&fused_values, &shape);
    let mut text_target_cache = model.new_cache();
    let (text_hidden, _) = model
        .hidden_and_logits(
            &crate::primitives::input_ids(&ids),
            &mut text_target_cache,
            0,
        )
        .unwrap();
    let mut fused_target_cache = model.new_cache();
    let (fused_hidden, _) = model
        .hidden_and_logits_from_embeds_with_deepstack(
            &fused,
            [&positions, &positions, &positions],
            &mut fused_target_cache,
            &[false, true, false],
            &[],
        )
        .unwrap();
    let seed_indices = Array::from_slice(&[0i32, 1], &[2]);
    let shifted_indices = Array::from_slice(&[1i32, 2], &[2]);
    let text_aligned = text_hidden.take_axis(&seed_indices, 1).unwrap();
    let fused_aligned = fused_hidden.take_axis(&seed_indices, 1).unwrap();
    let fused_shifted = fused.take_axis(&shifted_indices, 1).unwrap();
    let mut text_mtp_cache = model.new_mtp_cache().unwrap();
    let (_, text_mtp_logits) = model
        .mtp_step(
            &crate::primitives::input_ids(&ids[1..]),
            &text_aligned,
            &mut text_mtp_cache,
            1,
        )
        .unwrap();
    let mut fused_mtp_cache = model.new_mtp_cache().unwrap();
    let seed_positions = [1, 2];
    let (_, fused_mtp_logits) = model
        .mtp_step_from_embeds(
            &fused_shifted,
            &fused_aligned,
            &mut fused_mtp_cache,
            [&seed_positions, &seed_positions, &seed_positions],
        )
        .unwrap();
    let text_logits_f32 = text_mtp_logits.as_dtype(Dtype::Float32).unwrap();
    let fused_logits_f32 = fused_mtp_logits.as_dtype(Dtype::Float32).unwrap();
    let text_logits = text_logits_f32.as_slice::<f32>();
    let fused_logits = fused_logits_f32.as_slice::<f32>();
    assert!(
        text_logits
            .iter()
            .zip(fused_logits)
            .any(|(text, fused)| (text - fused).abs() > 1.0e-5),
        "an encoded visual row must change the MTP seed distribution"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::multimodal_mtp_seed_uses_fused_embeddings_and_matches_text_equivalent");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn decode_after_prefill_advances_cache() {
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let model = Qwen35Model::from_weights(
        &synthetic_weights(&cfg),
        "model.language_model",
        cfg.clone(),
    )
    .unwrap();
    let mut cache = model.new_cache();
    model
        .forward(&Array::from_slice(&[1i32, 2, 3], &[1, 3]), &mut cache, 0)
        .unwrap();
    assert_eq!(cache.offset(), 3);
    // One decode step at offset 3.
    let logits = model
        .forward(&Array::from_slice(&[4i32], &[1, 1]), &mut cache, 3)
        .unwrap();
    assert_eq!(logits.shape(), &[1, 1, cfg.vocab_size]);
    assert_eq!(cache.offset(), 4);
    for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
        assert!(x.is_finite());
    }
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::decode_after_prefill_advances_cache"
    );
}

/// The whole Gated DeltaNet layer, validated against a numeric oracle from the exact
/// `Qwen3_5GatedDeltaNet.forward` reference (4-way in-projection → short conv → contiguous q|k|v
/// split → L2-norm + q-scale → GQA delta recurrence → gated RMS-norm(z) → out-proj).
///
/// **Single token (S=1).** MLX routes f32 matmuls through an exact GEMV when M=1 but a
/// reduced-precision (bf16-class) GEMM when M>1, so a multi-token f32 oracle floors at ~6e-3
/// regardless of correctness. With S=1 every projection is a GEMV, so the whole layer runs in
/// exact f32 and the match is tight (≈1e-5) — pinning the *structural* assembly precisely. The
/// conv's multi-tap and the recurrence's multi-step carry are covered tightly by the primitive
/// oracles (sc-7627); cross-token cache carry by `prefill_equals_stepwise_decode`. Regenerate the
/// fixture with `/tmp/gen_s1.py` (see story sc-7629).
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn deltanet_layer_matches_qwen3_5_reference() {
    let json: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/qwen35_deltanet_oracle.json")).unwrap();
    let arr = |k: &str| -> Vec<f32> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    // Dims mirror the generator (r = Hv/Hk = 2 GQA, single token).
    let (h, hk, hv, dk, dv) = (8i32, 2i32, 4i32, 4i32, 4i32);
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let kk = 4i32;
    let (b, s) = (1i32, 1i32);

    let mk = |k: &str, shape: &[i32]| Array::from_slice(&arr(k), shape);
    let proj = |k: &str, shape: &[i32]| Projection::load(mk(k, shape), None).unwrap();
    let layer = GatedDeltaNet {
        precision: PrecisionProfile::Native,
        in_proj_qkv: proj("in_proj_qkv", &[conv_dim, h]),
        in_proj_z: proj("in_proj_z", &[value_dim, h]),
        in_proj_a: proj("in_proj_a", &[hv, h]),
        in_proj_b: proj("in_proj_b", &[hv, h]),
        conv_weight: mk("conv_weight", &[conv_dim, kk]),
        a_log: mk("A_log", &[hv]),
        dt_bias: mk("dt_bias", &[hv]),
        norm_weight: mk("norm_weight", &[dv]),
        out_proj: proj("out_proj", &[h, value_dim]),
        num_k_heads: hk,
        num_v_heads: hv,
        head_k_dim: dk,
        head_v_dim: dv,
        key_dim,
        value_dim,
        conv_dim,
        conv_kernel: kk,
        eps: 1e-6,
    };

    let x = mk("x", &[b, s, h]);
    let mut cache = DeltaNetCache::new();
    let out = layer.forward(&x, &mut cache).unwrap();
    assert_eq!(out.shape(), &[b, s, h]);

    let got = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let exp = arr("expected_output");
    let md = got
        .iter()
        .zip(&exp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < 2e-4,
        "deltanet layer vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}"
    );

    // The cache advanced and holds both the recurrent and conv state for a follow-on decode step.
    assert_eq!(cache.offset(), s);
    assert!(cache.conv_state.is_some() && cache.ssm_state.is_some());
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::deltanet_layer_matches_qwen3_5_reference");
}

/// A model-level invariant the hybrid cache must satisfy: prefilling a sequence in one pass must
/// produce the same final-token logits as feeding the tokens one at a time carrying the cache
/// (conv tail + recurrent SSM state for linear layers, growing KV for full-attention). Run on the
/// production bf16 path (MLX-vs-MLX, so no f32-GEMM-floor concern) — this is what guarantees the
/// short conv and delta recurrence resume correctly at decode time on real weights.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn prefill_equals_stepwise_decode() {
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let w = synthetic_weights(&cfg);
    let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
    let toks = [1i32, 7, 3, 42, 9, 2];

    // One-shot prefill of the whole sequence; keep the last-token logits.
    let mut c_pre = model.new_cache();
    let prefill = model
        .decode_logits(
            &Array::from_slice(&toks, &[1, toks.len() as i32]),
            &mut c_pre,
            0,
        )
        .unwrap();

    // Token-by-token with cache carry; keep the final step's logits.
    let mut c_step = model.new_cache();
    let mut last = None;
    for (i, &tok) in toks.iter().enumerate() {
        last = Some(
            model
                .decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32)
                .unwrap(),
        );
    }
    let step = last.unwrap();

    assert_eq!(c_pre.offset(), toks.len() as i32);
    assert_eq!(c_step.offset(), toks.len() as i32);
    let a = prefill
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let b = step
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let md = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // Both bf16 paths; allow small per-op bf16 reorder noise, far below any structural error.
    assert!(
        md < 5e-2,
        "prefill vs stepwise last-token logits diverged: max abs diff {md}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::prefill_equals_stepwise_decode"
    );
}

/// AC (sc-24443): the Gated DeltaNet dispatch (fused Metal kernel for prefill, decode and
/// verify) leaves Qwen35 greedy decoding unchanged against the op-by-op reference recurrence —
/// for a short and a multi-chunk prompt, a speculative-verify-width forward, and the decode
/// steps after it — on the production bf16 path, with every logit within one bf16 ULP of the
/// reference's. Runs on a GPU stream (the kernel's route) whatever the process default device.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn greedy_tokens_match_the_ops_reference_recurrence() {
    mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), greedy_tokens_match_on_the_gpu);
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::greedy_tokens_match_the_ops_reference_recurrence");
}

fn greedy_tokens_match_on_the_gpu() {
    use crate::primitives::gated_delta::{recording_routes, with_ops_reference, Route};
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let model =
        Qwen35Model::from_weights(&synthetic_weights(&cfg), "model.language_model", cfg).unwrap();
    let argmax_rows = |logits: &Array| -> Vec<i32> {
        let rows = logits.as_dtype(Dtype::Float32).unwrap();
        let v = *rows.shape().last().unwrap() as usize;
        host(&rows)
            .chunks(v)
            .map(|r| {
                r.iter()
                    .enumerate()
                    .fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m })
                    .0 as i32
            })
            .collect()
    };
    // Prefill, a 3-token verify block, then 6 greedy decode steps; every row's argmax.
    let run = |prompt: &[i32]| -> (Vec<i32>, Vec<f32>) {
        let mut cache = model.new_cache();
        let n = prompt.len() as i32;
        let mut out = Vec::new();
        let mut logits = Vec::new();
        let pre = model
            .decode_logits(&Array::from_slice(prompt, &[1, n]), &mut cache, 0)
            .unwrap();
        let mut next = argmax_rows(&pre)[0];
        logits.extend(host(&pre.as_dtype(Dtype::Float32).unwrap()));
        let block = [next, (next + 1) % 50, (next + 2) % 50];
        let verify = model
            .forward(&Array::from_slice(&block, &[1, 3]), &mut cache, n)
            .unwrap();
        let rows = argmax_rows(&verify);
        out.extend(&rows);
        logits.extend(host(&verify.as_dtype(Dtype::Float32).unwrap()));
        next = rows[2];
        for i in 0..6 {
            let step = model
                .decode_logits(&Array::from_slice(&[next], &[1, 1]), &mut cache, n + 3 + i)
                .unwrap();
            next = argmax_rows(&step)[0];
            out.push(next);
            logits.extend(host(&step.as_dtype(Dtype::Float32).unwrap()));
        }
        (out, logits)
    };
    let long: Vec<i32> = (0..90).map(|i| (i * 7 + 3) % 50).collect();
    for prompt in [&[1i32, 7, 3, 42, 9, 2][..], &long[..]] {
        let ((tokens, logits), routes) = recording_routes(|| run(prompt));
        let ((ref_tokens, ref_logits), ref_routes) =
            recording_routes(|| with_ops_reference(|| run(prompt)));
        // The production run took the fused kernel for every recurrence call and the reference
        // run the op loop — so the comparison is kernel against ops, not a path against itself.
        assert!(
            !routes.is_empty() && routes.iter().all(|r| *r == Route::Kernel),
            "production routes {routes:?}"
        );
        assert!(
            !ref_routes.is_empty() && ref_routes.iter().all(|r| *r == Route::Ops),
            "reference routes {ref_routes:?}"
        );
        let md = logits
            .iter()
            .zip(&ref_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // The logits are bf16: one ULP at the largest reference logit's binade (8 significand
        // bits, so the spacing in [2^e, 2^(e+1)) is 2^(e-7)) bounds a last-bit rounding flip.
        let scale = ref_logits.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let ulp = 2f32.powi(scale.log2().floor() as i32 - 7);
        eprintln!(
            "prompt len {}: max logit diff vs ops reference {md:.2e} (bf16 ulp {ulp:.2e} at \
                 |logit| {scale:.2e})",
            prompt.len()
        );
        assert_eq!(
            tokens,
            ref_tokens,
            "prompt len {}: greedy tokens diverged (max logit diff {md})",
            prompt.len()
        );
        assert!(
            md <= ulp,
            "prompt len {}: max logit diff {md} exceeds one bf16 ulp {ulp}",
            prompt.len()
        );
    }
    crate::model_inference_device::record_model_inference_device();
    crate::model_inference_device::record_model_inference_device();
}

/// A MoE config (`qwen3_5_moe`, the 35B-A3B shape, scaled down): 6 experts, top-2, with a shared
/// expert. Same 4-layer 3:1 mixer schedule as [`cfg_json`].
pub(crate) fn cfg_json_moe() -> serde_json::Value {
    let mut v = cfg_json();
    let tc = v["text_config"].as_object_mut().unwrap();
    tc.insert("model_type".into(), json!("qwen3_5_moe_text"));
    tc.insert("num_experts".into(), json!(6));
    tc.insert("num_experts_per_tok".into(), json!(2));
    tc.insert("moe_intermediate_size".into(), json!(16));
    tc.insert("shared_expert_intermediate_size".into(), json!(16));
    v
}

/// [`cfg_json_moe`] with a configured MTP head — the 35B-A3B checkpoint's layout (sc-24438).
pub(crate) fn cfg_json_moe_mtp() -> serde_json::Value {
    let mut v = cfg_json_moe();
    let tc = v["text_config"].as_object_mut().unwrap();
    tc.insert("mtp_num_hidden_layers".into(), json!(1));
    tc.insert("mtp_use_dedicated_embeddings".into(), json!(false));
    v
}

/// Synthetic tensors for a sparse-MoE `cfg` with an MTP head whose predictor layer carries a
/// sparse-MoE FFN (router, experts, shared expert) instead of the dense MLP — as the 35B-A3B
/// checkpoint ships it (sc-24438) — with every MoE tensor (body and head) seeded **non-zero
/// random**. `fused` stores the experts as the Qwen3.6 release does (`experts.gate_up_proj` /
/// `experts.down_proj`); otherwise each expert under its own keys, as the bf16 Qwen3.5 release
/// does (`experts.{e}.{gate,up,down}_proj.weight`) — the same values either way.
pub(crate) fn seeded_moe_mtp_tensors(cfg: &Qwen35Config, fused: bool) -> HashMap<String, Array> {
    use crate::primitives::sampler::{SplitMix64, TokenRng};
    let moe = cfg.moe.as_ref().expect("a MoE config");
    let h = cfg.hidden_size;
    let (e, mi, si) = (
        moe.num_experts,
        moe.moe_intermediate_size,
        moe.shared_expert_intermediate_size,
    );
    let mut m = synthetic_tensors_with_prefix(cfg, "model.language_model");
    for dense in ["gate_proj", "up_proj", "down_proj"] {
        assert!(m
            .remove(&format!("mtp.layers.0.mlp.{dense}.weight"))
            .is_some());
    }
    let mut rng = SplitMix64::new(0x5EED_24438);
    let mut layers: Vec<String> = (0..cfg.num_layers)
        .map(|i| format!("model.language_model.layers.{i}."))
        .collect();
    layers.push("mtp.layers.0.".into());
    for lp in layers {
        let mut rand = |key: &str, shape: &[i32]| {
            let n: i32 = shape.iter().product();
            let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.8).collect();
            m.insert(format!("{lp}{key}"), Array::from_slice(&data, shape));
            data
        };
        let gate_up = rand("mlp.experts.gate_up_proj", &[e, 2 * mi, h]);
        let down = rand("mlp.experts.down_proj", &[e, h, mi]);
        rand("mlp.gate.weight", &[e, h]);
        rand("mlp.shared_expert.gate_proj.weight", &[si, h]);
        rand("mlp.shared_expert.up_proj.weight", &[si, h]);
        rand("mlp.shared_expert.down_proj.weight", &[h, si]);
        rand("mlp.shared_expert_gate.weight", &[1, h]);
        if !fused {
            m.remove(&format!("{lp}mlp.experts.gate_up_proj"));
            m.remove(&format!("{lp}mlp.experts.down_proj"));
            let (rows, bank) = ((mi * h) as usize, (2 * mi * h) as usize);
            for x in 0..e as usize {
                let key = |p: &str| format!("{lp}mlp.experts.{x}.{p}_proj.weight");
                let gu = &gate_up[x * bank..(x + 1) * bank];
                m.insert(key("gate"), Array::from_slice(&gu[..rows], &[mi, h]));
                m.insert(key("up"), Array::from_slice(&gu[rows..], &[mi, h]));
                let dn = &down[x * rows..(x + 1) * rows];
                m.insert(key("down"), Array::from_slice(dn, &[h, mi]));
            }
        }
    }
    m
}

fn load_moe_mtp(cfg: &Qwen35Config, tensors: HashMap<String, Array>) -> Qwen35Model {
    Qwen35Model::from_weights(
        &Weights::from_map(tensors),
        "model.language_model",
        cfg.clone(),
    )
    .expect("a MoE checkpoint with an MTP head loads")
}

fn host_f32(a: &Array) -> Vec<f32> {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

/// sc-24438 AC2 (a): a sparse-MoE checkpoint carrying an MTP head loads the head — its
/// predictor layer the body's sparse-MoE block — and the fused (Qwen3.6) and per-expert
/// (Qwen3.5) expert layouts of the same seeded non-zero weights give identical target logits
/// and identical `mtp_step` logits / hidden state.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_moe_mtp_head_runs_identically_in_both_expert_layouts() {
    let cfg = Qwen35Config::from_json(&cfg_json_moe_mtp()).unwrap();
    let fused = load_moe_mtp(&cfg, seeded_moe_mtp_tensors(&cfg, true));
    let split = load_moe_mtp(&cfg, seeded_moe_mtp_tensors(&cfg, false));
    for model in [&fused, &split] {
        assert!(model.has_mtp());
        let Ffn::Moe(_) = &model.mtp.as_ref().unwrap().layers[0].ffn else {
            panic!("the predictor layer is the sparse-MoE block");
        };
    }
    let ids = Array::from_slice(&[1i32, 7, 3, 42], &[1, 4]);
    let run = |model: &Qwen35Model| {
        let hidden = model.hidden(&ids, &mut model.new_cache(), 0).unwrap();
        let mut cache = model.new_mtp_cache().unwrap();
        let shifted = Array::from_slice(&[7i32, 3, 42, 9], &[1, 4]);
        let (mtp_hidden, mtp_logits) = model.mtp_step(&shifted, &hidden, &mut cache, 0).unwrap();
        (
            host_f32(&hidden),
            host_f32(&mtp_hidden),
            host_f32(&mtp_logits),
        )
    };
    let (a, b) = (run(&fused), run(&split));
    assert!(
        a.2.iter().any(|x| x.abs() > 1e-3),
        "non-degenerate MTP logits"
    );
    assert_eq!(a.0, b.0, "target hidden");
    assert_eq!(a.1, b.1, "MTP hidden");
    assert_eq!(a.2, b.2, "MTP logits");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::a_moe_mtp_head_runs_identically_in_both_expert_layouts");
}

/// sc-24438 AC2 (b): the MTP predictor layer's FFN, built by the loader under the
/// `mtp.layers.0.` prefix from the `Qwen3_5MoeSparseMoeBlock.forward` oracle's weights (the
/// fused layout and the per-expert split of it), reproduces the oracle's output — the check
/// [`moe_ffn_matches_qwen3_5_moe_reference`] holds the block itself to, at bf16 tolerance.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_mtp_layer_ffn_matches_the_moe_reference() {
    let json: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/qwen35_moe_oracle.json")).unwrap();
    let arr = |k: &str| -> Vec<f32> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let (h, e, mi) = (8i32, 6i32, 4i32);
    let mut v = cfg_json_moe_mtp();
    let tc = v["text_config"].as_object_mut().unwrap();
    tc.insert("hidden_size".into(), json!(h));
    tc.insert("moe_intermediate_size".into(), json!(mi));
    tc.insert("shared_expert_intermediate_size".into(), json!(mi));
    let cfg = Qwen35Config::from_json(&v).unwrap();
    assert_eq!(
        (
            cfg.moe.unwrap().num_experts,
            cfg.moe.unwrap().experts_per_tok
        ),
        (e, 2)
    );
    for fused in [true, false] {
        let mut m = seeded_moe_mtp_tensors(&cfg, fused);
        let lp = |s: &str| format!("mtp.layers.0.mlp.{s}");
        let mut put = |key: String, data: &[f32], shape: &[i32]| {
            m.insert(key, Array::from_slice(data, shape));
        };
        let (gate_up, down) = (arr("gate_up"), arr("down"));
        if fused {
            put(lp("experts.gate_up_proj"), &gate_up, &[e, 2 * mi, h]);
            put(lp("experts.down_proj"), &down, &[e, h, mi]);
        } else {
            let rows = (mi * h) as usize;
            for x in 0..e as usize {
                let gu = &gate_up[x * 2 * rows..(x + 1) * 2 * rows];
                let key = |p: &str| lp(&format!("experts.{x}.{p}_proj.weight"));
                put(key("gate"), &gu[..rows], &[mi, h]);
                put(key("up"), &gu[rows..], &[mi, h]);
                put(key("down"), &down[x * rows..(x + 1) * rows], &[h, mi]);
            }
        }
        put(lp("gate.weight"), &arr("router"), &[e, h]);
        put(
            lp("shared_expert.gate_proj.weight"),
            &arr("sh_gate"),
            &[mi, h],
        );
        put(lp("shared_expert.up_proj.weight"), &arr("sh_up"), &[mi, h]);
        put(
            lp("shared_expert.down_proj.weight"),
            &arr("sh_down"),
            &[h, mi],
        );
        put(lp("shared_expert_gate.weight"), &arr("sh_gatew"), &[1, h]);
        let model = load_moe_mtp(&cfg, m);
        let ffn = &model.mtp.as_ref().unwrap().layers[0].ffn;
        let got = host_f32(
            &ffn.forward(
                &Array::from_slice(&arr("x"), &[1, 1, h]),
                PrecisionProfile::Native,
            )
            .unwrap(),
        );
        let exp = arr("expected_output");
        let md = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // The loader stores weights in bf16 (the block test above runs f32): bf16's 2^-8
        // relative step on outputs of magnitude ~0.5, far below a wrong route or expert.
        assert!(
            md < 5e-3,
            "fused {fused}: MTP-layer ffn vs reference: max abs diff {md}\n got {got:?}\n \
                 exp {exp:?}"
        );
    }
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::the_mtp_layer_ffn_matches_the_moe_reference");
}

/// The MoE FFN block, validated against a numeric oracle from the exact
/// `Qwen3_5MoeSparseMoeBlock.forward` reference: softmax router → top-k → renormalize → per-expert
/// SwiGLU → sigmoid-gated shared expert. Builds the shared [`SparseMoe`] block via the same split
/// as the loader (`experts.gate_up_proj` → stacked gate / up banks). Single token (S=1) so
/// MLX runs the exact GEMV path and the match is tight; regenerate with `/tmp/gen_moe.py`.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moe_ffn_matches_qwen3_5_moe_reference() {
    let json: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/qwen35_moe_oracle.json")).unwrap();
    let arr = |k: &str| -> Vec<f32> {
        json[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let (h, e, k, mi) = (8i32, 6i32, 2usize, 4i32);
    let mk = |key: &str, shape: &[i32]| Array::from_slice(&arr(key), shape);
    let proj = |a: Array| Projection::load(a, None).unwrap();

    // Split experts.gate_up_proj into the stacked gate / up banks (mirrors the loader).
    let gate_up = mk("gate_up", &[e, 2 * mi, h])
        .reshape(&[e, 2, mi, h])
        .unwrap();
    let half = |i: i32| {
        gate_up
            .take_axis(Array::from_slice(&[i], &[1]), 1)
            .unwrap()
            .reshape(&[e, mi, h])
            .unwrap()
    };
    let bank = |a: Array| SwitchLinear::load(a, None).unwrap();
    let moe = SparseMoe::new(
        mk("router", &[e, h]),
        bank(half(0)),
        bank(half(1)),
        bank(mk("down", &[e, h, mi])),
        SwiGlu {
            gate: proj(mk("sh_gate", &[mi, h])),
            up: proj(mk("sh_up", &[mi, h])),
            down: proj(mk("sh_down", &[h, mi])),
        },
        Some(mk("sh_gatew", &[1, h])),
        MoeRouting {
            experts_per_tok: k,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
        },
    )
    .unwrap();

    let out = moe.forward(&mk("x", &[1, 1, h])).unwrap();
    assert_eq!(out.shape(), &[1, 1, h]);
    let got = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let exp = arr("expected_output");
    let md = got
        .iter()
        .zip(&exp)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < 2e-4,
        "moe ffn vs reference: max abs diff {md}\n got {got:?}\n exp {exp:?}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("\nMODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::moe_ffn_matches_qwen3_5_moe_reference");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn moe_model_forward_and_prefill_equals_stepwise() {
    let cfg = Qwen35Config::from_json(&cfg_json_moe()).unwrap();
    assert!(cfg.moe.is_some());
    assert_eq!(cfg.moe.unwrap().num_experts, 6);
    let model = Qwen35Model::from_weights(
        &synthetic_weights(&cfg),
        "model.language_model",
        cfg.clone(),
    )
    .unwrap();

    // Multi-token prefill exercises routing/scatter across tokens; logits are finite + shaped.
    let toks = [1i32, 7, 3, 42, 9];
    let mut c_pre = model.new_cache();
    let logits = model
        .forward(
            &Array::from_slice(&toks, &[1, toks.len() as i32]),
            &mut c_pre,
            0,
        )
        .unwrap();
    assert_eq!(logits.shape(), &[1, toks.len() as i32, cfg.vocab_size]);
    for x in logits.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>() {
        assert!(x.is_finite(), "non-finite MoE logit");
    }

    // Prefill == stepwise decode over the hybrid cache, with the MoE FFN in the loop.
    let pre_last = model
        .decode_logits(
            &Array::from_slice(&toks, &[1, toks.len() as i32]),
            &mut model.new_cache(),
            0,
        )
        .unwrap();
    let mut c_step = model.new_cache();
    let mut last = None;
    for (i, &tok) in toks.iter().enumerate() {
        last = Some(
            model
                .decode_logits(&Array::from_slice(&[tok], &[1, 1]), &mut c_step, i as i32)
                .unwrap(),
        );
    }
    let a = pre_last
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let b = last
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let md = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md < 5e-2,
        "MoE prefill vs stepwise diverged: max abs diff {md}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::moe_model_forward_and_prefill_equals_stepwise");
}

/// sc-24442 AC2: the Qwen35 hybrid decoder (gated full attention with partial M-RoPE
/// interleaved with linear-attention layers) emits the same greedy tokens under the new `sdpa` routing as under
/// the pre-sc-24442 one, compared in-process — head dims 8 (unserved), 64 (full kernel) and 256
/// (vector-only) × GQA 2 / GQA 8 × prompts inside one tile, across tiles, and across many.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn greedy_tokens_match_pre_sc24442_sdpa_routing() {
    use crate::primitives::attention::route_override::{
        assert_greedy_matches_pre_sc24442, GreedyComparison,
    };
    use crate::primitives::sampler::{SplitMix64, TokenRng};

    let mut seen = GreedyComparison::default();
    for hd in [8, 64, 256] {
        for (nh, nkv) in [(4, 2), (8, 1)] {
            let mut v = cfg_json();
            let tc = v["text_config"].as_object_mut().unwrap();
            tc.insert("head_dim".into(), json!(hd));
            tc.insert("num_attention_heads".into(), json!(nh));
            tc.insert("num_key_value_heads".into(), json!(nkv));
            // Full attention at layers 1 and 3: an earlier full-attention layer's every row
            // feeds the last-position logits (the last layer's only reaches them via its
            // final row, which no masking or tiling choice changes).
            tc.insert("full_attention_interval".into(), json!(2));
            let cfg = Qwen35Config::from_json(&v).unwrap();
            assert!(!cfg.is_linear(1) && !cfg.is_linear(3));
            // Random (not periodic) values so greedy steps are rarely near-ties. Drawn in key
            // order: a `HashMap`'s iteration order is randomized per process, so drawing in it
            // gave every test process different weights — and the occasional draw whose
            // step-0 drift crosses the bound (the sc-24439 "flake").
            let mut rng = SplitMix64::new(0x2444_2350 + (hd * 16 + nh + nkv) as u64);
            let mut entries: Vec<(String, Array)> =
                synthetic_weights(&cfg).into_map().into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let weights: HashMap<String, Array> = entries
                .into_iter()
                .map(|(key, a)| {
                    let n = a.size();
                    let data: Vec<f32> = (0..n).map(|_| (rng.next_f32() - 0.5) * 0.4).collect();
                    (key, Array::from_slice(&data, a.shape()))
                })
                .collect();
            let model = Qwen35Model::from_weights(
                &Weights::from_map(weights),
                "model.language_model",
                cfg.clone(),
            )
            .unwrap();
            for prompt_len in [5, 20, 70] {
                let prompt: Vec<i32> = (0..prompt_len)
                    .map(|i| (i * 7 + 3) % cfg.vocab_size)
                    .collect();
                seen += assert_greedy_matches_pre_sc24442(
                    &format!("qwen35 hd {hd} {nh}/{nkv} prompt {prompt_len}"),
                    &prompt,
                    6,
                    || model.new_cache(),
                    |ids, cache, offset| model.decode_logits(ids, cache, offset).unwrap(),
                );
            }
        }
    }
    assert!(
        seen.differing_calls > 0,
        "no fixture exercised a routing change"
    );
    assert!(
        seen.compared_steps > seen.tie_steps,
        "most greedy steps must be decisive enough to compare: {seen:?}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::greedy_tokens_match_pre_sc24442_sdpa_routing");
}

fn synthetic_model() -> (Qwen35Config, Qwen35Model) {
    let cfg = Qwen35Config::from_json(&cfg_json()).unwrap();
    let w = synthetic_weights(&cfg);
    let model = Qwen35Model::from_weights(&w, "model.language_model", cfg.clone()).unwrap();
    (cfg, model)
}

/// `mrope_positions` (the `get_rope_index` port) must reproduce the reference 3-D position rows +
/// `mrope_delta` for an image+text sequence — exact integer index math (oracle /tmp/gen_mrope.py).
#[test]
fn mrope_positions_matches_reference() {
    let j: serde_json::Value =
        serde_json::from_str(include_str!("../testdata/qwen35_mrope_oracle.json")).unwrap();
    let r = &j["rope_index"];
    let ints = |k: &str| -> Vec<i32> {
        r[k].as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect()
    };
    let ids = ints("input_ids");
    let grid = {
        let g = &r["image_grid_thw"][0];
        vec![[
            g[0].as_i64().unwrap() as i32,
            g[1].as_i64().unwrap() as i32,
            g[2].as_i64().unwrap() as i32,
        ]]
    };
    let img_tok = r["image_token_id"].as_i64().unwrap() as i32;
    let merge = r["merge"].as_i64().unwrap() as i32;

    let (_cfg, model) = synthetic_model();
    let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img_tok, merge).unwrap();
    assert_eq!(t, ints("t"));
    assert_eq!(h, ints("h"));
    assert_eq!(w, ints("w"));
    assert_eq!(delta, r["delta"].as_i64().unwrap() as i32);
}

// --- Qwen3-VL HF-backed oracles (tools/gen_qwen3vl_mrope_oracle.py) ----------------------------

fn qwen3vl_mrope_oracle() -> serde_json::Value {
    serde_json::from_str(include_str!("../testdata/qwen3vl_mrope_oracle.json")).unwrap()
}

fn ints_of(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as i32)
        .collect()
}

fn grids_of(v: &serde_json::Value) -> Vec<[i32; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|g| {
            [
                g[0].as_i64().unwrap() as i32,
                g[1].as_i64().unwrap() as i32,
                g[2].as_i64().unwrap() as i32,
            ]
        })
        .collect()
}

/// **Interleaved-MRoPE position ids — mixed text/image.** `mrope_positions_mm` must reproduce the
/// real `Qwen3VLModel.get_rope_index` 3-D rows + `mrope_delta` for a text+image sequence, exact
/// integer match. This is the Qwen3-VL placement (image block offset by the running cursor, cursor
/// advanced by `max(t, h/merge, w/merge)`).
#[test]
fn qwen3vl_mrope_image_matches_hf_reference() {
    let j = qwen3vl_mrope_oracle();
    let r = &j["rope_index_image"];
    let ids = ints_of(&r["input_ids"]);
    let grid = grids_of(&r["image_grid_thw"]);
    let img = j["image_token_id"].as_i64().unwrap() as i32;
    let merge = j["merge"].as_i64().unwrap() as i32;

    let (_cfg, model) = synthetic_model();
    let (t, h, w, delta) = model.mrope_positions(&ids, &grid, img, merge).unwrap();
    assert_eq!(t, ints_of(&r["t"]), "image t-row vs HF get_rope_index");
    assert_eq!(h, ints_of(&r["h"]), "image h-row vs HF get_rope_index");
    assert_eq!(w, ints_of(&r["w"]), "image w-row vs HF get_rope_index");
    assert_eq!(
        delta,
        r["delta"].as_i64().unwrap() as i32,
        "image mrope_delta"
    );
}

/// **Interleaved-MRoPE position ids — synthetic time / multi-frame video axis.** Qwen3-VL splits a
/// `[t, h, w]` video into `t` per-frame `gt = 1` blocks (timestamps separate frames), so the
/// temporal index resets per frame and frames are ordered only by the advancing cursor.
/// `mrope_positions_mm` must reproduce the HF rows + delta exactly for the 2-frame case — the
/// Qwen3-VL-specific delta a single multi-`t` block would get wrong.
#[test]
fn qwen3vl_mrope_video_matches_hf_reference() {
    let j = qwen3vl_mrope_oracle();
    let r = &j["rope_index_video"];
    let ids = ints_of(&r["input_ids"]);
    let vgrid = grids_of(&r["video_grid_thw"]);
    let vid = j["video_token_id"].as_i64().unwrap() as i32;
    let img = j["image_token_id"].as_i64().unwrap() as i32;
    let merge = j["merge"].as_i64().unwrap() as i32;

    let (_cfg, model) = synthetic_model();
    let (t, h, w, delta) = model
        .mrope_positions_mm(&ids, &[], img, &vgrid, vid, merge)
        .unwrap();
    assert_eq!(
        t,
        ints_of(&r["t"]),
        "video t-row vs HF get_rope_index (per-frame reset)"
    );
    assert_eq!(h, ints_of(&r["h"]), "video h-row vs HF get_rope_index");
    assert_eq!(w, ints_of(&r["w"]), "video w-row vs HF get_rope_index");
    assert_eq!(
        delta,
        r["delta"].as_i64().unwrap() as i32,
        "video mrope_delta"
    );
}

/// **Interleaved-MRoPE table — end to end at Qwen3-VL config.** Build the partial-rotary RoPE at
/// the real text config (head_dim 128, `rotary_dim` 128, theta 5e6, `mrope_section [24,20,20]`),
/// feed it the HF rope rows for the image sequence, and require the interleaved cos/sin tables to
/// match the reference `apply_interleaved_mrope` within 1e-5 (f32). Closes the loop from
/// `get_rope_index` rows through the Qwen3-VL interleaving.
#[test]
fn qwen3vl_interleaved_cos_sin_matches_hf_reference() {
    let j = qwen3vl_mrope_oracle();
    let il = &j["interleaved"];
    let head_dim = j["head_dim"].as_i64().unwrap() as i32;
    let theta = j["rope_theta"].as_f64().unwrap() as f32;
    let section: Vec<usize> = j["mrope_section"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap() as usize)
        .collect();
    let sections = [section[0], section[1], section[2]];
    let (t, h, w) = (ints_of(&il["t"]), ints_of(&il["h"]), ints_of(&il["w"]));
    let expect = |k: &str| -> Vec<f32> {
        il[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };

    // Qwen3-VL text rope is full-rotary (partial_rotary_factor 1.0 ⇒ rotary_dim == head_dim),
    // NeoX half-split — exactly Rope::partial(head_dim, theta, false).
    let rope = crate::primitives::rope::Rope::partial(head_dim, theta, false);
    let (cos, sin) = rope
        .mrope_interleaved_cos_sin([&t, &h, &w], sections, Dtype::Float32)
        .unwrap();
    assert_eq!(cos.shape(), &[1, t.len() as i32, head_dim]);
    let cmp = |got: &[f32], exp: &[f32]| {
        assert_eq!(got.len(), exp.len(), "table length");
        got.iter()
            .zip(exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let dc = cmp(cos.as_slice::<f32>(), &expect("cos"));
    let ds = cmp(sin.as_slice::<f32>(), &expect("sin"));
    assert!(
        dc < 1e-5,
        "interleaved cos vs HF reference: max abs diff {dc}"
    );
    assert!(
        ds < 1e-5,
        "interleaved sin vs HF reference: max abs diff {ds}"
    );
}

/// **Image-placeholder token expansion matches the HF processor.** Given the raw chat ids with a
/// single `<|image_pad|>` framed by `<|vision_start|>` / `<|vision_end|>`, expanding the
/// placeholder to `grid.prod() / merge²` copies must reproduce the exact id stream
/// `Qwen3VLProcessor` emits — same count, same vision framing, surrounding text untouched.
#[test]
fn qwen3vl_token_expansion_matches_hf_processor() {
    let j = qwen3vl_mrope_oracle();
    let ex = &j["expand"];
    let expanded_hf = ints_of(&ex["expanded_ids"]);
    let img = ex["image_token_id"].as_i64().unwrap() as i32;
    let vs = ex["vision_start_token_id"].as_i64().unwrap() as i32;
    let ve = ex["vision_end_token_id"].as_i64().unwrap() as i32;
    let g = ints_of(&ex["grid_thw"]); // single [t, h, w]
    let grid = [g[0], g[1], g[2]];
    let merge = ex["merge"].as_i64().unwrap() as i32;
    let expected_count = ex["expected_count"].as_i64().unwrap() as usize;

    // The count the vision tower / processor agree on.
    let count = vision_merged_token_count(grid, merge);
    assert_eq!(
        count, expected_count,
        "merged-token count formula vs HF processor"
    );

    // Reconstruct the *raw* (pre-expansion) chat ids: the single image placeholder framed by
    // vision_start/vision_end, with all surrounding (non-image) tokens preserved in order. The HF
    // expanded ids are that with the placeholder repeated `count` times — so collapsing the image
    // run back to one token recovers the raw stream.
    let mut raw = Vec::new();
    let mut i = 0usize;
    while i < expanded_hf.len() {
        if expanded_hf[i] == img {
            raw.push(img); // one placeholder
            while i < expanded_hf.len() && expanded_hf[i] == img {
                i += 1;
            }
        } else {
            raw.push(expanded_hf[i]);
            i += 1;
        }
    }

    let expanded = expand_vision_placeholders(&raw, img, &[count]).unwrap();
    assert_eq!(expanded, expanded_hf, "expanded ids vs HF processor");

    // The expansion is framed by exactly one vision_start … vision_end with `count` image tokens
    // between, and the count matches what the merger emits.
    let si = expanded.iter().position(|&x| x == vs).unwrap();
    let ei = expanded.iter().position(|&x| x == ve).unwrap();
    assert_eq!(
        ei - si - 1,
        count,
        "image tokens framed between vision_start/vision_end"
    );
    assert_eq!(
        expanded[si + 1..ei].iter().filter(|&&x| x == img).count(),
        count
    );
}

/// **The text-path invariant.** Feeding token embeds + equal (text) 3-D positions through
/// `decode_logits_from_embeds` must be **bit-identical** to the token-id `decode_logits` — the
/// interleaved M-RoPE collapses to 1D and the embeds path is the same compute. This is the gate
/// that the multimodal hook doesn't perturb the (verified) text decoder.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn decode_from_embeds_text_only_equals_decode_logits() {
    let (_cfg, model) = synthetic_model();
    let toks = [1i32, 7, 3, 42, 9, 2];
    let ids = Array::from_slice(&toks, &[1, toks.len() as i32]);

    let a = model
        .decode_logits(&ids, &mut model.new_cache(), 0)
        .unwrap();
    let embeds = model.embed_input_ids(&ids).unwrap();
    let pos: Vec<i32> = (0..toks.len() as i32).collect();
    let b = model
        .decode_logits_from_embeds(&embeds, [&pos, &pos, &pos], &mut model.new_cache())
        .unwrap();

    assert_eq!(a.shape(), b.shape());
    let (av, bv) = (
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec(),
        b.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec(),
    );
    let md = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        md == 0.0,
        "embeds text path must equal token-id path bit-for-bit; max abs diff {md}"
    );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::decode_from_embeds_text_only_equals_decode_logits");
}

/// The splice hook overwrites exactly the image-token rows (in order) with the feature rows and
/// leaves text rows untouched.
#[test]
fn splice_image_features_replaces_image_rows() {
    let (cfg, model) = synthetic_model();
    let hidden = cfg.hidden_size as usize;
    let ids = [7i32, 49, 49, 8, 9]; // two image tokens (id 49) at positions 1,2
                                    // embeds[1,5,hidden]: row r filled with value r.
    let mut e = Vec::new();
    for r in 0..5 {
        e.extend(std::iter::repeat_n(r as f32, hidden));
    }
    let embeds = Array::from_slice(&e, &[1, 5, hidden as i32])
        .as_dtype(compute())
        .unwrap();
    // feats[2,hidden]: row j filled with 100 + j.
    let mut f = Vec::new();
    for j in 0..2 {
        f.extend(std::iter::repeat_n(100.0f32 + j as f32, hidden));
    }
    let feats = Array::from_slice(&f, &[2, hidden as i32]);

    let out = model
        .splice_image_features(&embeds, &ids, &feats, 49)
        .unwrap();
    assert_eq!(out.shape(), &[1, 5, hidden as i32]);
    let v = out
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let row = |r: usize| v[r * hidden]; // first element of each row (whole row is constant)
    assert_eq!(
        [row(0), row(1), row(2), row(3), row(4)],
        [0.0, 100.0, 101.0, 3.0, 4.0]
    );
}

/// DeepStack fusion is actually wired through the decoder: feeding non-zero tapped features
/// through `decode_logits_from_embeds_with_deepstack` must (a) run end to end to finite logits
/// and (b) **differ** from the same prefill with the fusion disabled (empty taps) — proving the
/// tapped features are consumed in the decoder layers, not computed and dropped.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn deepstack_fused_path_consumes_features_and_differs() {
    let (cfg, model) = synthetic_model();
    let img = 49i32;
    let ids = [1i32, 2, img, img, img, img, 3, 4];
    let grid = vec![[1i32, 4, 4]];
    let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

    let embeds = model.embed_input_ids(&ids_arr).unwrap();
    let feats = Array::from_slice(
        &(0..4 * cfg.hidden_size)
            .map(|i| (i % 7) as f32 * 0.1 - 0.3)
            .collect::<Vec<_>>(),
        &[4, cfg.hidden_size],
    );
    let spliced = model
        .splice_image_features(&embeds, &ids, &feats, img)
        .unwrap();
    let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
    let visual_pos_mask: Vec<bool> = ids.iter().map(|&id| id == img).collect();

    let ds = |scale: f32| {
        Array::from_slice(
            &(0..4 * cfg.hidden_size)
                .map(|i| (i % 5) as f32 * scale + 0.05)
                .collect::<Vec<_>>(),
            &[4, cfg.hidden_size],
        )
    };
    let deepstack = [ds(0.2), ds(-0.15)];

    let fused = model
        .decode_logits_from_embeds_with_deepstack(
            &spliced,
            [&t, &h, &w],
            &mut model.new_cache(),
            &visual_pos_mask,
            &deepstack,
        )
        .unwrap();
    let unfused = model
        .decode_logits_from_embeds_with_deepstack(
            &spliced,
            [&t, &h, &w],
            &mut model.new_cache(),
            &visual_pos_mask,
            &[],
        )
        .unwrap();
    let baseline = model
        .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
        .unwrap();

    assert_eq!(fused.shape(), &[1, cfg.vocab_size]);
    let fv = fused
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let uv = unfused
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let bv = baseline
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(fv.iter().all(|x| x.is_finite()), "non-finite fused logit");

    let unfused_vs_baseline = uv
        .iter()
        .zip(&bv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        unfused_vs_baseline, 0.0,
        "empty-deepstack path must equal plain embeds path"
    );

    let fused_vs_unfused = fv
        .iter()
        .zip(&uv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
            fused_vs_unfused > 1e-3,
            "DeepStack fusion did not change the logits (features dropped?): max abs diff {fused_vs_unfused}"
        );

    // The provider drives this decoder through the `VlmDecode` seam, which boxes the cache via
    // `Decode::make_cache` and downcasts it back to `Qwen35Cache` inside `prefill_with_deepstack`.
    // That trait path must reproduce the inherent fused logits bit-for-bit (same model, inputs,
    // and a fresh cache).
    let mut trait_cache = crate::decode::Decode::make_cache(&model);
    let via_trait = crate::models::VlmDecode::prefill_with_deepstack(
        &model,
        &spliced,
        [&t, &h, &w],
        trait_cache.as_mut(),
        &visual_pos_mask,
        &deepstack,
    )
    .unwrap();
    let tv = via_trait
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let trait_vs_inherent = fv
        .iter()
        .zip(&tv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
            trait_vs_inherent, 0.0,
            "VlmDecode::prefill_with_deepstack must equal the inherent fused path (cache downcast seam)"
        );
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::deepstack_fused_path_consumes_features_and_differs");
}

#[test]
fn deepstack_seam_is_decoder_agnostic_at_qwen3vl_shapes() {
    let hidden = 4096i32;
    let num_layers = 6usize;
    let taps = [8usize, 16, 24];
    let seq = 8i32;
    let visual_pos_mask: Vec<bool> = (0..seq).map(|i| (2..6).contains(&i)).collect();
    let num_visual = visual_pos_mask.iter().filter(|&&m| m).count() as i32;
    assert_eq!(num_visual, 4);

    let v0 = 0.5f32;
    let h0 = Array::from_slice(&vec![v0; (seq * hidden) as usize], &[1, seq, hidden])
        .as_dtype(compute())
        .unwrap();
    let feat_val = [0.25f32, 0.5, 0.75];
    let deepstack: Vec<Array> = (0..taps.len())
        .map(|t| {
            Array::from_slice(
                &vec![feat_val[t]; (num_visual * hidden) as usize],
                &[num_visual, hidden],
            )
            .as_dtype(compute())
            .unwrap()
        })
        .collect();

    let two = Array::from_f32(2.0).as_dtype(compute()).unwrap();
    let mut calls: Vec<usize> = Vec::new();
    let fused =
        deepstack_fused_decoder_layers(&h0, &visual_pos_mask, &deepstack, num_layers, |i, h| {
            calls.push(i);
            Ok(multiply(h, &two)?)
        })
        .unwrap();
    assert_eq!(
        calls,
        (0..num_layers).collect::<Vec<_>>(),
        "every decoder layer must run once, in order"
    );

    let unfused =
        deepstack_fused_decoder_layers(&h0, &visual_pos_mask, &[], num_layers, |_i, h| {
            Ok(multiply(h, &two)?)
        })
        .unwrap();

    let scale = 2f32.powi(num_layers as i32);
    let text_expected = v0 * scale;
    let visual_expected: f32 = v0 * scale
        + (0..taps.len())
            .map(|t| feat_val[t] * 2f32.powi((num_layers - 1 - t) as i32))
            .sum::<f32>();

    let fv = fused
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let uv = unfused
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let row = |buf: &[f32], r: i32| buf[(r * hidden) as usize];
    for r in 0..seq {
        let got = row(&fv, r);
        let want = if visual_pos_mask[r as usize] {
            visual_expected
        } else {
            text_expected
        };
        assert!(
            (got - want).abs() < 1e-1,
            "fused row {r}: got {got}, want {want}"
        );
    }
    assert!((visual_expected - 54.0).abs() < 1e-3);
    assert!((text_expected - 32.0).abs() < 1e-3);

    for r in 0..seq {
        assert!(
            (row(&uv, r) - text_expected).abs() < 1e-1,
            "unfused row {r} must skip injection"
        );
    }
    let fused_vs_unfused = fv
        .iter()
        .zip(&uv)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        fused_vs_unfused > 1.0,
        "fused path must differ from non-fused: max abs diff {fused_vs_unfused}"
    );
}

/// Smoke: the full image+text path (embed → splice features → M-RoPE positions →
/// decode_logits_from_embeds) runs end to end and yields finite `[1, vocab]` logits.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn image_text_decode_from_embeds_runs() {
    let (cfg, model) = synthetic_model();
    let img = 49i32; // within the synthetic vocab (50) so embed gather is in-bounds
    let ids = [1i32, 2, img, img, img, img, 3, 4]; // 2x2 image (4 tokens) between text
    let grid = vec![[1i32, 4, 4]];
    let ids_arr = Array::from_slice(&ids, &[1, ids.len() as i32]);

    let embeds = model.embed_input_ids(&ids_arr).unwrap();
    let feats = Array::from_slice(
        &(0..4 * cfg.hidden_size)
            .map(|i| (i % 7) as f32 * 0.1 - 0.3)
            .collect::<Vec<_>>(),
        &[4, cfg.hidden_size],
    );
    let spliced = model
        .splice_image_features(&embeds, &ids, &feats, img)
        .unwrap();
    let (t, h, w, _delta) = model.mrope_positions(&ids, &grid, img, 2).unwrap();
    let logits = model
        .decode_logits_from_embeds(&spliced, [&t, &h, &w], &mut model.new_cache())
        .unwrap();
    assert_eq!(logits.shape(), &[1, cfg.vocab_size]);
    assert!(logits
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .all(|x| x.is_finite()));
    crate::model_inference_device::record_model_inference_device();
    println!(
        "MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::image_text_decode_from_embeds_runs"
    );
}

// ---- The DeltaNet checkpoint ring at the model level (sc-24435). ----

/// Feed `tokens` one forward each, returning the last forward's logits on the host.
fn step_each(model: &Qwen35Model, cache: &mut Qwen35Cache, tokens: &[i32]) -> Vec<f32> {
    let mut last = Vec::new();
    for &t in tokens {
        let offset = cache.offset();
        let logits = model
            .decode_logits(&crate::primitives::input_ids(&[t]), cache, offset)
            .unwrap();
        last = host(&logits);
    }
    last
}

fn prefilled(model: &Qwen35Model, prompt: &[i32]) -> Qwen35Cache {
    let mut cache = model.new_cache();
    let logits = model
        .decode_logits(&crate::primitives::input_ids(prompt), &mut cache, 0)
        .unwrap();
    logits.eval().unwrap();
    cache
}

const RING_PROMPT: [i32; 6] = [3, 9, 4, 11, 3, 9];

/// The draft-model shape of the ring seam: one checkpoint window opened at the step start
/// spans several single-token forwards (and a multi-token one), and the hybrid cache truncates
/// to **each** position of it — the DeltaNet layers restore their kept state, the attention KV
/// drops the rest by offset — after which the next forward's logits are exactly those of a
/// cache that never saw the dropped tokens.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_checkpoint_window_over_several_forwards_truncates_to_each_position() {
    let model = crate::decode::engine::tests::qwen35(false);
    let p = RING_PROMPT.len() as i32;
    let drafts = [4, 11, 7, 20];
    let next = 13;
    for j in 0..=drafts.len() {
        let mut cache = prefilled(&model, &RING_PROMPT);
        cache.arm_checkpoints(drafts.len() as i32);
        step_each(&model, &mut cache, &drafts);
        assert_eq!(cache.restorable(), p..p + drafts.len() as i32);
        cache.truncate(p + j as i32).unwrap();
        assert_eq!(cache.offset(), p + j as i32);
        let got = step_each(&model, &mut cache, &[next]);
        let mut reference = prefilled(&model, &RING_PROMPT);
        step_each(&model, &mut reference, &drafts[..j]);
        let want = step_each(&model, &mut reference, &[next]);
        assert_eq!(got, want, "truncated to {j} kept drafts");
    }

    // A multi-token forward inside the window (the verify shape) followed by a single token.
    let mut cache = prefilled(&model, &RING_PROMPT);
    cache.arm_checkpoints(4);
    let offset = cache.offset();
    model
        .decode_logits(
            &crate::primitives::input_ids(&drafts[..3]),
            &mut cache,
            offset,
        )
        .unwrap()
        .eval()
        .unwrap();
    step_each(&model, &mut cache, &drafts[3..]);
    assert_eq!(cache.restorable(), p..p + 4);
    cache.truncate(p + 2).unwrap();
    let got = step_each(&model, &mut cache, &[next]);
    // The reference never saw the third token.
    let mut reference = prefilled(&model, &RING_PROMPT);
    step_each(&model, &mut reference, &drafts[..2]);
    let want = step_each(&model, &mut reference, &[next]);
    let max = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    // The kept states came out of a 3-token forward, the reference's out of single-token
    // ones: equal up to the multi-row projection's bf16 rounding.
    assert!(max < 5e-2, "restored-from-verify logits differ by {max}");
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::a_checkpoint_window_over_several_forwards_truncates_to_each_position");
}

/// `truncate` is refused (typed, cache untouched) outside the checkpoint window — with no
/// window at all, and before the window's start — and a truncation closes the window: the
/// next forward records nothing and drops it. `0` is a reset and the current length a no-op,
/// through the [`KvCache`] trait too.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn truncate_outside_the_checkpoint_window_is_refused_and_closes_it() {
    let model = crate::decode::engine::tests::qwen35(false);
    let p = RING_PROMPT.len() as i32;
    let mut cache = prefilled(&model, &RING_PROMPT);
    step_each(&model, &mut cache, &[4, 11]);
    let err = cache.truncate(p + 1).unwrap_err();
    assert!(matches!(err, Error::Unsupported(_)), "no window: {err}");
    assert_eq!(cache.offset(), p + 2);

    cache.arm_checkpoints(2);
    step_each(&model, &mut cache, &[7, 20]);
    let err = cache.truncate(p + 1).unwrap_err();
    assert!(
        matches!(err, Error::Unsupported(_)),
        "before the window: {err}"
    );
    assert_eq!(
        cache.offset(),
        p + 4,
        "a refused truncate leaves the cache untouched"
    );
    assert!(cache.truncate(p + 5).is_err(), "past the end");

    KvCache::truncate(&mut cache, p + 3).unwrap();
    assert_eq!(cache.offset(), p + 3);
    step_each(&model, &mut cache, &[5]);
    assert!(
        cache.restorable().is_empty(),
        "the truncation closed the window"
    );
    assert!(cache.truncate(p + 3).is_err());

    let offset = cache.offset();
    KvCache::truncate(&mut cache, offset).unwrap();
    assert_eq!(cache.offset(), p + 4);
    KvCache::truncate(&mut cache, 0).unwrap();
    assert_eq!(cache.offset(), 0);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::truncate_outside_the_checkpoint_window_is_refused_and_closes_it");
}

/// sc-24435: a checkpoint window armed for `2` tokens refuses a 3-token forward (typed)
/// before any layer runs — the offset, every DeltaNet state and the attention KV are as they
/// were, so the next forward's logits are a never-refused cache's.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn a_forward_past_the_armed_window_is_refused_untouched() {
    let model = crate::decode::engine::tests::qwen35(false);
    let p = RING_PROMPT.len() as i32;
    let mut cache = prefilled(&model, &RING_PROMPT);
    cache.arm_checkpoints(2);
    let before = cache.delta_states();
    let err = model
        .decode_logits(&crate::primitives::input_ids(&[4, 11, 7]), &mut cache, p)
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::CheckpointWindowFull {
                recorded: 0,
                requested: 3,
                max_tokens: 2
            }
        ),
        "{err}"
    );
    assert_eq!(cache.offset(), p);
    assert_eq!(cache.delta_states(), before);
    assert_eq!(cache.checkpointed_tokens(), 0);
    let got = step_each(&model, &mut cache, &[4, 11]);
    let want = step_each(&model, &mut prefilled(&model, &RING_PROMPT), &[4, 11]);
    assert_eq!(got, want);
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::a_forward_past_the_armed_window_is_refused_untouched");
}

/// E7 (sc-24435): the checkpoint ring's resident memory — the window's start state plus every
/// kept per-token state and conv input, measured on the cache — never exceeds what
/// [`Qwen35Model::checkpoint_ring_bytes`] prices for the speculative width, for a verify step
/// (`width + 1` tokens in one forward, the cap the window is armed with) and for a draft
/// model's window (`width` single-token forwards). Keeping every position — a full acceptance,
/// `truncate(offset())` — closes the window, so the next forward records nothing and releases
/// the ring.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn the_checkpoint_ring_footprint_is_what_admission_prices() {
    let model = crate::decode::engine::tests::qwen35(false);
    for width in [1usize, 4, 8] {
        let priced = model.checkpoint_ring_bytes(width).unwrap() as usize;
        let cap = width as i32 + 1;
        let tokens: Vec<i32> = (0..cap).map(|i| 4 + i).collect();
        let released = |cache: &mut Qwen35Cache, live: usize, label: &str| {
            let offset = cache.offset();
            cache.truncate(offset).unwrap();
            step_each(&model, cache, &[5]);
            assert!(cache.restorable().is_empty(), "{label}: the window closed");
            assert_eq!(cache.checkpointed_tokens(), 0, "{label}: nothing recorded");
            assert_eq!(cache.recurrent_bytes(), live, "{label}: the ring released");
        };

        let mut cache = prefilled(&model, &RING_PROMPT);
        let live = cache.recurrent_bytes();
        cache.arm_checkpoints(cap);
        step_each(&model, &mut cache, &tokens[..width]);
        let ring = cache.recurrent_bytes() - live;
        assert!(
            ring > 0 && ring <= priced,
            "draft {width}: ring {ring} B, priced {priced} B"
        );
        released(&mut cache, live, &format!("draft {width}"));

        let mut cache = prefilled(&model, &RING_PROMPT);
        cache.arm_checkpoints(cap);
        let offset = cache.offset();
        model
            .decode_logits(&crate::primitives::input_ids(&tokens), &mut cache, offset)
            .unwrap()
            .eval()
            .unwrap();
        let ring = cache.recurrent_bytes() - live;
        assert!(
            ring > 0 && ring <= priced,
            "verify {width}: ring {ring} B, priced {priced} B"
        );
        released(&mut cache, live, &format!("verify {width}"));
    }
    assert!(model.checkpoint_ring_bytes(8) > model.checkpoint_ring_bytes(4));
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::the_checkpoint_ring_footprint_is_what_admission_prices");
}

/// sc-20671: stored quantized projections whose `scales`/`biases` are F16 (the mlx-community
/// convention) must not promote the BF16 activations to F32 — the hidden states and the full-
/// attention K/V stay in the compute dtype. The BF16-scale twin is the unchanged control.
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn stored_quantized_scales_keep_activations_and_kv_in_compute_dtype() {
    use crate::primitives::quant::QuantizedLinear;
    let mut value = cfg_json();
    // qwen3_5 stores 64-wide groups: widen every stored projection's input to a multiple.
    let text = &mut value["text_config"];
    text["hidden_size"] = json!(64);
    text["intermediate_size"] = json!(128);
    text["head_dim"] = json!(16);
    text["linear_value_head_dim"] = json!(16);
    value["quantization"] = json!({"group_size": 64, "bits": 4});
    let cfg = Qwen35Config::from_json(&value).unwrap();
    let dense = synthetic_weights(&cfg);
    const STORED: [&str; 10] = [
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    for scale_dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let mut map = HashMap::new();
        for key in dense.keys() {
            let array = dense.require(key).unwrap().as_dtype(scale_dtype).unwrap();
            let base = key.strip_suffix(".weight").unwrap_or(key);
            if key.starts_with("model.language_model.layers.")
                && STORED.iter().any(|p| base.ends_with(p))
            {
                let q = QuantizedLinear::quantize(&array, 64, 4, None).unwrap();
                assert_eq!(q.scales.dtype(), scale_dtype);
                map.insert(format!("{base}.weight"), q.weight);
                map.insert(format!("{base}.scales"), q.scales);
                map.insert(format!("{base}.biases"), q.biases);
            } else {
                map.insert(key.to_string(), array);
            }
        }
        let model =
            Qwen35Model::from_weights(&Weights::from_map(map), "model.language_model", cfg.clone())
                .unwrap();
        assert!(model.is_quantized());
        let mut cache = model.new_cache();
        let hidden = model
            .hidden(&Array::from_slice(&[1i32, 2, 3], &[1, 3]), &mut cache, 0)
            .unwrap();
        assert_eq!(hidden.dtype(), compute(), "{scale_dtype:?} hidden");
        let mut attention_layers = 0;
        for layer in &cache.layers {
            if let Qwen35LayerCache::Attn(slot) = layer {
                let (k, v) = slot.kv.peek(0).unwrap().unwrap();
                assert_eq!(k.dtype(), compute(), "{scale_dtype:?} keys");
                assert_eq!(v.dtype(), compute(), "{scale_dtype:?} values");
                attention_layers += 1;
            }
        }
        assert_eq!(attention_layers, 1);
    }
    crate::model_inference_device::record_model_inference_device();
    println!("MODEL_INFERENCE_BODY_COMPLETED models::qwen35::tests::stored_quantized_scales_keep_activations_and_kv_in_compute_dtype");
}

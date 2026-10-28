//! Independent frozen upstream data, never expected values calculated by the native implementation.
use super::*;
use candle_gen::gen_core::{
    moss_image::*, moss_image_edit::edit_request, GenerationRequest, Progress,
};
use serde_json::Value;
fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../contracts/gen-core/tests/fixtures/moss_image")
        .join(name);
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}
fn multi_fixture() -> Value {
    let mut f = fixture("tiny_edit_v2/fixture.json");
    let extra = fixture("tiny_edit_multi/fixture.json");
    assert_eq!(f["weights"], extra["weights"]);
    assert_eq!(f["adapter"], extra["adapter"]);
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    f
}
fn array(v: &Value, device: &Device) -> Tensor {
    let shape: Vec<usize> = serde_json::from_value(v["shape"].clone()).unwrap();
    let values: Vec<f32> = serde_json::from_value(v["values"].clone()).unwrap();
    Tensor::from_vec(values, shape, device)
        .unwrap()
        .to_dtype(if v["dtype"] == "bfloat16" {
            DType::BF16
        } else {
            DType::F32
        })
        .unwrap()
}
fn values(t: &Tensor) -> Vec<f32> {
    crate::loader::tensor_f32(t).unwrap()
}
fn delta(a: &Tensor, b: &Tensor) -> f32 {
    assert_eq!(a.dims(), b.dims(), "oracle shape");
    let (a, b) = (values(a), values(b));
    assert_eq!(a.len(), b.len(), "oracle count");
    assert!(a.iter().chain(&b).all(|x| x.is_finite()), "oracle finite");
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max)
}
fn compare(a: &Tensor, b: &Value, bound: f32, label: &str) {
    let error = delta(a, &array(b, a.device()));
    eprintln!("{label}: max_abs={error}, bound={bound}");
    assert!(error <= bound, "{label}: {error} > {bound}");
}
fn bf16_bound(v: &Value) -> f32 {
    v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap().abs() as f32)
        .fold(0., f32::max)
        / 64.
        + 1. / 1024.
}
fn compare_bf16(a: &Tensor, b: &Value, label: &str) {
    compare(a, b, bf16_bound(b), label);
}
fn model(f: &Value, device: &Device, fp32: bool) -> Transformer {
    let weights = Weights::from_map(
        f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let t = array(v, device);
                (
                    k.clone(),
                    if fp32 {
                        t.to_dtype(DType::F32).unwrap()
                    } else {
                        t
                    },
                )
            })
            .collect(),
    );
    let mut model = Transformer::load(
        &weights,
        Geometry {
            dim: 16,
            ffn: 24,
            heads: 2,
            layers: 2,
            text: 12,
            freq: 8,
            channels: 4,
        },
        None,
    )
    .unwrap();
    let adapters = Weights::from_map(
        f["adapter"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), array(v, device)))
            .collect(),
    );
    model.install_edit(&adapters, 2).unwrap();
    if fp32 {
        // Diagnostic graph only: the exact represented BF16 weights widened to F32.
        // Production install continues to reject non-BF16 and non-rank64 packages.
        for block in &mut model.layers {
            for p in [
                &mut block.q,
                &mut block.k,
                &mut block.v,
                &mut block.o,
                &mut block.up_gate,
                &mut block.down,
            ] {
                let a = p.low_rank.as_mut().unwrap();
                a.down = a.down.to_dtype(DType::F32).unwrap();
                a.up = a.up.to_dtype(DType::F32).unwrap();
            }
        }
    }
    model
}
fn masked(v: &Value, mask: &Value, device: &Device) -> Tensor {
    let indices: Vec<u32> = mask["values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (v.as_f64().unwrap() != 0.).then_some(i as u32))
        .collect();
    array(v, device)
        .index_select(
            &Tensor::from_vec(indices.clone(), indices.len(), device).unwrap(),
            1,
        )
        .unwrap()
}
fn resolved() -> MossImageResolved {
    MossImageResolved::resolve(
        MossImageVariant::Sft,
        &edit_request(&GenerationRequest {
            width: 48,
            height: 32,
            steps: Some(2),
            guidance: Some(4.),
            ..Default::default()
        })
        .unwrap(),
    )
    .unwrap()
}
fn time_probes(model: &Transformer, f: &Value, device: &Device) {
    for probe in f["time_probes"].as_array().unwrap() {
        for (label, actual, expected) in [
            (
                "time first linear",
                model
                    .time1
                    .forward(&array(&probe["time1_input"], device))
                    .unwrap(),
                &probe["time1"],
            ),
            (
                "time exact SiLU",
                silu(&array(&probe["silu_input"], device)).unwrap(),
                &probe["silu"],
            ),
            (
                "time second linear",
                model
                    .time2
                    .forward(&array(&probe["time2_input"], device))
                    .unwrap(),
                &probe["time2"],
            ),
            (
                "time modulation",
                model
                    .shared
                    .forward(&array(&probe["modulation_flat_input"], device))
                    .unwrap(),
                &probe["modulation_flat"],
            ),
            (
                "time head",
                model
                    .head
                    .forward(&array(&probe["head_flat_input"], device))
                    .unwrap(),
                &probe["head_flat"],
            ),
        ] {
            if matches!(label, "time first linear" | "time exact SiLU") {
                compare(&actual, expected, 0., label);
            } else {
                compare_bf16(&actual, expected, label);
            }
        }
        let input = array(&probe["time1_input"], device);
        let dot = candle_nn::Linear::new(
            model.time1.base.weight().to_dtype(DType::F32).unwrap(),
            None,
        )
        .forward(&input.to_dtype(DType::F32).unwrap())
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
        let split_bias = dot.broadcast_add(model.time1.base.bias().unwrap()).unwrap();
        let silu_input = array(&probe["silu_input"], device);
        let eager = silu_input
            .mul(&candle_nn::ops::sigmoid(&silu_input).unwrap())
            .unwrap();
        for (wrong, expected) in [(&split_bias, &probe["time1"]), (&eager, &probe["silu"])] {
            assert!(
                delta(wrong, &array(expected, device)) > 0.,
                "rounding control must distinguish source-fused arithmetic"
            );
        }
        let (m, h) = model
            .edit_time(probe["normalized_timestep"].as_f64().unwrap() as f32)
            .unwrap();
        compare_bf16(&m, &probe["modulation"], "complete time modulation");
        compare_bf16(&h, &probe["head"], "complete time head");
    }
}
fn sampler_precision(f: &Value, device: &Device, fp32: bool) {
    const F32_BOUND: f32 = 2e-5;
    let mut oracle = fixture("tiny_edit_v4_sampler_analysis/fp32_sampler.json");
    oracle["cases"].as_array_mut().unwrap().extend(
        fixture("tiny_edit_multi/fp32_sampler.json")["cases"]
            .as_array()
            .unwrap()
            .iter()
            .cloned(),
    );
    let input = &f["inputs"];
    let model = model(f, device, fp32);
    for case in f["cases"].as_array().unwrap() {
        let count = case["reference_count"].as_u64().unwrap() as usize;
        let refs: Vec<_> = input["references"].as_array().unwrap()[..count]
            .iter()
            .map(|v| array(v, device))
            .collect();
        let positive = [masked(&input["positive"], &input["positive_mask"], device)];
        let negative = [masked(&input["negative"], &input["negative_mask"], device)];
        let resolved = resolved();
        let sampling = crate::edit::EditSampling {
            transformer: &model,
            resolved: &resolved,
            positive: &positive,
            negative: &negative,
            references: &refs,
            use_cache: true,
            images_per_prompt: 1,
        };
        crate::edit::SAMPLING_TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
        let output = sampling
            .run(
                &array(&input["hidden_states"], device),
                &CancelFlag::new(),
                &mut |_| {},
            )
            .unwrap();
        let trace = crate::edit::SAMPLING_TRACE.with(|t| t.borrow_mut().take().unwrap());
        assert_eq!(trace.len(), 2);
        assert_eq!(output.dtype(), DType::F32);
        if fp32 {
            let expected = oracle["cases"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["reference_count"] == case["reference_count"])
                .unwrap();
            compare(
                &output,
                &expected["correct"]["final"],
                F32_BOUND,
                "strict F32 complete sampler",
            );
            for (actual, step) in trace
                .iter()
                .zip(expected["correct"]["steps"].as_array().unwrap())
            {
                for (a, key) in
                    actual
                        .iter()
                        .zip(["input", "conditional", "unconditional", "output"])
                {
                    assert_eq!(a.dtype(), DType::F32);
                    compare(a, &step[key], F32_BOUND, "strict F32 sampler stage");
                }
            }
            for (name, control) in expected["controls"].as_object().unwrap() {
                assert!(
                    delta(&output, &array(&control["final"], device)) > 100. * F32_BOUND,
                    "undiscriminated sampler control {name}"
                );
            }
        } else {
            let cache = model.prepare_edit_cache(&refs, &CancelFlag::new()).unwrap();
            for (index, source) in case["sampling"]["steps"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                let sigmas = &case["sampling"]["sigmas"]["values"];
                let time = sigmas[index].as_f64().unwrap() as f32;
                for (context, key) in [
                    (&positive[0], "conditional"),
                    (&negative[0], "unconditional"),
                ] {
                    let prediction = model
                        .forward_edit(
                            &array(&source["input"], device),
                            time,
                            EditConditioning {
                                context,
                                references: &refs,
                                cache: Some(&cache),
                            },
                            &CancelFlag::new(),
                        )
                        .unwrap();
                    assert_eq!(prediction.dtype(), DType::BF16);
                    compare_bf16(&prediction, &source[key], "BF16 prediction at source input");
                }
                let rows: Vec<_> = trace[index]
                    .iter()
                    .map(|a| {
                        assert_eq!(a.dtype(), DType::F32);
                        values(a)
                    })
                    .collect();
                let dt = sigmas[index + 1].as_f64().unwrap() as f32 - time;
                assert!(rows.iter().all(|row| row.len() == rows[0].len()));
                for (((current, conditional), unconditional), output) in
                    rows[0].iter().zip(&rows[1]).zip(&rows[2]).zip(&rows[3])
                {
                    let expected =
                        current + dt * (unconditional + 4. * (conditional - unconditional));
                    assert!(
                        (output - expected).abs() <= 1e-6,
                        "production F32 CFG/Euler composition"
                    );
                }
            }
            eprintln!(
                "BF16 trajectory drift (diagnostic only): {}",
                delta(&output, &array(&case["sampling"]["final"], device))
            );
        }
    }
}
// Successive request discriminators execute actual tiny Transformer/LoRA and sampler math.
// Fresh model + uncached sampling is an independent lifetime baseline, not a new golden.
fn request_local_lifecycle(f: &Value, device: &Device) {
    let mut changed_backbone = f.clone();
    let mut changed_adapter = f.clone();
    for value in changed_backbone["weights"]["final_linear_video.weight"]["values"]
        .as_array_mut()
        .unwrap()
    {
        *value = serde_json::json!(-value.as_f64().unwrap());
    }
    for (name, tensor) in changed_adapter["adapter"].as_object_mut().unwrap() {
        if name.ends_with("lora_B") {
            for value in tensor["values"].as_array_mut().unwrap() {
                *value = serde_json::json!(-value.as_f64().unwrap());
            }
        }
    }
    for fp32 in [false, true] {
        let definitions = [f, &changed_backbone, &changed_adapter];
        let models: Vec<_> = definitions.iter().map(|f| model(f, device, fp32)).collect();
        let input = &f["inputs"];
        let refs: Vec<_> = input["references"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| array(v, device))
            .collect();
        let hidden = array(&input["hidden_states"], device);
        let positive = masked(&input["positive"], &input["positive_mask"], device);
        let negative = masked(&input["negative"], &input["negative_mask"], device);
        let mut baseline = None;
        // Return to A after each changed count/order/content/duplicate/model/adapter/context,
        // geometry/noise/CFG request. Both BF16 production boundaries and represented F32.
        for case in [
            0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0,
        ] {
            let model_index = match case {
                6 => 1,
                7 => 2,
                _ => 0,
            };
            let mut references = refs.clone();
            match case {
                1 => references.truncate(1),
                2 => references.truncate(2),
                3 => references.truncate(3),
                4 => references.reverse(),
                5 => references[2] = references[0].clone(),
                8 => references[1] = (&references[1] * -1.).unwrap(),
                _ => (),
            }
            let noise = if case == 9 {
                hidden.narrow(3, 0, 2).unwrap()
            } else if case == 10 {
                (&hidden + 0.25).unwrap()
            } else {
                hidden.clone()
            };
            let positive = [if case == 11 {
                negative.clone()
            } else {
                positive.clone()
            }];
            let negative = [if case == 11 {
                array(&input["negative"], device)
            } else {
                negative.clone()
            }];
            let mut resolved = resolved();
            if case == 12 {
                resolved.guidance = 1.;
            }
            let fresh = model(definitions[model_index], device, fp32);
            let sample = |transformer,
                          use_cache,
                          cancel: &CancelFlag,
                          progress: &mut dyn FnMut(Progress)| {
                crate::edit::EditSampling {
                    transformer,
                    resolved: &resolved,
                    positive: &positive,
                    negative: &negative,
                    references: &references,
                    use_cache,
                    images_per_prompt: 1,
                }
                .run(&noise, cancel, progress)
            };
            let actual =
                sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
            let expected = sample(&fresh, false, &CancelFlag::new(), &mut |_| {}).unwrap();
            assert_eq!(
                values(&actual),
                values(&expected),
                "fresh uncached baseline case={case} fp32={fp32}"
            );
            if case == 0 {
                if let Some(original) = &baseline {
                    assert_eq!(values(&actual), values(original));
                } else {
                    baseline = Some(actual.clone());
                }
            } else if case != 9 {
                assert!(
                    delta(&actual, baseline.as_ref().unwrap()) > 2e-3,
                    "changed request must change real numerical output: case={case} fp32={fp32}"
                );
            }
            let cancel = CancelFlag::new();
            assert!(matches!(
                sample(&models[model_index], true, &cancel, &mut |p| {
                    if matches!(p, Progress::Step { .. }) {
                        cancel.cancel();
                    }
                }),
                Err(CandleError::Canceled)
            ));
            // Fail inside the actual network after preparing K/V, then enter another run.
            let bad_context = [positive[0].narrow(2, 0, 11).unwrap()];
            assert!(crate::edit::EditSampling {
                transformer: &models[model_index],
                resolved: &resolved,
                positive: &bad_context,
                negative: &negative,
                references: &references,
                use_cache: true,
                images_per_prompt: 1,
            }
            .run(&noise, &CancelFlag::new(), &mut |_| {})
            .is_err());
            let recovered =
                sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
            assert_eq!(
                values(&recovered),
                values(&expected),
                "cancel/error reentry case={case}"
            );
        }
        eprintln!("request-local lifecycle fp32={fp32}: 25 changed/reentry requests, each cached/fresh-uncached/cancel/network-error/recovered; 4 reference counts");
        let resolved = resolved();
        let positive_batch = [positive.clone()];
        let negative_batch = [negative.clone()];
        crate::verify_numerical_renderer_transitions(
            device,
            |mode| {
                let definition = if mode == MossImageAcceleration::Off {
                    f
                } else {
                    &changed_backbone
                };
                let dtype = if mode == MossImageAcceleration::Off {
                    DType::BF16
                } else {
                    DType::F32
                };
                let weights = Weights::from_map(
                    definition["weights"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(name, value)| {
                            (name.clone(), array(value, device).to_dtype(dtype).unwrap())
                        })
                        .collect(),
                );
                Transformer::load(&weights, models[0].g, None).unwrap()
            },
            |model| {
                values(
                    &model
                        .forward(
                            &hidden,
                            0.73,
                            &positive,
                            &vec![1; positive.dim(1).unwrap()],
                            &[],
                            &CancelFlag::new(),
                        )
                        .unwrap(),
                )
            },
            || {
                values(
                    &crate::edit::EditSampling {
                        transformer: &models[0],
                        resolved: &resolved,
                        positive: &positive_batch,
                        negative: &negative_batch,
                        references: &refs,
                        use_cache: true,
                        images_per_prompt: 1,
                    }
                    .run(&hidden, &CancelFlag::new(), &mut |_| {})
                    .unwrap(),
                )
            },
        );
    }
}

fn run(device: &Device) {
    let f = multi_fixture();
    let model = model(&f, device, false);
    let probe = &f["projection_probe"];
    let projection = model.layers[0]
        .q
        .forward(&array(&probe["input"], device))
        .unwrap();
    compare(&projection, &probe["expected"], 0., "BF16 edit projection");
    for wrong in ["wrong_fp32_residual", "wrong_merged_bf16"] {
        assert!(
            delta(&projection, &array(&probe[wrong], device)) > 0.001,
            "indistinguishable adapter policy {wrong}"
        );
    }
    time_probes(&model, &f, device);
    let old = fixture("tiny_edit_v1_time_probe/fixture.json");
    let weights = Weights::from_map(
        old["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), array(v, device)))
            .collect(),
    );
    let mut original = Transformer::load(&weights, model.g, None).unwrap();
    original.enable_edit_precision();
    time_probes(&original, &old, device);
    let input = &f["inputs"];
    let hidden = array(&input["hidden_states"], device);
    let positive = masked(&input["positive"], &input["positive_mask"], device);
    let negative = masked(&input["negative"], &input["negative_mask"], device);
    let references: Vec<_> = input["references"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| array(v, device))
        .collect();
    let time = input["normalized_timestep"].as_f64().unwrap() as f32;
    let cancel = CancelFlag::new();
    for case in f["cases"].as_array().unwrap() {
        let refs = &references[..case["reference_count"].as_u64().unwrap() as usize];
        let (tokens, rope) = model.edit_reference_inputs(refs).unwrap();
        compare(&tokens, &case["reference_tokens"], 0., "reference tokens");
        compare(&rope, &case["reference_rope"], 0., "reference coordinates");
        compare_bf16(
            &model.edit_time(0.).unwrap().0,
            &case["reference_modulation"],
            "reference t0",
        );
        compare_bf16(
            &model.edit_time(time).unwrap().0,
            &case["body_modulation"],
            "body time",
        );
        let cache = model.prepare_edit_cache(refs, &cancel).unwrap();
        assert_eq!(
            cache.keys_values.len(),
            case["reference_kv"].as_array().unwrap().len()
        );
        for ((key, value), expected) in cache
            .keys_values
            .iter()
            .zip(case["reference_kv"].as_array().unwrap())
        {
            compare_bf16(key, &expected["key"], "reference cache key");
            compare_bf16(value, &expected["value"], "reference cache value");
        }
        let predict = |context, cache| {
            model
                .forward_edit(
                    &hidden,
                    time,
                    EditConditioning {
                        context,
                        references: refs,
                        cache,
                    },
                    &cancel,
                )
                .unwrap()
        };
        let cached = predict(&positive, Some(&cache));
        let uncached = predict(&positive, None);
        compare_bf16(&cached, &case["cached"], "cached prediction");
        compare_bf16(&uncached, &case["uncached"], "uncached prediction");
        assert_eq!(values(&cached), values(&uncached));
        compare_bf16(
            &predict(&negative, Some(&cache)),
            &case["negative_cached"],
            "negative context",
        );
        for (name, control) in case["controls"].as_object().unwrap() {
            if refs.len() == 2 && name == "reversed_reference_order" {
                // Source itself has only .0923 separation at a .1948 three-bound margin.
                // Preserve the BF16 bound; order is gated by exact RoPE + strict F32
                // trajectory/mutant checks. Also execute and retain real BF16 reversal.
                let reversed: Vec<_> = refs.iter().rev().cloned().collect();
                let output = model
                    .forward_edit(
                        &hidden,
                        time,
                        EditConditioning {
                            context: &positive,
                            references: &reversed,
                            cache: None,
                        },
                        &cancel,
                    )
                    .unwrap();
                compare_bf16(
                    &output,
                    &control["output"],
                    "two-reference reversed BF16 prediction",
                );
                let difference = delta(&cached, &output);
                eprintln!(
                    "two-reference BF16 order diagnostic: native={difference}, source={}",
                    control["max_abs_difference"]
                );
                assert!(difference > 0., "actual BF16 graph must retain order");
                continue;
            }
            assert!(
                delta(&cached, &array(&control["output"], device))
                    > 3. * bf16_bound(&case["cached"]),
                "undiscriminated graph control {name}"
            );
        }
        let resolved = resolved();
        let positive = [positive.clone()];
        let negative = [negative.clone()];
        let sampling = crate::edit::EditSampling {
            transformer: &model,
            resolved: &resolved,
            positive: &positive,
            negative: &negative,
            references: refs,
            use_cache: true,
            images_per_prompt: 1,
        };
        let mut steps = Vec::new();
        let sampled = sampling
            .run(&hidden, &CancelFlag::new(), &mut |p| {
                if let Progress::Step { current, total } = p {
                    steps.push((current, total));
                }
            })
            .unwrap();
        assert_eq!(steps, [(1, 2), (2, 2)]);
        let stop = CancelFlag::new();
        assert!(matches!(
            sampling.run(
                &hidden,
                &stop,
                &mut |p| if matches!(p, Progress::Step { .. }) {
                    stop.cancel();
                }
            ),
            Err(CandleError::Canceled)
        ));
        let reentry = sampling
            .run(&hidden, &CancelFlag::new(), &mut |_| {})
            .unwrap();
        assert_eq!(values(&sampled), values(&reentry));
    }
    sampler_precision(&f, device, false);
    request_local_lifecycle(&f, device);
    cancel.cancel();
    assert!(matches!(
        model.prepare_edit_cache(&references, &cancel),
        Err(CandleError::Canceled)
    ));
}
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn frozen_edit_oracle_cpu() {
    run(&Device::Cpu);
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::edit::tests::frozen_edit_oracle_cpu");
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn frozen_edit_oracle_cuda() {
    // Do not mutate process-global precision flags shared by other GPU tests.
    assert!(!candle_gen::candle_core::cuda_backend::gemm_reduced_precision_f32());
    assert!(!candle_gen::candle_core::cuda_backend::gemm_reduced_precision_bf16());
    run(&Device::new_cuda(0).expect("admitted CUDA lane logical device 0"));
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::edit::tests::frozen_edit_oracle_cuda");
}

#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn frozen_edit_strict_f32_sampler_cpu() {
    sampler_precision(&multi_fixture(), &Device::Cpu, true);
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::edit::tests::frozen_edit_strict_f32_sampler_cpu");
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "model inference; explicit model-proof workflow"]
fn frozen_edit_strict_f32_sampler_cuda() {
    assert!(!candle_gen::candle_core::cuda_backend::gemm_reduced_precision_f32());
    assert!(!candle_gen::candle_core::cuda_backend::gemm_reduced_precision_bf16());
    sampler_precision(
        &multi_fixture(),
        &Device::new_cuda(0).expect("admitted CUDA lane logical device 0"),
        true,
    );
    #[cfg(feature = "model-inference-proof")]
    println!("\nMODEL_INFERENCE_BODY_COMPLETED transformer::edit::tests::frozen_edit_strict_f32_sampler_cuda");
}

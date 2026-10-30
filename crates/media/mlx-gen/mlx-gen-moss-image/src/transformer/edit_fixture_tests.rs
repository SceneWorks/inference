use super::*;
use crate::pipeline::logical_f32_readback;
use serde_json::Value;
fn array(value: &Value) -> Array {
    let shape = value["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let values = value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    Array::from_slice(&values, &shape)
        .as_dtype(if value["dtype"] == "bfloat16" {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        })
        .unwrap()
}
fn compare(actual: &Array, expected: &Value, tolerance: f32, label: &str) {
    let expected_shape = expected["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    let expected_values = expected["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect::<Vec<_>>();
    let got = logical_f32_readback(actual).unwrap();
    let delta = checked_delta(&got, actual.shape(), &expected_values, &expected_shape);
    eprintln!("{label}: max_abs={delta}, bound={tolerance}");
    assert!(
        delta <= tolerance,
        "{label}: maximum absolute error {delta} > {tolerance}"
    );
}
fn bf16_bound(expected: &Value) -> f32 {
    let magnitude = expected["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| (v.as_f64().unwrap() as f32).abs())
        .fold(0.0_f32, f32::max);
    magnitude / 64.0 + 1.0 / 1024.0
}
fn compare_bf16(actual: &Array, expected: &Value, label: &str) {
    compare(actual, expected, bf16_bound(expected), label);
}
fn time_probes(model: &Transformer, f: &Value) {
    for probe in f["time_probes"].as_array().unwrap() {
        // Each stage also receives the independent upstream input, locating
        // backend rounding without allowing accumulated errors to hide it.
        for (label, actual, expected) in [
            (
                "time first linear",
                model.time1.forward(&array(&probe["time1_input"])).unwrap(),
                &probe["time1"],
            ),
            (
                "time exact SiLU",
                edit_silu(&array(&probe["silu_input"])).unwrap(),
                &probe["silu"],
            ),
            (
                "time second linear",
                model.time2.forward(&array(&probe["time2_input"])).unwrap(),
                &probe["time2"],
            ),
            (
                "time shared modulation",
                model
                    .modulation
                    .forward(&array(&probe["modulation_flat_input"]))
                    .unwrap(),
                &probe["modulation_flat"],
            ),
            (
                "time head modulation",
                model
                    .head
                    .forward(&array(&probe["head_flat_input"]))
                    .unwrap(),
                &probe["head_flat"],
            ),
        ] {
            if matches!(label, "time first linear" | "time exact SiLU") {
                compare(&actual, expected, 0.0, label);
            } else {
                compare_bf16(&actual, expected, label);
            }
        }
        let split_bias = add(
            array(&probe["time1_input"])
                .matmul(model.time1.weight.t())
                .unwrap(),
            model.time1.bias.as_ref().unwrap(),
        )
        .unwrap();
        let eager_silu = mlx_rs::fast::silu_exact(array(&probe["silu_input"])).unwrap();
        for (wrong, expected) in [
            (&split_bias, &probe["time1"]),
            (&eager_silu, &probe["silu"]),
        ] {
            assert_ne!(
                logical_f32_readback(wrong).unwrap(),
                logical_f32_readback(&array(expected)).unwrap(),
                "rounding control must distinguish source-fused arithmetic"
            );
        }
        let (modulation, head) = model
            .edit_time(probe["normalized_timestep"].as_f64().unwrap() as f32)
            .unwrap();
        compare_bf16(
            &modulation,
            &probe["modulation"],
            "complete time modulation",
        );
        compare_bf16(&head, &probe["head"], "complete time head");
    }
}
fn masked(value: &Value, mask: &Value) -> Array {
    let indices = mask["values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter_map(|(i, v)| (v.as_f64().unwrap() != 0.0).then_some(i as i32))
        .collect::<Vec<_>>();
    array(value)
        .take_axis(Array::from_slice(&indices, &[indices.len() as i32]), 1)
        .unwrap()
}
#[test]
fn edit_bf16_source_leaf_cpu() {
    const CHILD: &str = "MOSS_EDIT_LEAF_CPU_CHILD";
    if std::env::var(CHILD).as_deref() != Ok("1") {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transformer::edit_fixture_tests::edit_bf16_source_leaf_cpu",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "CPU leaf child: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
        let fixture: Value = serde_json::from_str(include_str!(
                "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_precision/leaf_ops.json"
            )).unwrap();
        let cancel = CancelFlag::new();
        for (i, call) in fixture["attention"].as_array().unwrap().iter().enumerate() {
            let (q, k, v) = (array(&call["q"]), array(&call["k"]), array(&call["v"]));
            let corrected = attention_edit_bf16(&q, &k, &v, &cancel).unwrap();
            compare(
                &corrected,
                &call["output"],
                0.0,
                &format!("source SDPA {i}"),
            );
            let unrounded = attention_fp32(
                &q.as_dtype(Dtype::Float32).unwrap(),
                &k.as_dtype(Dtype::Float32).unwrap(),
                &v.as_dtype(Dtype::Float32).unwrap(),
                1.0 / (q.shape()[3] as f32).sqrt(),
                &cancel,
            )
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            assert_ne!(
                logical_f32_readback(&corrected).unwrap(),
                logical_f32_readback(&unrounded).unwrap(),
                "source SDPA rounding discriminator {i}"
            );
            // Repeated independent queries preserve the source output; 17 rows
            // exercise two complete tiles and a partial tile, with all keys visible.
            if q.shape()[2] == 8 {
                let first = slice_axis(&q, 2, 0, 1).unwrap();
                let repeated = concatenate_axis(&[&q, &q, &first], 2).unwrap();
                let source = array(&call["output"]);
                let first_source = slice_axis(&source, 2, 0, 1).unwrap();
                let expected = concatenate_axis(&[&source, &source, &first_source], 2).unwrap();
                let actual = attention_edit_bf16(&repeated, &k, &v, &cancel).unwrap();
                assert_eq!(actual.shape()[2], 17);
                assert_eq!(
                    logical_f32_readback(&actual).unwrap(),
                    logical_f32_readback(&expected).unwrap()
                );
            }
        }
        for (i, call) in fixture["situ"].as_array().unwrap().iter().enumerate() {
            let (hidden, gate) = (array(&call["hidden"]), array(&call["gate"]));
            let corrected = edit_situ(&hidden, &gate).unwrap();
            compare(
                &corrected,
                &call["output"],
                0.0,
                &format!("source SiTU {i}"),
            );
            let unrounded = situ(
                &hidden.as_dtype(Dtype::Float32).unwrap(),
                &gate.as_dtype(Dtype::Float32).unwrap(),
            )
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            assert_ne!(
                logical_f32_readback(&corrected).unwrap(),
                logical_f32_readback(&unrounded).unwrap(),
                "source SiTU eager rounding discriminator {i}"
            );
        }
    });
}
#[test]
fn edit_projection_and_reference_coordinates_cpu() {
    const CHILD: &str = "MOSS_EDIT_PROJECTION_CPU_CHILD";
    if std::env::var(CHILD).as_deref() != Ok("1") {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transformer::edit_fixture_tests::edit_projection_and_reference_coordinates_cpu",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "CPU edit projection child: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
    mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || run_oracle(true));
}
#[test]
#[ignore = "requires coordinator-admitted Metal ownership; exact silu kernel is Metal-only"]
fn frozen_edit_oracle_on_metal() {
    assert_eq!(
        std::env::var("MOSS_NATIVE_TEST_ADMITTED").as_deref(),
        Ok("1")
    );
    mlx_rs::with_new_default_stream(mlx_rs::Stream::gpu(), || {
        author_fixture_tests::ordinary_author_time_conformance();
        run_oracle(false);
        trained_custom_adapters_compose_with_edit();
    });
}
// Reached only through the existing ignored Metal edit oracle. Exercise actual learning,
// portable export and the production custom-before-edit installation order for both methods.
fn trained_custom_adapters_compose_with_edit() {
    use gen_core_moss_image::moss_image_training as contract;
    use mlx_gen::gen_core::moss_image::MossImageVariant;
    use mlx_gen::train::lora::{build_lokr_targets, build_lora_targets, TrainAdapter};
    use mlx_gen::{NetworkType, TrainingConfig};
    let f: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
    ))
    .unwrap();
    let backbone = |fp32| {
        let weights = Weights::from_map(
            f["weights"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| {
                    let value = array(value);
                    (
                        key.clone(),
                        if fp32 {
                            value.as_dtype(Dtype::Float32).unwrap()
                        } else {
                            value
                        },
                    )
                })
                .collect(),
        );
        Transformer::load(
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
        )
        .unwrap()
    };
    let edit_weights = || {
        Weights::from_map(
            f["adapter"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(key, value)| (key.clone(), array(value)))
                .collect(),
        )
    };
    let input = &f["inputs"];
    let x = array(&input["hidden_states"]);
    let context = masked(&input["positive"], &input["positive_mask"]);
    let references: Vec<_> = input["references"]
        .as_array()
        .unwrap()
        .iter()
        .map(array)
        .collect();
    let noise = add(&x, Array::from_f32(0.37)).unwrap();
    let cancel = CancelFlag::new();
    let predict = |model: &Transformer, cached: bool| {
        let cache = cached.then(|| model.prepare_edit_cache(&references[..1], &cancel).unwrap());
        logical_f32_readback(
            &model
                .forward_edit(
                    &x,
                    0.37,
                    EditConditioning {
                        context: &context,
                        references: &references[..1],
                        cache: cache.as_ref(),
                    },
                    &cancel,
                )
                .unwrap(),
        )
        .unwrap()
    };
    let mut edit_only = backbone(false);
    edit_only.install_edit(&edit_weights(), 2).unwrap();
    let baseline = predict(&edit_only, true);
    let paths = vec!["layers.0.proj.o".to_owned()];
    for network in [NetworkType::Lora, NetworkType::Lokr] {
        eprintln!("custom edit {network:?}: native training");
        let mut trained = backbone(true);
        let (adapter, mut params) = match network {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(&mut trained, &paths, 2, 17).unwrap();
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(&mut trained, &paths, 2, 2, 17).unwrap();
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let config = TrainingConfig {
            rank: 2,
            alpha: 3.,
            network_type: network,
            decompose_factor: 2,
            ..Default::default()
        };
        let mut optimizer = mlx_gen::TrainOptimizer::from_config("adamw", 0.1, 0.0).unwrap();
        for update in 0..3 {
            let (loss, gradient) = crate::training::loss_grads(
                &mut trained,
                &params,
                &adapter,
                &config,
                &x,
                &context,
                &noise,
                0.37,
                None,
                &cancel,
            )
            .unwrap();
            assert!(loss.is_finite());
            assert!(gradient
                .values()
                .flat_map(|array| logical_f32_readback(array).unwrap())
                .any(|value| value != 0.));
            mlx_gen::train::lora::adapter_optimizer_update(
                &mut optimizer,
                &mut params,
                &gradient,
                &config,
                update,
                17,
            )
            .unwrap();
        }
        let root = tempfile::tempdir().unwrap();
        eprintln!("custom edit {network:?}: factor/optimizer save and CPU-backed reload");
        let snapshot = root.path().join("resume-factors.safetensors");
        Array::save_safetensors(
            params.iter().map(|(key, value)| (key.as_ref(), value)),
            None,
            &snapshot,
        )
        .unwrap();
        let (restored, _) = mlx_gen::weights::load_safetensors_on_cpu(&snapshot).unwrap();
        mlx_rs::transforms::eval_pending_loads(restored.values()).unwrap();
        let mut restored: mlx_gen::train::lora::LoraParams = restored
            .into_iter()
            .map(|(key, value)| (key.into(), value))
            .collect();
        let optimizer_snapshot = root.path().join("optimizer.safetensors");
        optimizer.save_state(&optimizer_snapshot).unwrap();
        let mut resumed_optimizer =
            mlx_gen::TrainOptimizer::from_config("adamw", 0.1, 0.0).unwrap();
        resumed_optimizer
            .load_state_checked(&optimizer_snapshot, &restored, 3)
            .unwrap();
        for (key, value) in &params {
            assert_eq!(
                logical_f32_readback(value).unwrap(),
                logical_f32_readback(&restored[key]).unwrap(),
                "resumed raw factor bytes differ"
            );
        }
        let (_, gradient) = crate::training::loss_grads(
            &mut trained,
            &params,
            &adapter,
            &config,
            &x,
            &context,
            &noise,
            0.37,
            None,
            &cancel,
        )
        .unwrap();
        mlx_gen::train::lora::adapter_optimizer_update(
            &mut optimizer,
            &mut params,
            &gradient,
            &config,
            3,
            17,
        )
        .unwrap();
        mlx_gen::train::lora::adapter_optimizer_update(
            &mut resumed_optimizer,
            &mut restored,
            &gradient,
            &config,
            3,
            17,
        )
        .unwrap();
        for (key, value) in &params {
            assert_eq!(
                logical_f32_readback(value).unwrap(),
                logical_f32_readback(&restored[key]).unwrap(),
                "resumed optimizer update differs"
            );
        }
        eprintln!("custom edit {network:?}: portable export");
        let path = root.path().join("trained.safetensors");
        let metadata = contract::metadata(MossImageVariant::Sft, contract::SAMPLER);
        let extra: Vec<_> = metadata
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_str()))
            .collect();
        crate::adapters::save_validated(&mut trained, &adapter, &params, &config, &extra, &path)
            .unwrap();
        let kind = match network {
            NetworkType::Lora => mlx_gen::runtime::AdapterKind::Lora,
            NetworkType::Lokr => mlx_gen::runtime::AdapterKind::Lokr,
        };
        for strength in [0., 1., -1.] {
            eprintln!("custom edit {network:?}: pinned custom load, strength={strength}");
            let mut model = backbone(false);
            let spec = mlx_gen::runtime::AdapterSpec::new(path.clone(), strength, kind);
            let pin = mlx_gen::gen_core::PinnedWeightsFile::pin(&path).unwrap();
            let reports =
                crate::adapters::apply_pinned(&mut model, MossImageVariant::Sft, &[spec], &[pin])
                    .unwrap();
            assert_eq!(reports[0].applied, 1);
            eprintln!("custom edit {network:?}: edit prediction, strength={strength}");
            model.install_edit(&edit_weights(), 2).unwrap();
            let output = predict(&model, true);
            assert!(output.iter().all(|value| value.is_finite()));
            assert_eq!(
                output,
                predict(&model, false),
                "custom adapters changed cached/uncached edit equivalence"
            );
            if strength == 0. {
                assert_eq!(
                    output, baseline,
                    "disabled custom adapter changed edit arithmetic"
                );
            } else {
                assert_ne!(
                    output, baseline,
                    "trained {network:?} residual disappeared during edit at strength {strength}"
                );
            }
        }
    }
}
fn fixture_model(f: &Value, fp32: bool) -> Transformer {
    let weights = Weights::from_map(
        f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| {
                let value = array(v);
                (
                    k.clone(),
                    if fp32 {
                        value.as_dtype(Dtype::Float32).unwrap()
                    } else {
                        value
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
    )
    .unwrap();
    let adapter = Weights::from_map(
        f["adapter"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), array(v)))
            .collect(),
    );
    model.install_edit(&adapter, 2).unwrap();
    if fp32 {
        // Diagnostic only: same represented source weights cast to FP32. The
        // production package validator continues requiring BF16 A/B tensors.
        for block in &mut model.blocks {
            for p in &mut block.projections {
                if let Some((a, b, scale)) = &p.adapter {
                    p.adapter = Some((
                        a.as_dtype(Dtype::Float32).unwrap(),
                        b.as_dtype(Dtype::Float32).unwrap(),
                        *scale,
                    ));
                }
            }
        }
    }
    model
}
// Separate network error from integration: BF16 predictions are compared at
// identical source states; FP32 represented-weight sampling checks the entire graph.
// The historical final-only BF16 bound did not compose through CFG and state feedback.
fn sampler_precision(f: &Value) {
    const FP32_BOUND: f32 = 2e-5;
    let mut oracle: Value = serde_json::from_str(include_str!("../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v4_sampler_analysis/fp32_sampler.json")).unwrap();
    let extra: Value = serde_json::from_str(include_str!("../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_multi/fp32_sampler.json")).unwrap();
    oracle["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    let input = &f["inputs"];
    for fp32 in [false, true] {
        let model = fixture_model(f, fp32);
        for case in f["cases"].as_array().unwrap() {
            let count = case["reference_count"].as_u64().unwrap() as usize;
            let refs = input["references"].as_array().unwrap()[..count]
                .iter()
                .map(array)
                .collect::<Vec<_>>();
            let positive = [masked(&input["positive"], &input["positive_mask"])];
            let negative = [masked(&input["negative"], &input["negative_mask"])];
            let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                &gen_core_moss_image::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                    steps: Some(2),
                    guidance: Some(4.0),
                    width: 48,
                    height: 32,
                    ..Default::default()
                })
                .unwrap(),
            )
            .unwrap();
            let sampling = crate::edit::EditSampling {
                transformer: &model,
                resolved: &resolved,
                positive: &positive,
                negative: &negative,
                references: &refs,
                use_cache: true,
            };
            crate::edit::SAMPLING_TRACE.with(|t| *t.borrow_mut() = Some(Vec::new()));
            let output = sampling
                .run(
                    &array(&input["hidden_states"]),
                    &CancelFlag::new(),
                    &mut |_| {},
                )
                .unwrap();
            let trace = crate::edit::SAMPLING_TRACE.with(|t| t.borrow_mut().take().unwrap());
            assert_eq!(trace.len(), 2);
            assert_eq!(output.dtype(), Dtype::Float32);
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
                    FP32_BOUND,
                    "strict FP32 complete sampler",
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
                        assert_eq!(a.dtype(), Dtype::Float32);
                        compare(a, &step[key], FP32_BOUND, "strict FP32 sampler stage");
                    }
                }
                for (name, control) in expected["controls"].as_object().unwrap() {
                    let wrong = array(&control["final"]);
                    let delta = checked_delta(
                        &logical_f32_readback(&output).unwrap(),
                        output.shape(),
                        &logical_f32_readback(&wrong).unwrap(),
                        wrong.shape(),
                    );
                    eprintln!(
                        "full sampler semantic control {name}: delta={delta}, bound={FP32_BOUND}"
                    );
                    assert!(
                        delta > 100.0 * FP32_BOUND,
                        "undiscriminated full sampler control {name}: {delta}"
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
                                &array(&source["input"]),
                                time,
                                EditConditioning {
                                    context,
                                    references: &refs,
                                    cache: Some(&cache),
                                },
                                &CancelFlag::new(),
                            )
                            .unwrap();
                        assert_eq!(
                            prediction.dtype(),
                            Dtype::Bfloat16,
                            "production edit prediction dtype"
                        );
                        compare_bf16(
                            &prediction,
                            &source[key],
                            "BF16 prediction at source step input",
                        );
                    }
                    let values = trace[index]
                        .iter()
                        .map(|a| {
                            assert_eq!(a.dtype(), Dtype::Float32);
                            logical_f32_readback(a).unwrap()
                        })
                        .collect::<Vec<_>>();
                    let dt = sigmas[index + 1].as_f64().unwrap() as f32 - time;
                    let mut error = 0.0_f32;
                    for (i, &current) in values[0].iter().enumerate() {
                        let guided = values[2][i] + 4.0 * (values[1][i] - values[2][i]);
                        error = error.max((values[3][i] - (current + dt * guided)).abs());
                    }
                    assert!(
                        error <= 1e-6,
                        "actual production sampler composition error {error}"
                    );
                }
            }
        }
    }
}

fn lifecycle_values(a: &Array) -> Vec<f32> {
    logical_f32_readback(a).unwrap()
}
fn lifecycle_delta(a: &Array, b: &Array) -> f32 {
    checked_delta(
        &lifecycle_values(a),
        a.shape(),
        &lifecycle_values(b),
        b.shape(),
    )
}
// Successive request discriminators execute actual tiny Transformer/LoRA and sampler math.
// Fresh model + uncached sampling is an independent lifetime baseline, not a new golden.
fn request_local_lifecycle(f: &Value) {
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
        let models: Vec<_> = definitions.iter().map(|f| fixture_model(f, fp32)).collect();
        let input = &f["inputs"];
        let refs: Vec<_> = input["references"]
            .as_array()
            .unwrap()
            .iter()
            .map(array)
            .collect();
        let hidden = array(&input["hidden_states"]);
        let positive = masked(&input["positive"], &input["positive_mask"]);
        let negative = masked(&input["negative"], &input["negative_mask"]);
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
                8 => {
                    references[1] =
                        multiply(&references[1], Array::from_slice(&[-1.0_f32], &[1])).unwrap()
                }
                _ => (),
            }
            let noise = if case == 9 {
                slice_axis(&hidden, 3, 0, 2).unwrap()
            } else if case == 10 {
                add(&hidden, Array::from_slice(&[0.25_f32], &[1])).unwrap()
            } else {
                hidden.clone()
            };
            let positive = [if case == 11 {
                negative.clone()
            } else {
                positive.clone()
            }];
            let negative = [if case == 11 {
                array(&input["negative"])
            } else {
                negative.clone()
            }];
            let mut resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
                mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
                &gen_core_moss_image::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                    width: 48,
                    height: 32,
                    steps: Some(2),
                    guidance: Some(4.),
                    ..Default::default()
                })
                .unwrap(),
            )
            .unwrap();
            if case == 12 {
                resolved.guidance = 1.;
            }
            let fresh = fixture_model(definitions[model_index], fp32);
            let sample = |transformer,
                          use_cache,
                          cancel: &CancelFlag,
                          progress: &mut dyn FnMut(mlx_gen::Progress)| {
                crate::edit::EditSampling {
                    transformer,
                    resolved: &resolved,
                    positive: &positive,
                    negative: &negative,
                    references: &references,
                    use_cache,
                }
                .run(&noise, cancel, progress)
            };
            let actual =
                sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
            let expected = sample(&fresh, false, &CancelFlag::new(), &mut |_| {}).unwrap();
            assert_eq!(
                lifecycle_values(&actual),
                lifecycle_values(&expected),
                "fresh uncached baseline case={case} fp32={fp32}"
            );
            if case == 0 {
                if let Some(original) = &baseline {
                    assert_eq!(lifecycle_values(&actual), lifecycle_values(original));
                } else {
                    baseline = Some(actual.clone());
                }
            } else if case != 9 {
                assert!(
                    lifecycle_delta(&actual, baseline.as_ref().unwrap()) > 2e-3,
                    "changed request must change real numerical output: case={case} fp32={fp32}"
                );
            }
            let cancel = CancelFlag::new();
            assert!(matches!(
                sample(&models[model_index], true, &cancel, &mut |p| {
                    if matches!(p, mlx_gen::Progress::Step { .. }) {
                        cancel.cancel();
                    }
                }),
                Err(Error::Canceled)
            ));
            // Fail inside the actual network after preparing K/V, then enter another run.
            let bad_context = [slice_axis(&positive[0], 2, 0, 11).unwrap()];
            assert!(crate::edit::EditSampling {
                transformer: &models[model_index],
                resolved: &resolved,
                positive: &bad_context,
                negative: &negative,
                references: &references,
                use_cache: true,
            }
            .run(&noise, &CancelFlag::new(), &mut |_| {})
            .is_err());
            let recovered =
                sample(&models[model_index], true, &CancelFlag::new(), &mut |_| {}).unwrap();
            assert_eq!(
                lifecycle_values(&recovered),
                lifecycle_values(&expected),
                "cancel/error reentry case={case}"
            );
        }
        eprintln!("request-local lifecycle fp32={fp32}: 25 changed/reentry requests, each cached/fresh-uncached/cancel/network-error/recovered; 4 reference counts");
        let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
            mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
            &gen_core_moss_image::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                width: 48,
                height: 32,
                steps: Some(2),
                guidance: Some(4.),
                ..Default::default()
            })
            .unwrap(),
        )
        .unwrap();
        let positive_batch = [positive.clone()];
        let negative_batch = [negative.clone()];
        crate::model::verify_numerical_renderer_transitions(
            |mode| {
                let ordinary = mode == mlx_gen::gen_core::moss_image::MossImageAcceleration::Off;
                let definition = if ordinary { f } else { &changed_backbone };
                let dtype = if ordinary {
                    Dtype::Bfloat16
                } else {
                    Dtype::Float32
                };
                let weights = Weights::from_map(
                    definition["weights"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(name, value)| (name.clone(), array(value).as_dtype(dtype).unwrap()))
                        .collect(),
                );
                Transformer::load(&weights, models[0].g.clone()).unwrap()
            },
            |model| {
                lifecycle_values(
                    &model
                        .forward(&hidden, &[0.73], &positive_batch, &[], &CancelFlag::new())
                        .unwrap(),
                )
            },
            || {
                lifecycle_values(
                    &crate::edit::EditSampling {
                        transformer: &models[0],
                        resolved: &resolved,
                        positive: &positive_batch,
                        negative: &negative_batch,
                        references: &refs,
                        use_cache: true,
                    }
                    .run(&hidden, &CancelFlag::new(), &mut |_| {})
                    .unwrap(),
                )
            },
        );
    }
}

fn run_oracle(cpu_only: bool) {
    let mut f: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
    ))
    .unwrap();
    let extra: Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_multi/fixture.json"
    ))
    .unwrap();
    assert_eq!(f["weights"], extra["weights"]);
    assert_eq!(f["adapter"], extra["adapter"]);
    f["cases"]
        .as_array_mut()
        .unwrap()
        .extend(extra["cases"].as_array().unwrap().iter().cloned());
    let weights = Weights::from_map(
        f["weights"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), array(v)))
            .collect(),
    );
    let adapter = Weights::from_map(
        f["adapter"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), array(v)))
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
    )
    .unwrap();
    model.install_edit(&adapter, 2).unwrap();
    let probe = &f["projection_probe"];
    let projected = model.blocks[0].projections[0]
        .forward(&array(&probe["input"]))
        .unwrap();
    compare(&projected, &probe["expected"], 0.0, "BF16 edit projection");
    if !cpu_only {
        let original: Value = serde_json::from_str(include_str!(
                "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v1_time_probe/fixture.json"
            )).unwrap();
        let original_weights = Weights::from_map(
            original["weights"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), array(v)))
                .collect(),
        );
        let mut original_model = Transformer::load(&original_weights, model.g.clone()).unwrap();
        original_model.enable_edit_precision();
        time_probes(&original_model, &original);
        time_probes(&model, &f);
    }
    let input = &f["inputs"];
    let hidden = array(&input["hidden_states"]);
    let positive = masked(&input["positive"], &input["positive_mask"]);
    let negative = masked(&input["negative"], &input["negative_mask"]);
    let references = input["references"]
        .as_array()
        .unwrap()
        .iter()
        .map(array)
        .collect::<Vec<_>>();
    let time = input["normalized_timestep"].as_f64().unwrap() as f32;
    let cancel = CancelFlag::new();
    for case in f["cases"].as_array().unwrap() {
        let refs = &references[..case["reference_count"].as_u64().unwrap() as usize];
        let (tokens, rope) = model.edit_reference_inputs(refs).unwrap();
        compare(&tokens, &case["reference_tokens"], 0.0, "reference tokens");
        compare(&rope, &case["reference_rope"], 0.0, "reference coordinates");
        if cpu_only {
            continue;
        }
        let (reference_modulation, _) = model.edit_time(0.0).unwrap();
        compare_bf16(
            &reference_modulation,
            &case["reference_modulation"],
            "reference t0",
        );
        let (body_modulation, _) = model.edit_time(time).unwrap();
        compare_bf16(&body_modulation, &case["body_modulation"], "body time");
        let cache = model.prepare_edit_cache(refs, &cancel).unwrap();
        for ((key, value), expected) in cache
            .keys_values
            .iter()
            .zip(case["reference_kv"].as_array().unwrap())
        {
            compare_bf16(key, &expected["key"], "reference cache key");
            compare_bf16(value, &expected["value"], "reference cache value");
        }
        let cached = model
            .forward_edit(
                &hidden,
                time,
                EditConditioning {
                    context: &positive,
                    references: refs,
                    cache: Some(&cache),
                },
                &cancel,
            )
            .unwrap();
        let uncached = model
            .forward_edit(
                &hidden,
                time,
                EditConditioning {
                    context: &positive,
                    references: refs,
                    cache: None,
                },
                &cancel,
            )
            .unwrap();
        compare_bf16(&cached, &case["cached"], "cached prediction");
        compare_bf16(&uncached, &case["uncached"], "uncached prediction");
        for (name, control) in case["controls"].as_object().unwrap() {
            if refs.len() == 2 && name == "reversed_reference_order" {
                // Two-reference source BF16 reversal cannot meet the unchanged 3x
                // margin. Exact ordinal RoPE + strict F32 trajectory/mutant gates
                // establish order; this still runs the actual reversed BF16 graph.
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
                let difference = lifecycle_delta(&cached, &output);
                eprintln!(
                    "two-reference BF16 order diagnostic: native={difference}, source={}",
                    control["max_abs_difference"]
                );
                assert!(difference > 0., "actual BF16 graph must retain order");
                continue;
            }
            let delta = checked_delta(
                &logical_f32_readback(&cached).unwrap(),
                cached.shape(),
                &logical_f32_readback(&array(&control["output"])).unwrap(),
                cached.shape(),
            );
            assert!(
                delta > 3.0 * bf16_bound(&case["cached"]),
                "undiscriminated control {name}: {delta}"
            );
        }
        let negative = model
            .forward_edit(
                &hidden,
                time,
                EditConditioning {
                    context: &negative,
                    references: refs,
                    cache: Some(&cache),
                },
                &cancel,
            )
            .unwrap();
        compare_bf16(&negative, &case["negative_cached"], "negative prediction");
        assert_eq!(
            logical_f32_readback(&cached).unwrap(),
            logical_f32_readback(&uncached).unwrap()
        );
        let resolved = mlx_gen::gen_core::moss_image::MossImageResolved::resolve(
            mlx_gen::gen_core::moss_image::MossImageVariant::Sft,
            &gen_core_moss_image::moss_image_edit::edit_request(&mlx_gen::GenerationRequest {
                steps: Some(2),
                guidance: Some(4.0),
                width: 48,
                height: 32,
                ..Default::default()
            })
            .unwrap(),
        )
        .unwrap();
        let positive_batch = [positive.clone()];
        let negative_batch = [masked(&input["negative"], &input["negative_mask"])];
        let sampling = crate::edit::EditSampling {
            transformer: &model,
            resolved: &resolved,
            positive: &positive_batch,
            negative: &negative_batch,
            references: refs,
            use_cache: true,
        };
        let mut steps = Vec::new();
        let sampled = sampling
            .run(&hidden, &cancel, &mut |p| {
                if let mlx_gen::Progress::Step { current, total } = p {
                    steps.push((current, total))
                }
            })
            .unwrap();
        assert_eq!(sampled.dtype(), Dtype::Float32, "sampler accumulator dtype");
        // Aggregate BF16 drift is diagnostic, not a fictitious final-latent ULP gate.
        let expected = array(&case["sampling"]["final"]);
        eprintln!(
            "BF16 complete sampler aggregate drift: {}",
            checked_delta(
                &logical_f32_readback(&sampled).unwrap(),
                sampled.shape(),
                &logical_f32_readback(&expected).unwrap(),
                expected.shape()
            )
        );
        assert_eq!(steps, [(1, 2), (2, 2)]);
        let stop = CancelFlag::new();
        assert!(matches!(
            sampling.run(
                &hidden,
                &stop,
                &mut |p| if matches!(p, mlx_gen::Progress::Step { .. }) {
                    stop.cancel()
                }
            ),
            Err(Error::Canceled)
        ));
        let reentry = sampling
            .run(&hidden, &CancelFlag::new(), &mut |_| {})
            .unwrap();
        assert_eq!(
            logical_f32_readback(&sampled).unwrap(),
            logical_f32_readback(&reentry).unwrap()
        );
    }
    if !cpu_only {
        sampler_precision(&f);
        request_local_lifecycle(&f);
    }
    cancel.cancel();
    assert!(matches!(
        model.prepare_edit_cache(&references, &cancel),
        Err(Error::Canceled)
    ));
}

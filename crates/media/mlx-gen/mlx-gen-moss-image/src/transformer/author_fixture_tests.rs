use super::*;
use mlx_rs::ops::subtract;
use std::collections::HashMap;
pub(super) fn fixture(name: &str) -> serde_json::Value {
    serde_json::from_str(match name{"ordinary"=>include_str!("../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/fixture.json"),_=>include_str!("../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_author/dmd_fixture.json")}).unwrap()
}
pub(super) fn array(v: &serde_json::Value) -> Array {
    let shape: Vec<i32> = v["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let values: Vec<f32> = v["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    Array::from_slice(&values, &shape)
}
pub(super) fn geometry(v: &serde_json::Value) -> Geometry {
    let c = &v["config"];
    Geometry {
        dim: c["dim"].as_i64().unwrap() as i32,
        ffn: c["ffn_dim"].as_i64().unwrap() as i32,
        heads: c["num_heads"].as_i64().unwrap() as i32,
        layers: c["num_layers"].as_u64().unwrap() as usize,
        text: c["text_dim"].as_i64().unwrap() as i32,
        freq: c["freq_dim"].as_i64().unwrap() as i32,
        channels: c["latent_channels"].as_i64().unwrap() as i32,
    }
}
pub(super) fn weights(v: &serde_json::Value, dtype: Dtype) -> Weights {
    Weights::from_map(
        v.as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    array(value)
                        .as_dtype(if key == "rope.bands" {
                            Dtype::Float32
                        } else {
                            dtype
                        })
                        .unwrap(),
                )
            })
            .collect(),
    )
}
pub(super) fn contexts(v: &serde_json::Value, mask: &serde_json::Value, text: i32) -> Vec<Array> {
    let source = array(v);
    let mask = mask["values"].as_array().unwrap();
    let s = source.shape()[1] as usize;
    (0..source.shape()[0])
        .map(|b| {
            let row = slice_axis(&source, 0, b, 1).unwrap();
            let ids: Vec<i32> = (0..s)
                .filter(|i| mask[b as usize * s + *i].as_f64().unwrap() != 0.0)
                .map(|i| i as i32)
                .collect();
            row.take_axis(Array::from_slice(&ids, &[ids.len() as i32]), 1)
                .unwrap()
                .reshape(&[1, ids.len() as i32, text])
                .unwrap()
        })
        .collect()
}
fn delta(actual: &Array, expected: &serde_json::Value) -> f32 {
    // Host slices expose physical storage; a B=1 concatenation can retain a transposed view.
    let actual = mlx_gen::array::contiguous(&actual.as_dtype(Dtype::Float32).unwrap()).unwrap();
    eval([&actual]).unwrap();
    let expected_values: Vec<f32> = serde_json::from_value(expected["values"].clone()).unwrap();
    let expected_shape: Vec<i32> = serde_json::from_value(expected["shape"].clone()).unwrap();
    super::checked_delta(
        actual.try_as_slice::<f32>().unwrap(),
        actual.shape(),
        &expected_values,
        &expected_shape,
    )
}

// These original default-Metal numerical proofs run only on the explicit hosted
// moss-metal-fixture profile. Their device contract is not converted to CPU.
#[test]
#[ignore = "manual MOSS Metal model proof; CI lanes=moss-metal-fixture"]
fn frozen_author_full_graph_ragged_masks_fp32_bf16_and_skipping() {
    #[cfg(feature = "test-support")]
    crate::test_support::assert_default_metal_profile_device();
    ordinary_author_time_conformance();
}

pub(super) fn ordinary_author_time_conformance() {
    // CI_IMPACT: reuse actual nn.SiLU hooks from the frozen author edit packet;
    // these tiny BF16 operations run only through existing manual Metal oracles.
    // The edit entry also reuses four tiny author forwards plus two training forwards.
    // The ordinary packet has no time-activation hook. Both graphs use nn.SiLU.
    let time_fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../contracts/gen-core/tests/fixtures/moss_image/tiny_edit_v2/fixture.json"
    ))
    .unwrap();
    for probe in time_fixture["time_probes"].as_array().unwrap() {
        let input = array(&probe["silu_input"])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let expected = &probe["silu"];
        for training in [false, true] {
            assert_eq!(
                delta(&ordinary_time_silu(&input, training).unwrap(), expected),
                0.0,
                "author BF16 time SiLU training={training}"
            );
        }
        for (label, eager) in [
            ("inference", mlx_rs::fast::silu_exact(&input).unwrap()),
            (
                "training",
                multiply(&input, sigmoid(&input).unwrap()).unwrap(),
            ),
        ] {
            assert!(
                delta(&eager, expected) > 0.0,
                "author time SiLU hook must discriminate eager BF16 {label} rounding"
            );
        }
    }
    let f = fixture("ordinary");
    let g = geometry(&f);
    let context = contexts(
        &f["inputs"]["encoder_hidden_states"],
        &f["inputs"]["encoder_attention_mask"],
        g.text,
    );
    let input = array(&f["inputs"]["hidden_states"]);
    let times: Vec<f32> = f["inputs"]["timestep"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    for case in f["cases"].as_array().unwrap() {
        let dtype = if case["dtype"] == "float32" {
            Dtype::Float32
        } else {
            Dtype::Bfloat16
        };
        let w = weights(&f["weights"], dtype);
        let mut model = Transformer::load(&w, g.clone()).unwrap();
        assert!(w.unused_keys().is_empty());
        assert!(context.iter().all(|c| c.dtype() == Dtype::Float32));
        if dtype == Dtype::Float32 {
            // Full FP32 must preserve supplied embedding precision until its F32 projection.
            let precise = model.text.forward(&context[0]).unwrap();
            let narrowed = model
                .text
                .forward(&context[0].as_dtype(Dtype::Bfloat16).unwrap())
                .unwrap();
            assert_eq!(precise.dtype(), Dtype::Float32);
            let difference = subtract(&precise, &narrowed)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap();
            eval([&difference]).unwrap();
            assert!(
                difference.item::<f32>() > 1e-6,
                "embedding precision probe does not discriminate narrowing"
            );
        }
        let skip: Vec<u32> = case["skip_layers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let output = model
            .forward(&input, &times, &context, &skip, &CancelFlag::new())
            .unwrap();
        let difference = delta(&output, &case["velocity"]);
        let tolerance = if dtype == Dtype::Float32 { 2e-5 } else { 1e-3 };
        assert!(
            difference < tolerance,
            "dtype={dtype:?} skip={skip:?} max_abs={difference}"
        );
        if dtype == Dtype::Bfloat16 {
            // Training mode must match the same frozen author packet,
            // including its fused SiLU boundary; no regenerated golden or wider tolerance.
            model.training = true;
            let training_output = model
                .forward(&input, &times, &context, &skip, &CancelFlag::new())
                .unwrap();
            let training_difference = delta(&training_output, &case["velocity"]);
            assert!(
                training_difference < tolerance,
                "training dtype={dtype:?} skip={skip:?} max_abs={training_difference}"
            );
        }
    }
    let hidden = array(&f["leaf_ops"]["situ_hidden"]);
    let gate = array(&f["leaf_ops"]["situ_gate"]);
    assert!(
        delta(
            &situ(&hidden, &gate).unwrap(),
            &f["leaf_ops"]["situ_expected"]
        ) < 1e-6
    );
    assert!(delta(&rope(2, 3, 3, 8).unwrap(), &f["leaf_ops"]["rope_2x3_text3"]) < 1e-6);
}
#[test]
#[ignore = "manual MOSS Metal model proof; CI lanes=moss-metal-fixture"]
fn frozen_reference_unmerged_fp32_projection_and_three_step_dmd() {
    #[cfg(feature = "test-support")]
    crate::test_support::assert_default_metal_profile_device();
    let f = fixture("dmd");
    let g = geometry(&f);
    let w = weights(&f["base_weights"], Dtype::Bfloat16);
    let mut model = Transformer::load(&w, g.clone()).unwrap();
    let mut adapters = HashMap::new();
    for (key, value) in f["adapters"].as_object().unwrap() {
        let p = format!("diffusion_model.{key}");
        adapters.insert(format!("{p}.lora_down.weight"), array(&value["down"]));
        adapters.insert(format!("{p}.lora_up.weight"), array(&value["up"]));
        adapters.insert(
            format!("{p}.alpha"),
            Array::from_slice(&[value["alpha"].as_f64().unwrap() as f32], &[]),
        );
    }
    model
        .install_dmd(
            &Weights::from_map(adapters),
            &weights(&f["replacement_tensors"], Dtype::Float32),
            3,
        )
        .unwrap();
    let probe = model.blocks[0].projections[3]
        .forward(&array(&f["projection_probe"]["input"]))
        .unwrap();
    assert_eq!(probe.dtype(), Dtype::Float32);
    assert!(delta(&probe, &f["projection_probe"]["correct"]) < 1e-3);
    let narrowed = probe
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let difference = subtract(&probe, &narrowed)
        .unwrap()
        .abs()
        .unwrap()
        .max(None)
        .unwrap()
        .item::<f32>();
    assert!(difference > 1e-4, "residual was narrowed");
    assert_eq!(model.time1.dtype(), Dtype::Float32);
    assert_eq!(model.time2.dtype(), Dtype::Float32);
    assert_eq!(model.modulation.dtype(), Dtype::Float32);
    assert_eq!(model.head.dtype(), Dtype::Float32);
    let noise = array(&f["noise"]);
    let q = array(&f["mix_matrix"]);
    let context = contexts(&f["conditioning"], &f["mask"], g.text);
    let mut clean = multiply(&noise, constant(0.0, Dtype::Float32).unwrap()).unwrap();
    let shape = noise.shape();
    for (i, code) in f["step_codes"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let flat = clean
            .transpose_axes(&[0, 2, 3, 1])
            .unwrap()
            .reshape(&[-1, g.channels])
            .unwrap();
        let mixed = linear_fp32(&flat, &q)
            .unwrap()
            .reshape(&[shape[0], shape[2], shape[3], g.channels])
            .unwrap()
            .transpose_axes(&[0, 3, 1, 2])
            .unwrap();
        let input = add(&noise, mixed).unwrap();
        let velocity = model
            .forward(
                &input,
                &[code.as_f64().unwrap() as f32],
                &context,
                &[],
                &CancelFlag::new(),
            )
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let expected = &f["runs"]["correct_widened"]["steps"][i];
        let velocity_error = delta(&velocity, &expected["velocity"]);
        assert!(velocity_error < 1e-3,
                "DMD step={i} velocity max_abs={velocity_error}; input max_abs={}; actual={:?}; expected={:?}",
                delta(&input, &expected["input"]), crate::pipeline::logical_f32_readback(&velocity).unwrap(), expected["velocity"]["values"]);
        clean = subtract(&noise, velocity).unwrap();
        eval([&clean]).unwrap();
        let clean_error = delta(&clean, &expected["x0"]);
        assert!(clean_error < 1e-3, "DMD step={i} x0 max_abs={clean_error}");
    }
}

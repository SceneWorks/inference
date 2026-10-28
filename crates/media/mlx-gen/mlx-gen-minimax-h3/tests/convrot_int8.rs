//! Tiny executable-Kitchen oracle, reached through the production H3 projection loader.
use mlx_gen::weights::Weights;
use mlx_gen_minimax_h3::quant;
use mlx_rs::{transforms::eval, Array, Dtype};
use serde_json::{json, Value};
use std::collections::HashMap;

fn fixture() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../../tests/fixtures/h3_convrot_int8.json"
    )))
    .unwrap()
}
fn floats(v: &Value) -> Vec<f32> {
    serde_json::from_value(v.clone()).unwrap()
}
fn operands(k: i32, n: i32, group: i32) -> HashMap<String, Array> {
    let codes: Vec<i8> = (0..n * k)
        .map(|i| (((i / k * 19 + i % k * 7) % 255) - 127) as i8)
        .collect();
    let scales: Vec<f32> = (0..n).map(|i| (i + 1) as f32 / 128.).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i - 8) as f32 / 32.).collect();
    let descriptor = serde_json::to_vec(
        &json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":group}),
    )
    .unwrap();
    HashMap::from([
        ("p.weight".into(), Array::from_slice(&codes, &[n, k])),
        ("p.weight_scale".into(), Array::from_slice(&scales, &[n, 1])),
        ("p.bias".into(), Array::from_slice(&bias, &[n])),
        (
            "p.comfy_quant".into(),
            Array::from_slice(&descriptor, &[descriptor.len() as i32]),
        ),
    ])
}

#[test]
fn malformed_metadata_is_not_dense_or_affine_q8() {
    for bad in [
        json!({"format":"int8_tensorwise","convrot":false,"convrot_groupsize":64}),
        json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}),
        json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64,"group_size":64}),
    ] {
        let mut map = operands(192, 16, 64);
        let blob = serde_json::to_vec(&bad).unwrap();
        map.insert(
            "p.comfy_quant".into(),
            Array::from_slice(&blob, &[blob.len() as i32]),
        );
        assert!(quant::lin(&Weights::from_map(map), "p", true).is_err());
    }
    let mut map = operands(192, 16, 64);
    map.remove("p.comfy_quant");
    assert!(quant::lin(&Weights::from_map(map), "p", true).is_err());
    let mut map = operands(192, 16, 64);
    map.insert("p.scales".into(), Array::from_slice(&[1f32; 48], &[16, 3]));
    assert!(quant::lin(&Weights::from_map(map), "p", true).is_err());
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        let mut map = operands(192, 16, 64);
        map.insert(
            "p.weight_scale".into(),
            Array::from_slice(&[scale; 16], &[16, 1]),
        );
        assert!(quant::lin(&Weights::from_map(map), "p", true).is_err());
    }
}

#[test]
#[ignore = "requires coordinator-reserved exclusive local Metal token; never skip-as-pass"]
fn native_projections_match_pinned_kitchen() {
    let fixture = fixture();
    assert_eq!(
        fixture["kitchen_revision"],
        "f8b7e97a1418d2e529bd5405fbf86312be198ae6"
    );
    assert_eq!(
        fixture["producer_revision"],
        "d6797787e6bdb1a1fb0094d588a26f8e71a1c757"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let (k, n, m, g) = (
            case["k"].as_i64().unwrap() as i32,
            case["n"].as_i64().unwrap() as i32,
            case["m"].as_i64().unwrap() as i32,
            case["group"].as_i64().unwrap() as i32,
        );
        let dtype = match case["dtype"].as_str().unwrap() {
            "f32" => Dtype::Float32,
            "bf16" => Dtype::Bfloat16,
            "f16" => Dtype::Float16,
            _ => panic!("fixture dtype"),
        };
        // Serialize and reload actual I8/F32/U8 operands via the production Weights reader.
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.safetensors");
        let parts = operands(k, n, g);
        Array::save_safetensors(parts.iter().map(|(k, v)| (k.as_str(), v)), None, &file).unwrap();
        let w = Weights::from_file(&file).unwrap();
        let mut linear = quant::lin(&w, "p", true).unwrap();
        assert!(linear.is_quantized());
        assert!(
            linear.convrot_base().is_some(),
            "must reach signed integer kernel, not affine Q8"
        );
        assert!(linear.dense_weight().is_none());
        assert!(linear.quantized_params().is_none());
        assert_eq!(linear.base_shape(), vec![n, k]);
        drop(w);
        let x = Array::from_slice(&floats(&case["input"]), &[m, k])
            .as_dtype(dtype)
            .unwrap();
        let out = linear
            .forward(&x)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        eval([&out]).unwrap();
        let expected = floats(&case["expected"]);
        for (i, (&got, &want)) in out.as_slice::<f32>().iter().zip(&expected).enumerate() {
            let tolerance = if dtype == Dtype::Float32 {
                2e-5 * want.abs().max(1.)
            } else {
                0.0
            };
            assert!(
                (got - want).abs() <= tolerance,
                "group{g} {dtype:?} output{i}: {got} != {want}"
            );
        }
        assert!(linear
            .merge_dense_delta(&Array::from_slice(&vec![0f32; (n * k) as usize], &[n, k]))
            .is_err());
        println!("native Metal group{g} {dtype:?} {m}x{k} -> {m}x{n}, oracle matched");
    }
}

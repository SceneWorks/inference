//! Native H3 loader -> shared Int8Linear/cuBLASLt, never a float fallback.
use candle_gen::{
    candle_core::{DType, Device},
    Weights,
};
use candle_gen_minimax_h3::quant::{lin, DIT};
use safetensors::{tensor::TensorView, Dtype};
use serde_json::{json, Value};

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
fn write_parts(
    path: &std::path::Path,
    k: usize,
    n: usize,
    g: usize,
    descriptor: Option<Value>,
    scale: f32,
) {
    let codes: Vec<u8> = (0..n * k)
        .map(|i| (((i / k * 19 + i % k * 7) % 255) as i32 - 127) as i8 as u8)
        .collect();
    let scales: Vec<u8> = (0..n)
        .flat_map(|i| ((i + 1) as f32 / 128. * scale).to_le_bytes())
        .collect();
    let bias: Vec<u8> = (0..n)
        .flat_map(|i| ((i as f32 - 8.) / 32.).to_le_bytes())
        .collect();
    let desc = serde_json::to_vec(&descriptor.unwrap_or_else(
        || json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":g}),
    ))
    .unwrap();
    let tensors = std::collections::HashMap::from([
        (
            "p.weight",
            TensorView::new(Dtype::I8, vec![n, k], &codes).unwrap(),
        ),
        (
            "p.weight_scale",
            TensorView::new(Dtype::F32, vec![n, 1], &scales).unwrap(),
        ),
        (
            "p.bias",
            TensorView::new(Dtype::F32, vec![n], &bias).unwrap(),
        ),
        (
            "p.comfy_quant",
            TensorView::new(Dtype::U8, vec![desc.len()], &desc).unwrap(),
        ),
    ]);
    safetensors::tensor::serialize_to_file(tensors, &None, path).unwrap();
}

#[test]
fn loader_preserves_signed_codes_and_f32_scales() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("int8.safetensors");
    write_parts(&file, 192, 16, 64, None, 1.);
    let w = Weights::from_files_native_int8(&[&file], &Device::Cpu).unwrap();
    let codes = w.require("p.weight").unwrap();
    assert_eq!(codes.dtype(), DType::U8);
    assert_eq!(codes.elem_count(), 16 * 192);
    assert_eq!(w.require("p.weight_scale").unwrap().dtype(), DType::F32);
    let bytes = codes.flatten_all().unwrap().to_vec1::<u8>().unwrap();
    assert_eq!(bytes[0] as i8, -127);
    assert_eq!(bytes[1] as i8, -120);
    // A CPU build must refuse native execution rather than dequantize and call it INT8.
    let err = lin(&w, DIT, "p", true, DType::BF16)
        .unwrap_err()
        .to_string();
    assert!(err.contains("CUDA") || err.contains("context"), "{err}");
}

#[test]
fn malformed_metadata_fails_before_backend_execution() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("bad.safetensors");
    for desc in [
        json!({"format":"int8_tensorwise","convrot":false,"convrot_groupsize":64}),
        json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}),
        json!({"format":"int8_tensorwise","convrot":true,"convrot_groupsize":64,"input_scale":1.0}),
    ] {
        write_parts(&file, 192, 16, 64, Some(desc), 1.);
        let err = Weights::from_files_native_int8(&[&file], &Device::Cpu)
            .err()
            .expect("invalid metadata must fail before loading tensors")
            .to_string();
        assert!(
            !err.contains("requires native CUDA"),
            "must reject descriptor first: {err}"
        );
    }
    for scale in [0., -1., f32::NAN, f32::INFINITY] {
        write_parts(&file, 192, 16, 64, None, scale);
        assert!(Weights::from_files_native_int8(&[&file], &Device::Cpu)
            .err()
            .unwrap()
            .to_string()
            .contains("finite positive"));
    }
}

#[test]
fn unsigned_storage_cannot_masquerade_as_symmetric_int8() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("unsigned.safetensors");
    write_parts(&file, 192, 16, 64, None, 1.);
    let mut bytes = std::fs::read(&file).unwrap();
    let at = bytes.windows(4).position(|s| s == b"\"I8\"").unwrap();
    bytes[at + 1] = b'U';
    std::fs::write(&file, bytes).unwrap();
    assert!(Weights::from_files_native_int8(&[&file], &Device::Cpu)
        .err()
        .unwrap()
        .to_string()
        .contains("original I8"));
}

#[test]
fn companions_may_cross_shards_but_duplicates_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("original.safetensors");
    let codes = dir.path().join("codes.safetensors");
    let companions = dir.path().join("companions.safetensors");
    write_parts(&original, 192, 16, 64, None, 1.);
    let bytes = std::fs::read(&original).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    safetensors::tensor::serialize_to_file(
        st.tensors().into_iter().filter(|(k, _)| k == "p.weight"),
        &None,
        &codes,
    )
    .unwrap();
    safetensors::tensor::serialize_to_file(
        st.tensors().into_iter().filter(|(k, _)| k != "p.weight"),
        &None,
        &companions,
    )
    .unwrap();
    let w = Weights::from_files_native_int8(&[&codes, &companions], &Device::Cpu).unwrap();
    assert_eq!(w.require("p.weight").unwrap().dtype(), DType::U8);
    assert!(
        Weights::from_files_native_int8(&[&codes, &companions, &codes], &Device::Cpu,)
            .err()
            .unwrap()
            .to_string()
            .contains("duplicate tensor key")
    );
}

#[test]
fn host_contract_matches_pinned_executable_oracle() {
    use candle_gen::candle_core::Tensor;
    use candle_gen::quant::{convrot_rotate, regular_hadamard};
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
        let (k, m, g) = (
            case["k"].as_u64().unwrap() as usize,
            case["m"].as_u64().unwrap() as usize,
            case["group"].as_u64().unwrap() as usize,
        );
        let round = |v: f32| match case["dtype"].as_str().unwrap() {
            "bf16" => half::bf16::from_f32(v).to_f32(),
            "f16" => half::f16::from_f32(v).to_f32(),
            _ => v,
        };
        let x = Tensor::from_vec(floats(&case["input"]), (m, k), &Device::Cpu).unwrap();
        let rotated = convrot_rotate(&x, &regular_hadamard(g, &Device::Cpu).unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let rotated: Vec<f32> = rotated.into_iter().map(round).collect();
        let expected_codes: Vec<i32> =
            serde_json::from_value(case["activation_codes"].clone()).unwrap();
        let expected_scales = floats(&case["activation_scales"]);
        for (row, values) in rotated.chunks_exact(k).enumerate() {
            let scale =
                (values.iter().map(|v| v.abs()).fold(0f32, f32::max) * (1f32 / 127.)).max(1e-30);
            assert_eq!(scale, expected_scales[row]);
            for (j, &value) in values.iter().enumerate() {
                let z = round(value / round(scale));
                let code = if z.is_nan() {
                    -128
                } else {
                    z.round_ties_even().clamp(-128., 127.) as i32
                };
                assert_eq!(
                    code,
                    expected_codes[row * k + j],
                    "group{g} {} row{row} col{j}",
                    case["dtype"]
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires coordinator-reserved CUDA device; absence is failure, not skip"]
fn native_projections_match_pinned_kitchen() {
    use candle_gen::candle_core::Tensor;
    let device = Device::new_cuda(0).expect("reserved CUDA runner must provide a real device");
    let fixture = fixture();
    assert_eq!(
        fixture["kitchen_revision"],
        "f8b7e97a1418d2e529bd5405fbf86312be198ae6"
    );
    for case in fixture["cases"].as_array().unwrap() {
        let (k, n, m, g) = (
            case["k"].as_u64().unwrap() as usize,
            case["n"].as_u64().unwrap() as usize,
            case["m"].as_u64().unwrap() as usize,
            case["group"].as_u64().unwrap() as usize,
        );
        let dtype = match case["dtype"].as_str().unwrap() {
            "f32" => DType::F32,
            "bf16" => DType::BF16,
            "f16" => DType::F16,
            _ => panic!("fixture dtype"),
        };
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("weights.safetensors");
        write_parts(&file, k, n, g, None, 1.);
        let headers =
            candle_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(&file).unwrap();
        let bias_header = headers
            .iter()
            .find(|header| header.name == "p.bias")
            .unwrap();
        assert_eq!(
            bias_header.dtype,
            candle_gen::gen_core::weightsmeta::Dtype::F32
        );
        assert_eq!(bias_header.shape, [n]);
        let expected_bias_bytes =
            bias_header.shape.iter().product::<usize>() * bias_header.dtype.size();
        let w = Weights::from_files_native_int8(&[&file], &device).unwrap();
        let mut loaded = lin(&w, DIT, "p", true, dtype).unwrap();
        assert!(loaded.linear.is_convrot_int8());
        assert!(loaded.linear.is_packed());
        let stored_bias = w.require("p.bias").unwrap();
        assert_eq!(
            stored_bias.dtype(),
            DType::F32,
            "the fixture stores its bias F32 regardless of activation dtype"
        );
        assert_eq!(loaded.base_bytes, n * k + n * 4 + expected_bias_bytes);
        assert_eq!(loaded.linear.base_shape(), (n, k));
        drop(w);
        let x = Tensor::from_vec(floats(&case["input"]), (m, k), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let out = loaded
            .linear
            .forward(&x)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (&got, &want)) in out.iter().zip(floats(&case["expected"]).iter()).enumerate() {
            let tolerance = if dtype == DType::F32 {
                2e-5 * want.abs().max(1.)
            } else {
                0.
            };
            assert!(
                (got - want).abs() <= tolerance,
                "group{g} {dtype:?} output{i}: {got} != {want}"
            );
        }
        assert!(loaded
            .linear
            .quantize(candle_gen::gen_core::Quant::Q8)
            .is_err());
        println!("native CUDA group{g} {dtype:?} {m}x{k} -> {m}x{n}, oracle matched");
    }
}
